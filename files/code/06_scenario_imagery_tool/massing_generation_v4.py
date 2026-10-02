"""Give the model the building solid; let it cast the shadow from the scene's own light.

Measuring the sun failed its own boundary test. The darkness score rewards short shadows,
because a short shadow hugs the building where dark pixels always sit -- facade shade, wall
base. Raising the search ceiling from 76 to 87 degrees moved the estimate from 75.0 to
85.0, both exactly at the bound, and dragged the azimuth from 35 to 50 with it. Melbourne
cannot exceed 75.6 degrees, so the estimator was reporting its own ceiling. Split-half
agreement had looked convincing, but reproducibility only proves a bias is stable.

So the massing now carries the building solid alone -- roof plane and visible facades,
which v3 showed the model does follow -- and the shadow is left to the model, which can
read the light off dozens of real buildings in the same frame.

The ROI still has to be large enough to contain a shadow whose direction we no longer
predict. One hard constraint survives: Melbourne sits at -37.8 degrees, well outside the
tropics, so the sun is north of zenith all year and shadows always fall in the southern
half. The ROI therefore sweeps a half-disc southwards rather than a full disc.

    python outputs/phase2_offnadir_masking_v1/massing_generation_v4.py
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from phase2_large_unet_build_data import rasterize_footprint, stitch_patch  # noqa: E402

# Direct to OpenAI by default. The relay sat behind a proxy that hung up after sixty seconds
# of silence, and these requests measure 65-80s against the official endpoint -- so the work
# was finishing and the connection was being cut before it could be handed back. That is why
# some sites failed every single attempt while others always passed: the gate ran straight
# through the middle of this task's duration. Set OPENAI_BASE_URL to go back through a relay.
EDITS_URL = os.environ.get(
    "OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/images/edits"
MODEL = "gpt-image-2"
EXTERNAL = PROJECT / "outputs/phase2_large_unet_v1/data/phase2_large_unet_external_test_manifest.csv"
PRETRAIN = PROJECT / "outputs/phase2_large_unet_v1/data/phase2_large_unet_pretrain_manifest.csv"
OUT_ROOT = Path(__file__).resolve().parent / "massing_pilot_v6"
RENDER = 1024
QUALITY = "medium"

LEAN_DEGREES = 15.0
LEAN_AZIMUTH_DEGREES = 315.0
MIN_SUN_ELEVATION_DEGREES = 45.0   # shadow budget only; a lower bound costs ROI area

PROMPT = """Render the supplied grey massing block as it would appear in this satellite photograph.

The first image is an overhead satellite view of a site in Melbourne with the development
parcel erased. The second image is a geometric massing study of the approved building at
the correct scale, position and viewing lean: light grey is the roof plane, mid grey are
the visible facades. It deliberately carries no shadow.

Reproduce that geometry exactly -- same roof outline, same facade faces, same footprint
position -- and give it photographic substance: roof plant and services, parapets, glazing
and material on the facades, and the grain, sharpness, colour balance and haze of the
surrounding photograph.

Then cast its shadow yourself. Read the lighting off the real buildings already in the
frame: every existing shadow runs the same direction, and its length relative to its
building's height tells you the sun elevation. The new building is {height:.0f} m tall --
compare it with the neighbours whose shadows you can see, and give it a shadow of matching
direction and proportionate length. Getting the shadow consistent with the rest of the
scene matters more than any other single cue.

The building is about {storeys} storeys, footprint {area:.0f} square metres, land use
{use}.

Everything outside the erased region stays pixel-identical. Add no outline, label or
annotation anywhere."""


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def lean_vector(height_m: float) -> np.ndarray:
    lean = height_m * math.tan(math.radians(LEAN_DEGREES))
    return np.array([lean * math.cos(math.radians(LEAN_AZIMUTH_DEGREES)),
                     lean * math.sin(math.radians(LEAN_AZIMUTH_DEGREES))])


def shadow_budget_m(height_m: float) -> float:
    return height_m / math.tan(math.radians(MIN_SUN_ELEVATION_DEGREES))


def southern_half_disc(radius_px: int) -> np.ndarray:
    """Structuring element covering every direction a Melbourne shadow can fall.

    The half kept here is the northern one, which looks wrong and is not. cv2.dilate reads
    the source at a positive offset, dst(p) = max over k of src(p + k), so it reflects the
    element: an element weighted north grows the mask south. Filling the southern half
    instead grew every editable region away from the shadow, and the blend then discarded
    whatever shadow the model had painted.
    """
    size = 2 * radius_px + 1
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (((xx ** 2 + yy ** 2) <= radius_px ** 2) & (yy <= 0)).astype(np.uint8)


def shadow_region(solid: np.ndarray, radius_px: int, factor: int = 4) -> np.ndarray:
    """The same region, built on a coarser grid because it costs seconds to build exactly.

    A tall tower on a fine patch needs an element most of a thousand pixels across, which
    OpenCV has no fast path for -- close to seven seconds for one call, which is fine in a
    batch run and not fine behind a button. The region is a bound rather than a measurement:
    it assumes the sun never rises above MIN_SUN_ELEVATION_DEGREES and then adds slack on
    top. Resolving its edge to the pixel is precision the quantity does not have.

    Rounding the coarse radius up keeps the result a superset of the exact one to within a
    fraction of a metre, and the massing is unioned back in at full resolution so nothing
    against the building itself is lost to the resampling.
    """
    small = cv2.resize(solid.astype(np.uint8), None, fx=1 / factor, fy=1 / factor,
                       interpolation=cv2.INTER_AREA) > 0
    grown = cv2.dilate(small.astype(np.uint8),
                       southern_half_disc(int(math.ceil(radius_px / factor)) + 1))
    upsampled = cv2.resize(grown, (solid.shape[1], solid.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
    return upsampled | solid.astype(bool)


def plan_patch(footprint_width_m: float, height_m: float,
               context_ratio: float) -> tuple[float, np.ndarray]:
    lean = lean_vector(height_m)
    shadow = shadow_budget_m(height_m)
    half = footprint_width_m / 2.0
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    # shadow may fall anywhere in the southern half: east, south, or west of the footprint
    southern = np.vstack([corners + [shadow, 0], corners + [0, shadow], corners + [-shadow, 0]])
    points = np.vstack([corners, corners + lean, southern])
    low, high = points.min(axis=0), points.max(axis=0)
    return float(max(high - low)) * context_ratio, (low + high) / 2.0


def shift_lonlat(longitude: float, latitude: float, offset_m: np.ndarray) -> tuple[float, float]:
    east, south = float(offset_m[0]), float(offset_m[1])
    return (longitude + east / (111_320.0 * math.cos(math.radians(latitude))),
            latitude - south / 110_540.0)


def polygons_of(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 3]


def render_massing(mask: np.ndarray, height_m: float, metres_per_px: float) -> np.ndarray:
    """Building solid only: roof plane plus the facades the lean exposes. No shadow."""
    lean = lean_vector(height_m) / metres_per_px
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for polygon in polygons_of(mask):
        top = polygon + lean
        for index in range(len(polygon)):
            nxt = (index + 1) % len(polygon)
            cv2.fillPoly(canvas, [np.array([polygon[index], polygon[nxt], top[nxt], top[index]],
                                           dtype=np.int32)], (120, 120, 125))
        cv2.fillPoly(canvas, [top.astype(np.int32)], (205, 205, 208))
    return canvas


def call(scene: Image.Image, massing: Image.Image, roi: np.ndarray, prompt: str,
         retries: int = 4, timeout: int = 900) -> tuple[Image.Image | None, str]:
    alpha = np.where(roi, 0, 255).astype(np.uint8)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    files = [("image[]", ("scene.png", png_bytes(scene), "image/png")),
             ("image[]", ("massing.png", png_bytes(massing), "image/png")),
             ("mask", ("mask.png", png_bytes(Image.fromarray(rgba)), "image/png"))]
    data = {"model": MODEL, "prompt": prompt, "n": "1", "size": f"{RENDER}x{RENDER}",
            "quality": QUALITY}
    last = ""
    for attempt in range(retries):
        try:
            response = requests.post(
                EDITS_URL, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                files=files, data=data, timeout=timeout)
            if response.status_code == 200:
                entry = response.json().get("data", [{}])[0]
                if entry.get("b64_json"):
                    return Image.open(
                        BytesIO(base64.b64decode(entry["b64_json"]))).convert("RGB"), ""
                last = f"no b64_json: {list(entry.keys())}"
            else:
                last = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = type(exc).__name__
        time.sleep(8 * (attempt + 1))
    return None, last


def race(make_call, lanes: int = 3, log=None):
    """Fire identical requests together and keep the first one the proxy lets through.

    Only needed in front of the relay: its proxy hangs up after sixty seconds of silence
    while these requests measure sixty-five to eighty against the official endpoint, so the
    model finishes and the connection is cut before the result can come back. Each request
    is then a coin toss costing a full minute when it loses; sending several at once pays
    that minute once. On a payload that had failed five times out of five in series, two
    rounds of three lanes both won, at fifty-two and fifty-five seconds.

    Three, not more. A five-lane round lost every lane: the generations share upstream
    capacity, so piling them up slows each one down and pushes them all over the line.
    """
    import threading as _threading

    results: list = [None] * lanes

    def lane(index: int) -> None:
        try:
            results[index] = make_call()
        except Exception as exc:  # noqa: BLE001
            results[index] = (None, type(exc).__name__)

    threads = [_threading.Thread(target=lane, args=(i,), daemon=True) for i in range(lanes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    errors = []
    for index, entry in enumerate(results):
        if entry and entry[0] is not None:
            if log:
                log(f"lane {index + 1} of {lanes} came back first")
            return entry
        errors.append(entry[1] if entry else "no result")
    if log:
        log(f"all {lanes} lanes lost: {', '.join(errors)}")
    return None, errors[0] if errors else "no result"


def blend_into_scene(scene: Image.Image, generated: Image.Image,
                     roi: np.ndarray, feather_px: int = 24) -> np.ndarray:
    original = np.asarray(scene, dtype=np.float32)
    painted = np.asarray(generated.resize(scene.size, Image.Resampling.LANCZOS),
                         dtype=np.float32)
    # colour-match the painted region to what the scene looked like there
    for channel in range(3):
        source, target = painted[..., channel][roi], original[..., channel][roi]
        if source.size < 16:
            continue
        scale = (target.std() + 1e-6) / (source.std() + 1e-6)
        painted[..., channel][roi] = (source - source.mean()) * scale + target.mean()
    # feather the ROI edge; the taper lives inside the ROI so outside stays untouched
    distance = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance / max(feather_px, 1), 0.0, 1.0)[..., None]
    merged = original * (1.0 - alpha) + np.clip(painted, 0, 255) * alpha
    return merged.astype(np.uint8)


def watermark(image: Image.Image) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    try:
        font = ImageFont.truetype("arial.ttf", max(marked.size[0] // 40, 11))
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, marked.size[1] - 30, marked.size[0], marked.size[1]], fill=(0, 0, 0))
    draw.text((5, marked.size[1] - 26), "SYNTHETIC - AI GENERATED", fill=(255, 90, 90), font=font)
    return marked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--developments", nargs="+", default=["X0008015", "X0011004"])
    parser.add_argument("--heights", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--context-ratio", type=float, default=1.8)
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    external = pd.read_csv(EXTERNAL).drop_duplicates("development_key")
    templates = pd.read_csv(PRETRAIN).groupby("target_year").imagery_url_template.first().to_dict()
    records = []

    for key in args.developments:
        rows = external[external.development_key.astype(str) == key]
        if rows.empty:
            continue
        row = rows.iloc[0]
        before_year = [y for y in sorted(templates) if y < int(row.completion_year)][-1]
        sample_dir = OUT_ROOT / key
        sample_dir.mkdir(parents=True, exist_ok=True)
        base_height = float(row.metadata_height_m)
        footprint_width = math.sqrt(max(float(row.footprint_area_m2), 1.0))
        geometry = json.loads(str(row.geometry_geojson))

        tallest = base_height * max(args.heights)
        patch_m, centre = plan_patch(footprint_width, tallest, args.context_ratio)
        longitude, latitude = shift_lonlat(float(row.centroid_longitude),
                                           float(row.centroid_latitude), centre)
        metres_per_px = patch_m / RENDER
        scene = stitch_patch(longitude=longitude, latitude=latitude, patch_size_m=patch_m,
                             output_size_px=RENDER, zoom=18,
                             url_template=templates[before_year], timeout=90)
        footprint = rasterize_footprint(geometry, latitude=latitude, longitude=longitude,
                                        patch_size_m=patch_m, shape_hw=(RENDER, RENDER))
        scene.save(sample_dir / "scene.png")
        print(f"\n{key}: frame {patch_m:.0f} m ({metres_per_px:.2f} m/px), "
              f"tallest case {tallest:.0f} m", flush=True)

        for factor in args.heights:
            height = base_height * factor
            massing = render_massing(footprint, height, metres_per_px)
            radius = int(math.ceil(shadow_budget_m(height) / metres_per_px)) + 8
            roi = cv2.dilate(massing.any(axis=2).astype(np.uint8),
                             southern_half_disc(radius)).astype(bool)
            roi_fraction = float(roi.mean())
            Image.fromarray(massing).save(sample_dir / f"massing_x{factor:g}_{height:.0f}m.png")

            prompt = PROMPT.format(height=height, storeys=max(int(round(height / 3.2)), 1),
                                   area=float(row.footprint_area_m2),
                                   use="mixed use" if float(row.metadata_commercial_m2 or 0) > 0
                                   else "residential")
            print(f"  x{factor:g} = {height:.0f} m | shadow budget "
                  f"{shadow_budget_m(height):.0f} m | ROI {roi_fraction:.1%}", flush=True)
            generated, error = call(scene, Image.fromarray(massing), roi, prompt)
            if generated is None:
                print(f"    FAILED: {error}", flush=True)
                records.append({"development_key": key, "factor": factor, "height_m": height,
                                "status": "failed", "error": error})
                continue
            merged = blend_into_scene(scene, generated, roi)
            watermark(Image.fromarray(merged)).save(
                sample_dir / f"generated_x{factor:g}_{height:.0f}m.png")
            outside = float(np.abs(merged[~roi].astype(np.float32)
                                   - np.asarray(scene, np.float32)[~roi]).mean())
            print(f"    ok | outside delta {outside:.6f}", flush=True)
            records.append({"development_key": key, "factor": factor, "height_m": height,
                            "status": "ok", "patch_m": round(patch_m, 1),
                            "metres_per_px": round(metres_per_px, 3),
                            "roi_fraction": round(roi_fraction, 4),
                            "outside_roi_delta": outside})

    (OUT_ROOT / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n{len(ok)}/{len(records)} succeeded; locality preserved on "
          f"{sum(r['outside_roi_delta'] == 0 for r in ok)}/{len(ok)}")
    print(f"-> {OUT_ROOT}")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set")
    main()
