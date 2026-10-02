"""Draw the extruded massing into the photograph, so position and height are both encoded.

Two fixes were made separately and each broke what the other had solved.

Passing the massing as a second image left the model to guess which part of the satellite
frame it referred to, and it guessed badly: a keyhole footprint came back as a round tower,
and three arms of the same site put the building in three different places. Compositing a
flat grey block into the photograph fixed that -- placement error halved, 7.2 m median
against 13.5 m -- but the block carries no height. Height fell back to a sentence of
prompt text, which is exactly the weak conditioning the massing route existed to replace,
and the 0.5x/1x/2x sweep that answered the supervisor's three-storeys-to-six question was
built on the extruded block, not a flat one.

So extrude the block and composite that. Roof plane and visible facades go into the
photograph at the true position, so height changes the image by construction while the
footprint stays pinned.

The lean is nominal, not measured -- both attempts to measure it failed their own checks.
It is kept small and its only job is to make facades visible so height reads; the model is
still asked to reconcile the lean with the neighbours.

    python outputs/phase2_offnadir_masking_v1/inline_solid_massing.py --height-sweep 0.5 1.0 2.0
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

import massing_generation_v4 as base  # noqa: E402


def call_single(image, roi, prompt, retries: int = 6, timeout: int = 1200, log=None):
    """One image in, so the payload is half what base.call would send.

    `log` is called with a line per failed attempt. Without it a call that quietly retried
    five times is indistinguishable from one slow call, which made a two hundred second
    round trip impossible to account for.
    """
    import base64 as _b64
    import time as _time
    from io import BytesIO as _BytesIO

    import requests as _requests

    alpha = np.where(roi, 0, 255).astype(np.uint8)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    files = [("image", ("scene.png", base.png_bytes(image), "image/png")),
             ("mask", ("mask.png", base.png_bytes(Image.fromarray(rgba)), "image/png"))]
    data = {"model": base.MODEL, "prompt": prompt, "n": "1",
            "size": f"{base.RENDER}x{base.RENDER}", "quality": base.QUALITY}
    last = ""
    for attempt in range(retries):
        started = _time.time()
        try:
            response = _requests.post(
                base.EDITS_URL,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                files=files, data=data, timeout=timeout)
            if response.status_code == 200:
                entry = response.json().get("data", [{}])[0]
                if entry.get("b64_json"):
                    return Image.open(
                        _BytesIO(_b64.b64decode(entry["b64_json"]))).convert("RGB"), ""
                last = f"no b64_json: {list(entry.keys())}"
            else:
                # keep a slice of the body: a 504 from the proxy and a 503 from the relay
                # saying it has no channel free look identical as bare status codes, and
                # today that cost a round of extra requests to tell apart
                detail = response.text.strip().replace("\n", " ")[:160]
                last = f"HTTP {response.status_code}" + (f" {detail}" if detail else "")
        except Exception as exc:  # noqa: BLE001
            last = type(exc).__name__
        # A short flat pause, not a growing one. Backing off exponentially is the right
        # answer to being rate limited and the wrong answer to what actually happens here:
        # the relay gives up at about sixty seconds and our calls take forty to seventy, so
        # a failure has already cost a minute and says nothing about how busy anything is.
        # Four consecutive failures were seen with waits of 10, 20, 30 and 40s between them,
        # so the waiting bought nothing while adding a hundred seconds to the wait.
        if log:
            log(f"attempt {attempt + 1}/{retries} failed after "
                f"{_time.time() - started:.0f}s: {last}; retrying in 5s")
        _time.sleep(5)
    return None, last
from phase2_large_unet_build_data import rasterize_footprint, stitch_patch  # noqa: E402

EXTERNAL = PROJECT / "outputs/phase2_large_unet_v1/data/phase2_large_unet_external_test_manifest.csv"
PRETRAIN = PROJECT / "outputs/phase2_large_unet_v1/data/phase2_large_unet_pretrain_manifest.csv"
OUT_ROOT = HERE / "inline_solid"
RENDER = base.RENDER

NOMINAL_LEAN_DEGREES = 6.0     # only to expose facades; not a measurement
NOMINAL_LEAN_AZIMUTH = 315.0
ROOF_GREY, FACADE_GREY = 205, 120

PROMPT = """This is an overhead satellite photograph of a site in Melbourne. A grey massing
study of an approved building has been painted onto it, at the exact position and scale it
will occupy: the lighter grey is the roof, the darker grey are the facades below it.

Turn that grey solid into a photographic building. Its ground position and its outline are
fixed -- do not move it, rotate it, or simplify its shape. Its height is {height:.0f} m,
about {storeys} storeys, and the block already shows that height: the taller the building,
the more facade the block exposes.

Blend it into the photograph rather than pasting it. Two things must match the rest of the
scene, and both should be read off the real buildings in frame rather than assumed:

- Viewing geometry, which is the satellite's angle and not the sun's. Check how the
  neighbouring buildings lean -- whether you see their facades and towards which side. The
  block's lean is only a guide; adjust it to agree with the neighbours, keeping the ground
  footprint exactly where it is.
- Lighting. Every shadow in the frame runs the same way, and its length relative to its
  building's height gives the sun elevation. Cast this building a shadow of matching
  direction and proportionate length.

Give it photographic substance: roof plant and services, parapets, glazing and material on
the facades, and the grain, sharpness, colour balance and haze of the surrounding
photograph. The edges where it meets the ground, neighbouring roofs and street should look
photographed, not cut out. Footprint {area:.0f} square metres, land use {use}.

Everything outside the editable region stays pixel-identical. Add no outline, label or
annotation anywhere."""


def paint_massing(scene: np.ndarray, footprint: np.ndarray, height_m: float,
                  metres_per_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Composite roof plane and facades into the photograph. Returns image and solid mask."""
    lean = height_m * math.tan(math.radians(NOMINAL_LEAN_DEGREES)) / metres_per_px
    offset = np.array([lean * math.cos(math.radians(NOMINAL_LEAN_AZIMUTH)),
                       lean * math.sin(math.radians(NOMINAL_LEAN_AZIMUTH))])
    painted = scene.copy()
    solid = np.zeros(footprint.shape, dtype=np.uint8)
    for polygon in base.polygons_of(footprint):
        top = polygon + offset
        for index in range(len(polygon)):
            nxt = (index + 1) % len(polygon)
            quad = np.array([polygon[index], polygon[nxt], top[nxt], top[index]],
                            dtype=np.int32)
            cv2.fillPoly(painted, [quad], (FACADE_GREY,) * 3)
            cv2.fillPoly(solid, [quad], 1)
        cv2.fillPoly(painted, [top.astype(np.int32)], (ROOF_GREY,) * 3)
        cv2.fillPoly(solid, [top.astype(np.int32)], 1)
    return painted, solid.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--developments", nargs="+",
                        default=["X0012807", "X000619", "X001112"])
    parser.add_argument("--height-sweep", nargs="*", type=float, default=[])
    parser.add_argument("--context-ratio", type=float, default=1.8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--pace-seconds", type=float, default=30.0,
                        help="wait between calls; the relay rate limits bursts")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    external = pd.read_csv(EXTERNAL).drop_duplicates("development_key")
    templates = pd.read_csv(PRETRAIN).groupby("target_year").imagery_url_template.first().to_dict()
    records = []

    for key in args.developments:
        rows = external[external.development_key.astype(str) == key]
        if rows.empty:
            print(f"skip {key}: not in external test")
            continue
        row = rows.iloc[0]
        base_height = float(row.metadata_height_m)
        width = math.sqrt(max(float(row.footprint_area_m2), 1.0))
        before_year = [y for y in sorted(templates) if y < int(row.completion_year)][-1]
        factors = args.height_sweep or [1.0]

        # one framing for the whole sweep, sized by the tallest case, so scale never
        # confounds the height comparison
        patch_m, centre = base.plan_patch(width, base_height * max(factors),
                                          args.context_ratio)
        longitude, latitude = base.shift_lonlat(float(row.centroid_longitude),
                                                float(row.centroid_latitude), centre)
        metres_per_px = patch_m / RENDER
        sample_dir = OUT_ROOT / key
        sample_dir.mkdir(parents=True, exist_ok=True)

        scene = stitch_patch(longitude=longitude, latitude=latitude, patch_size_m=patch_m,
                             output_size_px=RENDER, zoom=18,
                             url_template=templates[before_year], timeout=90)
        footprint = rasterize_footprint(json.loads(str(row.geometry_geojson)),
                                        latitude=latitude, longitude=longitude,
                                        patch_size_m=patch_m, shape_hw=(RENDER, RENDER))
        if footprint.sum() < 64:
            print(f"skip {key}: footprint too small at this framing")
            continue
        scene.save(sample_dir / "scene.png")
        scene_array = np.asarray(scene)
        print(f"\n{key}: frame {patch_m:.0f} m ({metres_per_px:.2f} m/px), "
              f"base height {base_height:.0f} m", flush=True)

        for factor in factors:
            height = base_height * factor
            tag = f"x{factor:g}_{height:.0f}m"
            target = sample_dir / f"generated_{tag}.png"
            if args.resume and target.is_file():
                print(f"  {tag} already done, skipping", flush=True)
                continue

            painted, solid = paint_massing(scene_array, footprint, height, metres_per_px)
            radius = int(math.ceil(base.shadow_budget_m(height) / metres_per_px)) + 8
            roi = cv2.dilate(solid.astype(np.uint8),
                             base.southern_half_disc(radius)).astype(bool)
            guide = Image.fromarray(painted)
            guide.save(sample_dir / f"massing_{tag}.png")

            prompt = PROMPT.format(height=height, storeys=max(int(round(height / 3.2)), 1),
                                   area=float(row.footprint_area_m2),
                                   use="mixed use" if float(row.metadata_commercial_m2 or 0) > 0
                                   else "residential")
            print(f"  {tag} | facade {height * math.tan(math.radians(NOMINAL_LEAN_DEGREES)):.1f} m"
                  f" | ROI {roi.mean():.1%}", flush=True)
            if records:
                time.sleep(args.pace_seconds)
            generated, error = call_single(guide, roi, prompt)
            if generated is None:
                print(f"    FAILED: {error}", flush=True)
                records.append({"development_key": key, "factor": factor, "status": "failed"})
                continue
            merged = base.blend_into_scene(scene, generated, roi)
            base.watermark(Image.fromarray(merged)).save(target)
            outside = float(np.abs(merged.astype(np.float32)
                                   - scene_array.astype(np.float32))[~roi].mean())
            print(f"    ok | outside {outside:.6f}", flush=True)
            records.append({"development_key": key, "factor": factor, "height_m": height,
                            "status": "ok", "patch_m": round(patch_m, 1),
                            "metres_per_px": round(metres_per_px, 3),
                            "roi_fraction": round(float(roi.mean()), 4),
                            "outside_roi_delta": outside})

    (OUT_ROOT / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    ok = sum(r["status"] == "ok" for r in records)
    print(f"\n{ok}/{len(records)} succeeded  ->  {OUT_ROOT}")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set")
    main()
