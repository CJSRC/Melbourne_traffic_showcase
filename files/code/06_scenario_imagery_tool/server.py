"""Local service behind the three-panel tool.

Six attempts to measure the satellite's viewing lean all failed, and leaving it to the model
costs the one thing the supervisor cares about: two runs from identical input need not
agree on how tall the building reads. Putting the angle on a slider sidesteps both. The
operator turns the massing until it sits the way the neighbouring towers do, which is the
same judgement they make by eye in seconds, and the geometry that reaches the model is then
fixed and reproducible.

The browser cannot call the relay itself -- cross-origin rules block it, and shipping the
key to the page would expose it. So this sits in between: it fetches imagery, holds the
footprint library, paints the massing, and is the only thing that ever sees the key.

    set OPENAI_API_KEY, then:
    python outputs/phase2_offnadir_masking_v1/webapp/server.py
    open http://127.0.0.1:8000
"""

import base64
import csv
import io
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE.parent))

import massing_generation_v4 as base  # noqa: E402
from inline_solid_massing import FACADE_GREY, ROOF_GREY, call_single  # noqa: E402
from prompts import GROUNDS, MODES  # noqa: E402
import phase2_large_unet_build_data as build_data  # noqa: E402
from phase2_large_unet_build_data import stitch_patch  # noqa: E402
from validate_phase2_real_building_events import iter_polygons  # noqa: E402

EXTERNAL = PROJECT / "outputs/phase2_large_unet_v1/data/phase2_large_unet_external_test_manifest.csv"
PRETRAIN = PROJECT / "outputs/phase2_large_unet_v1/data/phase2_large_unet_pretrain_manifest.csv"

app = Flask(__name__, static_folder=str(HERE), static_url_path="")

# Two greys eighty-five levels apart told roof from wall by brightness alone, and brightness
# is the one channel a photograph is already full of -- a pale roof and a shadowed wall are
# ordinary things in this imagery. Separating them by hue instead, on the axis blue/orange,
# leaves no reading in which a face could be mistaken for the ground it sits on. Neither
# colour occurs at this saturation in Melbourne aerial imagery, so the block cannot be read
# as something already there.
ROOF_COLOUR = (105, 195, 225)      # light blue, the plane facing the sky
WALL_COLOUR = (200, 110, 70)       # burnt orange, the faces the lean exposes

# Four ways of drawing the same prism, so which one the model reads best can be settled by
# looking rather than by argument. The colour words travel with the style: the prompt names
# the faces by colour, so a grey block described as blue and orange would be worse than no
# description at all.
STYLES = {
    "colour": {"roof": ROOF_COLOUR, "wall": WALL_COLOUR, "shading": 0.28, "edges": True,
               "roof_word": "blue", "wall_word": "orange",
               "label": "colour + edges"},
    "grey": {"roof": (205, 205, 205), "wall": (120, 120, 120), "shading": 0.0,
             "edges": False, "roof_word": "light grey", "wall_word": "dark grey",
             "label": "all grey, as it began"},
    "colour_noedge": {"roof": ROOF_COLOUR, "wall": WALL_COLOUR, "shading": 0.28,
                      "edges": False, "roof_word": "blue", "wall_word": "orange",
                      "label": "colour, no edges"},
    "yellow_noedge": {"roof": ROOF_COLOUR, "wall": (230, 195, 60), "shading": 0.28,
                      "edges": False, "roof_word": "blue", "wall_word": "yellow",
                      "label": "yellow walls, no edges"},
    # Orange is the weakest colour in the set for this job: terracotta tiles, brick and
    # rusted metal roofs are all orange from above in Melbourne, so an orange face has a
    # ready reading as a roof. Magenta has no counterpart in aerial imagery at all.
    "magenta": {"roof": ROOF_COLOUR, "wall": (225, 60, 190), "shading": 0.28, "edges": True,
                "roof_word": "blue", "wall_word": "magenta",
                "label": "blue roof, magenta walls"},
    "magenta_green": {"roof": (60, 235, 120), "wall": (225, 60, 190), "shading": 0.28,
                      "edges": True, "roof_word": "green", "wall_word": "magenta",
                      "label": "green roof, magenta walls"},
    # In a real satellite photograph a facade seen obliquely is darker than the roof above
    # it -- it faces sideways, catches less sun and sits in its own shade. Our shading did
    # the opposite, brightening the sunward wall, so on a slender tower the largest and
    # brightest area in the block was facade, which is exactly the reading of a roof. This
    # keeps the hue separation and puts the brightness back the way a photograph has it.
    "dark_walls": {"roof": (150, 225, 255), "wall": (120, 30, 100), "shading": 0.28,
                   "edges": True, "roof_word": "pale blue", "wall_word": "dark purple",
                   "label": "pale roof, dark walls"},
}


def style_of(name: str) -> dict:
    return STYLES.get(name or "colour", STYLES["colour"])


def name_colours(text: str, style: dict) -> str:
    """Keep the prompt's colour words in step with what was actually painted."""
    text = re.sub(r"\bblue\b", style["roof_word"], text)
    return re.sub(r"\borange\b", style["wall_word"], text)

# Trimmed of everything that told the model what it could already see. The storey count and
# the note that a taller block exposes more facade were both restating the block; the lean
# bullet used to call the block "only a guide" and invite revision, which gave away the one
# input the operator sets by hand. Shadow is now the one thing left open, because the block
# cannot know what a new building does to the light around it -- and the closing sentence
# says so, rather than demanding a frame that is pixel-identical everywhere including where
# the new shadow has to fall.
#
# {storeys} is no longer used; str.format ignores the extra keyword.
PROMPT = """This is an overhead satellite photograph of a site in Melbourne. A massing
study of an approved building has been painted onto it, at the exact position and scale it
will occupy: the blue face is the roof, the orange faces are the building's sides. These two
colours are used only to tell the roof from the sides; they are not the building's materials.

Turn that solid into a photographic building. Its ground position and its outline are
fixed -- do not move it, rotate it, or simplify its shape. Its height is {height:.0f} m.

Blend it into the photograph rather than pasting it. Two things must match the rest of the
scene, and both should be read off the real buildings in frame rather than assumed:

- Viewing geometry. The block already carries it: show facade on the same side and to the
  same extent the block does, and keep the roof and the ground footprint exactly where they
  are drawn.
- Lighting. Every shadow in the frame runs the same way. Add, alter or remove shadow as this
  new building's height and shape require.

Give it photographic substance: roof plant and services, parapets, glazing and material on
the facades, and the grain, sharpness, colour balance and haze of the surrounding
photograph. The edges where it meets the ground, neighbouring roofs and street should look
photographed, not cut out. Footprint {area:.0f} square metres, land use {use}.

Apart from the new building and the changes in shadow it brings, everything else stays
pixel-identical. Add no outline, label or annotation anywhere."""

# Kept as a separate wording rather than made plural on the fly. The single-building prompt
# above is the one that has actually been tuned against results, and rewriting its sentences
# at send time would put an untested string in front of the model every time. Two buildings
# also raise a question one never does -- which stands in front, and what falls on what --
# and that has to be said, not inflected.
PROMPT_MANY = """This is an overhead satellite photograph of a site in Melbourne. Massing
studies of {count} approved buildings have been painted onto it, at the exact positions and
scales they will occupy: the blue faces are the roofs, the orange faces are the buildings'
sides. These two colours are used only to tell roofs from sides; they are not the buildings'
materials.

Turn those solids into photographic buildings. Their ground positions and their outlines are
fixed -- do not move them, rotate them, merge them, or simplify their shapes. Where the
blocks overlap, the one painted in front stands in front. They are:

{buildings}

Blend them into the photograph rather than pasting them. Two things must match the rest of
the scene, and both should be read off the real buildings in frame rather than assumed:

- Viewing geometry. The blocks already carry it: show facade on the same side and to the
  same extent each block does, and keep every roof and ground footprint exactly where it is
  drawn.
- Lighting. Every shadow in the frame runs the same way. Add, alter or remove shadow as these
  new buildings' heights and shapes require, including the shadow one of them casts on
  another.

Give them photographic substance: roof plant and services, parapets, glazing and material on
the facades, and the grain, sharpness, colour balance and haze of the surrounding
photograph. The edges where they meet the ground, neighbouring roofs and street should look
photographed, not cut out.

Apart from the new buildings and the changes in shadow they bring, everything else stays
pixel-identical. Add no outline, label or annotation anywhere."""


def load_library() -> list[dict]:
    """The twenty locked projects, each as a polygon in metres east/south of the scene centre.

    This has to reproduce rasterize_footprint exactly, because that is what every validated
    batch run used and what the model was shown. Two things it is easy to get wrong and both
    were: iter_polygons hands back (lat, lon), not the (lon, lat) the GeoJSON stores, and the
    scene is centred on the manifest centroid rather than on the mean of the ring's vertices,
    which sit unevenly around it.
    """
    frame = pd.read_csv(EXTERNAL).drop_duplicates("development_key")
    entries = []
    for _, row in frame.iterrows():
        try:
            rings = [ring for polygon in iter_polygons(json.loads(str(row.geometry_geojson)))
                     for ring in polygon[:1]]
        except Exception:  # noqa: BLE001
            continue
        if not rings:
            continue
        latitude = float(row.centroid_latitude)
        longitude = float(row.centroid_longitude)
        metres_per_lon = 111_320.0 * math.cos(math.radians(latitude))
        entries.append({
            "key": str(row.development_key),
            "height_m": round(float(row.metadata_height_m), 1),
            "area_m2": round(float(row.footprint_area_m2)),
            "longitude": longitude, "latitude": latitude,
            "use": "mixed use" if float(row.metadata_commercial_m2 or 0) > 0 else "residential",
            "polygon_m": [[round((lon - longitude) * metres_per_lon, 2),
                           round((latitude - lat) * 111_320.0, 2)] for lat, lon in rings[0]],
        })
    return sorted(entries, key=lambda e: -e["height_m"])


LIBRARY = load_library()
TEMPLATES = pd.read_csv(PRETRAIN).groupby("target_year").imagery_url_template.first().to_dict()

# An edited prompt outlives the process, so a wording that finally works is not lost to a
# restart. The file wins over the constant above whenever it exists.
PROMPT_FILE = HERE / "prompt.txt"
PROMPT_MANY_FILE = HERE / "prompt_many.txt"
EXAMPLES = HERE / "examples"


def prompt_slot(mode: str, many: bool) -> tuple[Path, str]:
    """Which file holds this job's wording, and what it falls back to."""
    spec = MODES.get(mode) or MODES["add"]
    if many and spec["many"]:
        return HERE / f"prompt_{mode}_many.txt", spec["many"]
    return HERE / f"prompt_{mode}.txt", spec["prompt"]


def current_prompt(many: bool = False, mode: str = "add") -> str:
    path, fallback = prompt_slot(mode, many)
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return fallback


def unwrap(text: str) -> str:
    """Undo the line breaks that only exist to keep the source readable.

    Written out, the prompt has newlines in the middle of sentences -- "A massing\\nstudy of
    an approved building" -- and those newlines are really in the string that gets sent.
    Whether they matter has never been tested here, and since removing them costs nothing,
    the text that goes out now reads the way it looks. Blank lines between paragraphs and
    the boundaries between bullet points are kept, because those carry structure.
    """
    paragraphs = []
    for paragraph in text.split("\n\n"):
        lines, buffer = [], []
        for line in paragraph.split("\n"):
            if line.lstrip().startswith("- ") and buffer:
                lines.append(" ".join(buffer))
                buffer = [line.strip()]
            else:
                buffer.append(line.strip())
        if buffer:
            lines.append(" ".join(buffer))
        paragraphs.append("\n".join(" ".join(l.split()) for l in lines))
    return "\n\n".join(paragraphs)


def check_prompt(text: str, many: bool = False) -> str:
    """Reject a wording that would blow up at generation time rather than at edit time."""
    fields = {"count": 3, "buildings": "- 40 m tall", "height": 100.0, "storeys": 31,
              "area": 1000.0, "use": "residential", "existing": 52.0, "shift": 14.0,
              "bearing": "north-east", "lean": 8.0, "ground": GROUNDS[0], "ratio": 2.0,
              "roof_word": "blue", "wall_word": "orange"}
    try:
        text.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:
        available = ", ".join(f"{{{name}}}" for name in fields)
        raise ValueError(f"{type(exc).__name__}: {exc}. Only {available} are available here, "
                         f"and a literal brace must be doubled as {{{{ or }}}}.") from exc
    return text


def describe(blocks: list) -> str:
    """One line per building, in the order they were placed rather than painted."""
    return "\n".join(
        f"- {float(b['height_m']):.0f} m tall, footprint {float(b['area_m2']):.0f} square "
        f"metres, land use {b.get('use') or 'residential'}" for b in blocks)


@app.get("/api/prompt")
def get_prompt():
    many = request.args.get("many") == "1"
    mode = request.args.get("mode") or "add"
    text = current_prompt(many, mode)
    path, _ = prompt_slot(mode, many)
    return jsonify({"prompt": text, "edited": path.is_file(), "words": len(text.split()),
                    "mode": mode,
                    "grounds": GROUNDS,
                    "modes": [{"name": k, "label": v["label"], "many": v["many"] is not None,
                               "needs_existing": v["needs_existing"],
                               "needs_ground": v["needs_ground"], "paints": v["paint"]}
                              for k, v in MODES.items()]})


@app.post("/api/prompt")
def set_prompt():
    payload = request.get_json() or {}
    many = bool(payload.get("many"))
    path, fallback = prompt_slot(str(payload.get("mode") or "add"), many)
    if payload.get("reset"):
        path.unlink(missing_ok=True)
        return jsonify({"prompt": fallback, "edited": False, "words": len(fallback.split())})
    text = payload.get("prompt", "")
    try:
        check_prompt(text, many)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    path.write_text(text.strip(), encoding="utf-8")
    return jsonify({"prompt": text.strip(), "edited": True, "words": len(text.split())})


@app.post("/api/examples")
def save_example():
    """Keep a generation worth returning to, with everything needed to explain it later."""
    payload = request.get_json()
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = "".join(c for c in str(payload.get("name") or "") if c.isalnum() or c in "-_")
    example_id = f"{stamp}{'-' + name if name else ''}"
    for field in ("image", "massing"):
        blob = payload.get(field)
        if blob:
            (EXAMPLES / f"{example_id}_{field}.png").write_bytes(
                base64.b64decode(blob.split(",")[-1]))
    meta = {k: payload.get(k) for k in
            ("key", "height_m", "lean_deg", "azimuth_deg", "metres_per_px", "size_m",
             "year", "footprint_area_m2", "outside_roi_delta", "note")}
    meta.update({"id": example_id, "saved_at": stamp, "prompt": current_prompt()})
    (EXAMPLES / f"{example_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return jsonify({"id": example_id, "count": len(list(EXAMPLES.glob("*.json")))})


@app.get("/api/examples")
def list_examples():
    if not EXAMPLES.is_dir():
        return jsonify({"items": []})
    items = []
    for path in sorted(EXAMPLES.glob("*.json"), reverse=True):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return jsonify({"items": items})


@app.get("/api/examples/<example_id>/<field>.png")
def example_image(example_id: str, field: str):
    safe = "".join(c for c in example_id if c.isalnum() or c in "-_")
    if field not in ("image", "massing"):
        return jsonify({"error": "unknown field"}), 404
    return send_from_directory(EXAMPLES, f"{safe}_{field}.png")


@app.delete("/api/examples/<example_id>")
def delete_example(example_id: str):
    safe = "".join(c for c in example_id if c.isalnum() or c in "-_")
    removed = 0
    for path in EXAMPLES.glob(f"{safe}*"):
        path.unlink(missing_ok=True)
        removed += 1
    return jsonify({"removed": removed})

# One request again. Racing three at a time existed only to beat the relay's sixty-second
# read timeout, which cut roughly half of these off with the work already finished upstream.
# That timeout is now ten minutes and the same payload that lost five times out of five
# comes back in seventy-two seconds, so the race would only triple the bill. The lock stays
# so two tabs queue rather than compete.
_relay_lock = threading.Lock()


def paced_call(image, roi, prompt):
    """One call at a time, generously timed -- these run sixty to eighty seconds."""
    with _relay_lock:
        started = time.time()
        result = call_single(image, roi, prompt, retries=4, timeout=1200,
                             log=lambda line: print(f"  relay: {line}", flush=True))
        print(f"  relay: returned after {time.time() - started:.0f}s", flush=True)
        return result


@app.get("/tool")
def index():
    return send_from_directory(HERE, "index.html")


# ---------------------------------------------------------------- project showcase
# The showcase wraps this tool rather than changing it: the site's sections all serve one page,
# the tool keeps its own page at /tool, and the report's PDFs, figures and numbers are read from
# docs/report where the build scripts leave them, so the site always shows what the report says.
REPORT = PROJECT / "docs/report"
SHOWCASE = HERE / "showcase"
CHAPTERS = {1: "Introduction", 2: "Project scope and report outline",
            3: "Study area and exploratory analysis of datasets",
            4: "Traffic volume prediction model", 5: "Land-use sensitivity analysis",
            6: "Development scenario imagery tool", 7: "Conclusion"}
APPENDICES = {"A": "Data dictionary", "B": "Definitions of the land-use attributes",
              "C": "Full experimental results", "D": "Prompts for the development scenario imagery tool"}
REPORT_FILE = re.compile(r"^(?:ch(\d)|report)_(zh|en)\.pdf$")
FIGURE_FILE = re.compile(r"^fig[3-6]_[A-Za-z0-9_]+\.png$")


@app.get("/")
@app.get("/report")
@app.get("/imagery")
@app.get("/model")
@app.get("/code")
def showcase():
    return send_from_directory(SHOWCASE, "index.html")


def pdf_pages(path: Path):
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:
        return None


def report_outline(path: Path):
    """The full report's chapters and appendices, from its top-level bookmarks (make_report.py),
    each with the page it starts on, so the list can move within the one file."""
    entries = []
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            for level, title, page in doc.get_toc():
                if level != 1 or page < 2:              # the cover is the file's first page
                    continue
                first = title.split(" ")[0]
                appendix = first in ("附录", "Appendix")          # Chinese or English report
                if first.isdigit():
                    n, name = first, CHAPTERS.get(int(first), title)
                    heading = f"Chapter {n} · {name}"
                elif appendix:
                    n = title.split(" ")[1]
                    name = APPENDICES.get(n, title)
                    heading = f"Appendix {n} · {name}"
                else:
                    n, name = "", {"执行摘要": "Executive summary", "目录": "Contents",
                               "参考文献": "References"}.get(title, title)
                    heading = name
                entries.append({"n": n, "title": name, "heading": heading, "page": page,
                                "label": doc[page - 1].get_label(),
                                "appendix": appendix})
    except Exception:
        return []
    return entries


@app.get("/api/showcase/reports")
def showcase_reports():
    items = []
    for path in sorted((REPORT / "build").glob("*.pdf")):
        match = REPORT_FILE.match(path.name)
        if not match:
            continue
        chapter = int(match.group(1)) if match.group(1) else None
        item = {"file": path.name, "chapter": chapter,
                "title": CHAPTERS.get(chapter, "Full report"),
                "language": match.group(2), "pages": pdf_pages(path),
                "updated": time.strftime("%d %b %Y", time.localtime(path.stat().st_mtime)),
                "size_mb": round(path.stat().st_size / 1e6, 1)}
        if chapter is None:
            item["outline"] = report_outline(path)
        items.append(item)
    return jsonify({"items": items, "chapters": CHAPTERS})


@app.get("/files/report/<name>")
def report_file(name):
    if not REPORT_FILE.match(name):
        return jsonify({"error": "not a report file"}), 404
    return send_from_directory(REPORT / "build", name)


@app.get("/files/report-thumb/<name>")
def report_thumb(name):
    """The first page of a report PDF, small, for the overview's entry card."""
    path = REPORT / "build" / name
    if not REPORT_FILE.match(name) or not path.is_file():
        return jsonify({"error": "not a report file"}), 404
    import pymupdf
    with pymupdf.open(path) as doc:
        png = doc[0].get_pixmap(matrix=pymupdf.Matrix(0.8, 0.8)).tobytes("png")
    return app.response_class(png, mimetype="image/png")


@app.get("/files/figure/<name>")
def report_figure(name):
    if not FIGURE_FILE.match(name):
        return jsonify({"error": "not a report figure"}), 404
    return send_from_directory(REPORT / "figures", name)


# The Code view reads code_release/ as its MANIFEST.csv groups it: each stage's files under the
# stage's title, highlighted here. The READMEs are not shown (user's request, 2026-10-02), only
# their first heading names the stage. Only files the manifest lists are ever served.
CODE = PROJECT / "code_release"
CODE_STYLE = "xcode"


def code_files():
    with open(CODE / "MANIFEST.csv", newline="", encoding="utf-8") as fh:
        return {row["path"]: row for row in csv.DictReader(fh)}


def stage_title(folder: Path) -> str:
    path = folder / "README.md"
    title = re.match(r"#\s*(.+)", path.read_text("utf-8")) if path.is_file() else None
    return title.group(1).strip() if title else folder.name


@app.get("/api/showcase/code")
def showcase_code():
    files = code_files()
    stages = []
    for key in sorted({row["stage"] for row in files.values()}):
        stages.append({"key": key, "title": stage_title(CODE / key), "files": [
            {"path": p, "name": p.split("/", 1)[1], "role": r["role"], "lines": int(r["lines"])}
            for p, r in files.items() if r["stage"] == key]})
    return jsonify({"stages": stages, "files": len(files),
                    "lines": sum(int(r["lines"]) for r in files.values())})


@app.get("/api/showcase/code/file")
def showcase_code_file():
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import TextLexer, get_lexer_for_filename
    from pygments.util import ClassNotFound
    path = request.args.get("path", "")
    row = code_files().get(path)
    if row is None:
        return jsonify({"error": "not in the code release"}), 404
    text = (CODE / path).read_text("utf-8", errors="replace")
    try:
        lexer = get_lexer_for_filename(path, stripnl=False)
    except ClassNotFound:
        lexer = TextLexer(stripnl=False)
    html = highlight(text, lexer, HtmlFormatter(linenos="table", cssclass="hl", lineanchors="L",
                                                anchorlinenos=True, style=CODE_STYLE))
    return jsonify({"path": path, "source": row["source"], "role": row["role"],
                    "lines": int(row["lines"]), "language": lexer.name, "html": html})


@app.get("/api/showcase/code/style.css")
def showcase_code_style():
    from pygments.formatters import HtmlFormatter
    return app.response_class(HtmlFormatter(style=CODE_STYLE).get_style_defs(".hl"),
                              mimetype="text/css")


@app.get("/files/code/<path:name>")
def code_file(name):
    if name not in code_files():
        return jsonify({"error": "not in the code release"}), 404
    return send_from_directory(CODE, name, mimetype="text/plain")


@app.get("/api/footprints")
def footprints():
    return jsonify({"items": LIBRARY, "years": sorted(TEMPLATES),
                    "styles": [{"name": k, "label": v["label"], "roof": v["roof"],
                                "wall": v["wall"], "shading": v["shading"],
                                "edges": v["edges"]} for k, v in STYLES.items()]})


# A four hundred metre patch at zoom 18 spans about twenty tiles, and stitch_patch fetches
# them one after another with no cache -- twenty round trips in series, which is the eight
# seconds you wait every time you pick a project or change the year. The tiles are immutable
# per year and coordinate, and neighbouring patches share most of them, so caching them and
# fetching the misses at once turns that into about a second, and into nothing at all the
# second time you look at the same place.
_tiles: dict[str, Image.Image] = {}
_tile_lock = threading.Lock()
_original_fetch_tile = build_data.fetch_tile


def cached_fetch_tile(url: str, timeout: int) -> Image.Image:
    with _tile_lock:
        hit = _tiles.get(url)
    if hit is not None:
        return hit
    tile = _original_fetch_tile(url, timeout)
    with _tile_lock:
        if len(_tiles) > 3000:                    # a few hundred MB at 256px RGB
            _tiles.clear()
        _tiles[url] = tile
    return tile


build_data.fetch_tile = cached_fetch_tile


def warm_tiles(longitude: float, latitude: float, size_m: float, year: int) -> None:
    """Pull the tiles this patch needs concurrently, so the stitch itself finds them ready.

    The indices come from the same helpers stitch_patch uses, not from a second copy of the
    arithmetic; a missed tile here simply costs a serial fetch later rather than a wrong
    picture.
    """
    zoom, tile_size = 18, build_data.TILE_SIZE
    tile_x, tile_y = build_data.lonlat_to_tile_fraction(longitude, latitude, zoom)
    span = size_m / build_data.ground_resolution_m_per_pixel(latitude, zoom)
    centre_x, centre_y = tile_x * tile_size, tile_y * tile_size
    half = span / 2.0
    urls = []
    for row in range(int(math.floor((centre_y - half) / tile_size)),
                     int(math.floor((centre_y + half - 1) / tile_size)) + 1):
        for column in range(int(math.floor((centre_x - half) / tile_size)),
                            int(math.floor((centre_x + half - 1) / tile_size)) + 1):
            urls.append(TEMPLATES[year].format(
                z=zoom, zoom=zoom, x=column, y=row, TileMatrixSet="default028mm",
                TileMatrix=zoom, TileRow=row, TileCol=column))
    missing = [u for u in urls if u not in _tiles]
    if not missing:
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(cached_fetch_tile, u, 90) for u in missing]:
            try:
                future.result()
            except Exception:  # noqa: BLE001
                pass                              # stitch_patch will retry it serially


@app.get("/api/scene")
def scene():
    longitude = float(request.args.get("lon"))
    latitude = float(request.args.get("lat"))
    size_m = float(request.args.get("size_m", 400))
    pixels = int(request.args.get("px", 1400))
    year = int(request.args.get("year", max(TEMPLATES)))
    started = time.time()
    warm_tiles(longitude, latitude, size_m, year)
    image = stitch_patch(longitude=longitude, latitude=latitude, patch_size_m=size_m,
                         output_size_px=pixels, zoom=18,
                         url_template=TEMPLATES[year], timeout=90)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    print(f"  scene: {size_m:.0f} m {year} in {time.time() - started:.1f}s "
          f"({len(_tiles)} tiles cached)", flush=True)
    return app.response_class(buffer.getvalue(), mimetype="image/png")


def face_shade(colour, cosine: float, strength: float):
    """Lighten a wall that faces the sensor, darken one that turns away."""
    factor = 1.0 + strength * cosine
    return tuple(int(min(255, max(0, round(channel * factor)))) for channel in colour)


def paint(scene_rgb: np.ndarray, blocks: list, metres_per_px: float, azimuth_deg: float,
          lean_deg: float, style_name: str = "colour", draw: str = "solid"):
    """Paint every block into the photograph, nearest to the sensor last.

    With one block the order is moot; with two or three it is the whole difference between a
    scene and a collage. The lean gives the view a horizontal depth axis -- the direction the
    roofs are displaced away from -- so sorting the blocks along it and painting far to near
    lets a nearer tower stand in front of one behind it, which is what the photograph would
    show. Each block is drawn complete before the next begins, so a nearer block covers a
    farther one's walls rather than interleaving with them.
    """
    radians = math.radians(azimuth_deg)
    towards_sensor = np.array([-math.cos(radians), -math.sin(radians)])
    painted = scene_rgb.copy()
    solid = np.zeros(scene_rgb.shape[:2], dtype=np.uint8)

    def depth(block):
        return float(np.dot(np.asarray(block["polygon_px"], dtype=np.float64).mean(axis=0),
                            towards_sensor))

    for block in sorted(blocks, key=depth):
        polygon = np.asarray(block["polygon_px"], dtype=np.float32)
        # An alteration shows both states: the building as it stands, outlined, and the
        # shape it becomes, filled. Without the outline the model is told a building is
        # being lowered but not which one or from what -- and it has to know the old height
        # to work out how much shadow to give back.
        existing = float(block.get("existing_m") or 0.0)
        if draw == "solid+wire" and existing > 0:
            paint_one(painted, solid, polygon, existing, metres_per_px, radians,
                      math.radians(lean_deg), towards_sensor, style_of(style_name), "wire")
        # A demolition is drawn as a wireframe: the footprint, the roof outline and the
        # vertical edges, with nothing filled. Filling it would hide the building the model
        # has to remove, and painting nothing at all left it guessing which of several
        # buildings in frame was meant.
        paint_one(painted, solid, polygon, float(block["height_m"]), metres_per_px, radians,
                  math.radians(lean_deg), towards_sensor, style_of(style_name),
                  "solid" if draw == "solid+wire" else draw)
    return painted, solid.astype(bool)


def paint_one(painted: np.ndarray, solid: np.ndarray, polygon_px: np.ndarray, height_m: float,
              metres_per_px: float, radians: float, lean_radians: float,
              towards_sensor: np.ndarray, style: dict, draw: str = "solid"):
    """Roof plane plus the walls the lean exposes, drawn straight into the photograph.

    Flat-filling every wall the same orange merges adjacent facets into one shape, and the
    corners between them -- which is where the building's plan actually shows up -- become
    invisible. Shading each facet by how it is turned relative to the sensor separates them
    without drawing anything: a line would be an annotation, and the prompt tells the model
    to remove annotations, which it demonstrably does.
    """
    shading = style["shading"]
    shift = height_m * math.tan(lean_radians) / metres_per_px
    offset = np.array([shift * math.cos(radians), shift * math.sin(radians)])

    top = polygon_px + offset
    base = polygon_px.astype(np.int32)

    footprint = np.zeros(solid.shape, dtype=np.uint8)
    cv2.fillPoly(footprint, [base], 1)
    solid[footprint.astype(bool)] = 1

    # Which way is "out" depends on the winding, and the footprints come both ways round.
    # cv2.contourArea returns an unsigned area unless asked otherwise, so testing its sign
    # always said "counter-clockwise" -- which inverted every normal on half the parcels and
    # took the shading and the hidden-corner test down with it. Shoelace, signed, instead.
    shoelace = float(sum(polygon_px[i][0] * polygon_px[(i + 1) % len(polygon_px)][1]
                         - polygon_px[(i + 1) % len(polygon_px)][0] * polygon_px[i][1]
                         for i in range(len(polygon_px))))
    winding = 1.0 if shoelace >= 0 else -1.0
    normals = []
    for index in range(len(polygon_px)):
        nxt = (index + 1) % len(polygon_px)
        edge = polygon_px[nxt] - polygon_px[index]
        length = float(np.hypot(*edge)) or 1.0
        normal = np.array([edge[1], -edge[0]]) / length * winding
        normals.append(normal)
        cosine = float(np.dot(normal, towards_sensor))
        quad = np.array([polygon_px[index], polygon_px[nxt], top[nxt], top[index]],
                        dtype=np.int32)
        cv2.fillPoly(solid, [quad], 1)
        if draw != "solid":
            continue
        # Only the walls turned towards the sensor get painted. A wall facing away is
        # behind the solid and can never be seen; drawing it anyway was harmless on a
        # convex parcel, where the roof covered it, and on the L-shaped and notched ones
        # it painted straight over the walls in front -- which is the block looking
        # transparent. Front walls plus the roof already cover the whole silhouette.
        if cosine <= 0:
            continue
        cv2.fillPoly(painted, [quad], face_shade(style["wall"], cosine, shading))
    cv2.fillPoly(solid, [top.astype(np.int32)], 1)
    if draw == "wire":
        # footprint, roof outline, and the vertical edges that can be seen -- enough to say
        # which building and how tall, while leaving it visible underneath
        thickness = max(2, int(round(solid.shape[0] / 420)))
        cv2.polylines(painted, [base], True, style["wall"], thickness, cv2.LINE_AA)
        cv2.polylines(painted, [top.astype(np.int32)], True, style["roof"], thickness,
                      cv2.LINE_AA)
        facing = [float(np.dot(normals[i], towards_sensor)) > 0
                  for i in range(len(polygon_px))]
        for index in range(len(polygon_px)):
            if not (facing[index] or facing[index - 1]):
                continue
            cv2.line(painted, tuple(base[index]), tuple(top[index].astype(int)),
                     style["wall"], thickness, cv2.LINE_AA)
        return
    if draw != "solid":
        return
    cv2.fillPoly(painted, [top.astype(np.int32)], style["roof"])

    # Only the corners that can actually be seen. A corner is visible when one of the two
    # facades meeting there faces the sensor; the ones behind sit inside the solid, and
    # drawing them anyway laid lines across the roof as though the block were glass.
    if not style["edges"]:
        return
    edge_colour = tuple(int(channel * 0.62) for channel in style["wall"])
    facing = [float(np.dot(normals[i], towards_sensor)) > 0 for i in range(len(polygon_px))]
    for index in range(len(polygon_px)):
        if not (facing[index] or facing[index - 1]):
            continue
        cv2.line(painted, tuple(base[index]), tuple(top[index].astype(int)),
                 edge_colour, 1, cv2.LINE_AA)


def blocks_of(payload: dict, metres_per_px: float) -> list[dict]:
    """Accept a list of buildings, or the single-building payload the tool has always sent."""
    raw = payload.get("blocks") or [{"polygon_px": payload["polygon_px"],
                                     "height_m": payload["height_m"],
                                     "use": payload.get("use")}]
    blocks = []
    for item in raw:
        polygon = np.asarray(item["polygon_px"], dtype=np.float32)
        blocks.append({"polygon_px": polygon, "height_m": float(item["height_m"]),
                       "existing_m": float(item.get("existing_m")
                                           or payload.get("existing_m") or 0.0),
                       "use": item.get("use") or payload.get("use") or "residential",
                       "area_m2": float(cv2.contourArea(polygon)) * metres_per_px ** 2})
    return blocks


@app.post("/api/massing")
def massing_preview():
    """Preview only -- no model call, so the sliders stay responsive."""
    payload = request.get_json()
    scene_rgb = decode_scene(payload)
    metres_per_px = float(payload["metres_per_px"])
    spec = MODES.get(str(payload.get("mode") or "add")) or MODES["add"]
    painted, _ = paint(scene_rgb, blocks_of(payload, metres_per_px), metres_per_px,
                       float(payload["azimuth_deg"]), float(payload["lean_deg"]),
                       style_name=str(payload.get("style") or "colour"), draw=spec["paint"])
    return jsonify({"image": encode(Image.fromarray(painted))})


def decode_scene(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["scene_png"].split(",")[-1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def encode(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@app.post("/api/snapshot")
def snapshot():
    """Write a panel exactly as the browser drew it, for write-ups and for checking the UI."""
    payload = request.get_json()
    name = "".join(c for c in str(payload.get("name", "panel")) if c.isalnum() or c in "-_")
    target = HERE / "snapshots" / f"{name or 'panel'}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(payload["image"].split(",")[-1]))
    return jsonify({"saved": str(target)})


@app.post("/api/generate")
def generate():
    payload = request.get_json()
    scene_rgb = decode_scene(payload)
    metres_per_px = float(payload["metres_per_px"])
    blocks = blocks_of(payload, metres_per_px)

    style_name = str(payload.get("style") or "colour")
    mode = str(payload.get("mode") or "add")
    spec = MODES.get(mode) or MODES["add"]
    painted, solid = paint(scene_rgb, blocks, metres_per_px,
                           float(payload["azimuth_deg"]), float(payload["lean_deg"]),
                           style_name=style_name, draw=spec["paint"])
    # The editable region has to cover the tallest shadow in play, or the shadow that needs to
    # be cast -- or removed -- falls outside the mask and the model cannot touch it. For a
    # demolition the building coming down is the tall one, so its height decides the radius.
    tallest = max(max(float(block["height_m"]), float(block.get("existing_m") or 0.0))
                  for block in blocks)
    radius = int(math.ceil(base.shadow_budget_m(tallest) / metres_per_px)) + 8
    roi = base.shadow_region(solid, radius)

    style = style_of(style_name)
    many = len(blocks) > 1 and spec["many"] is not None
    one = blocks[0]
    lean = float(payload["lean_deg"])
    # The roof/facade confusion is a geometry fact, so state it as one: how far the roof sits
    # from the footprint, and which way. Compass bearing rather than the screen angle, since
    # the frame is north-up and the model reads a bearing more reliably than "315 degrees".
    shift = one["height_m"] * math.tan(math.radians(lean))
    compass = ["east", "south-east", "south", "south-west", "west", "north-west", "north",
               "north-east"]
    bearing = compass[int(round(float(payload["azimuth_deg"]) % 360 / 45)) % 8]
    fields = {"count": len(blocks), "buildings": describe(blocks),
              "height": one["height_m"], "area": sum(b["area_m2"] for b in blocks),
              "storeys": max(int(round(one["height_m"] / 3.2)), 1), "use": one["use"],
              "existing": float(one.get("existing_m") or 0.0),
              "shift": shift, "bearing": bearing, "lean": lean,
              "ratio": (float(one.get("existing_m") or 0.0) / one["height_m"]
                        if one["height_m"] > 0 else 1.0),
              "ground": str(payload.get("ground") or GROUNDS[0]),
              # the prompts name the faces by placeholder rather than by a literal colour,
              # so a change of massing style reaches the wording without a regex
              "roof_word": style["roof_word"], "wall_word": style["wall_word"]}
    prompt = name_colours(unwrap(current_prompt(many=many, mode=mode)), style).format(**fields)

    original_render = base.RENDER
    base.RENDER = scene_rgb.shape[0]
    try:
        generated, error = paced_call(Image.fromarray(painted), roi, prompt)
    finally:
        base.RENDER = original_render
    if generated is None:
        return jsonify({"error": error}), 502

    merged = base.blend_into_scene(Image.fromarray(scene_rgb), generated, roi)
    outside = float(np.abs(merged.astype(np.float32)
                           - scene_rgb.astype(np.float32))[~roi].mean())
    return jsonify({
        "image": encode(base.watermark(Image.fromarray(merged))),
        "massing": encode(Image.fromarray(painted)),
        "outside_roi_delta": outside,
        "roi_fraction": float(roi.mean()),
        "footprint_area_m2": round(sum(block["area_m2"] for block in blocks)),
        "buildings": len(blocks),
        "mode": mode,
    })


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("set OPENAI_API_KEY before starting the server")
    port = int(os.environ.get("PORT", "8000"))
    print(f"{len(LIBRARY)} footprints loaded; imagery years {sorted(TEMPLATES)}")
    print(f"open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
