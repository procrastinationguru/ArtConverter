"""
HD art pipeline for arcanum-ce (Phase 8: native-res UI/HUD rendering,
docs/GPU_BLITTER_ROADMAP.md).

Pure upscale + directory movement, nothing else: unpack a vanilla .ART,
ESRGAN x4 each frame (colour-key inpainted first so the model doesn't see
it), derive alpha from the colour-key mask, and write RGBA PNG sidecars to
Arcanum's loose `hd/art/` tree - a purely additive, visual-only asset the
engine loads alongside the untouched vanilla .ART (still authoritative for
metrics/hit-tests). No repacking, no quantizing to a fixed pixel size, no
touching the game's `data/` dat-override layer - that whole
unpack->upscale->quantize->repack->deploy workflow (and `--keep-size`) is
retired now that the engine composites native-resolution PNGs directly
instead of shipping pre-shrunk replacement .ART files.

Usage:
    python pipeline.py hd art/interface/MainMenuBack.ART
    python pipeline.py hd interface --list category_lists/interface_nofont.txt

`hd-overlay` is a separate, still-active mechanism: packages a `hd`-produced
frame-0 PNG as the loose full-res BMP the main menu's HD spike loads directly
(tig_video_set_hd_overlay()) - see its docstring.
"""

import argparse
import re
import shutil
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

# Townmap composites (cmd_hd_townmap) are legitimately huge - a town's real
# bounding box at 4x can run well past Pillow's default ~179M px decompression
# -bomb guard (Ashbury alone: 71x80 tiles -> 186M px at 4x). Locally-generated
# content, not untrusted input, so disabling the guard is the standard escape
# hatch here rather than a real risk.
Image.MAX_IMAGE_PIXELS = None

# hqx 1.0's __init__.py type-annotates against PIL.PyAccess, which newer
# Pillow (12.x, dropped the pure-Python pixel access shim) no longer ships.
# The annotation is never actually called at runtime, so a dummy module
# satisfies the import.
import types as _types
if "PIL.PyAccess" not in sys.modules:
    _dummy_pyaccess = _types.ModuleType("PIL.PyAccess")
    _dummy_pyaccess.PyAccess = object
    sys.modules["PIL.PyAccess"] = _dummy_pyaccess
    import PIL as _PIL
    _PIL.PyAccess = _dummy_pyaccess
import hqx

import config


def find_source_art(rel_path: str) -> Path:
    rel_path = rel_path.replace("\\", "/")
    for root in config.EXTRACTED_DAT_ROOTS:
        candidate = root / rel_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find '{rel_path}' under any of: "
        + ", ".join(str(r) for r in config.EXTRACTED_DAT_ROOTS)
    )


def work_dir_for(rel_path: str) -> Path:
    rel_path = rel_path.replace("\\", "/")
    stem = Path(rel_path).with_suffix("").name
    return config.WORK_DIR / stem


def run_art_converter(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(config.ART_CONVERTER_EXE), str(src), str(dst)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"art-converter.exe failed ({src} -> {dst}):\n{result.stdout}\n{result.stderr}"
        )


def cmd_unpack(rel_path: str, quiet: bool = False) -> Path:
    src = find_source_art(rel_path)
    wd = work_dir_for(rel_path)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    basename = wd / Path(rel_path).with_suffix("").name
    run_art_converter(src, basename)
    if not quiet:
        print(f"Unpacked {src} -> {wd}")
    return wd


def frame_bmps(wd: Path) -> list[Path]:
    return sorted(p for p in wd.glob("*.bmp") if "_hd" not in p.stem)


# A general-purpose 4x photo-upscale model occasionally has too little
# spatial context on a tiny/near-flat source frame and produces RGB static
# ("pixel soup") instead of an enhanced image - confirmed this session on
# XP_Pip1-10 (thin flat-color bars) and ScrllSlideB/T, across all 4 models
# available (realesrgan-x4plus, 4xNomos8kSC, realesrgan-x4plus-anime,
# realesr-animevideov3-x4) - not a model-choice problem, the tiny/degenerate
# input itself triggers it. Mean adjacent-pixel absolute difference,
# averaged over RGB and both axes, cleanly separates the two groups on this
# session's full interface sweep:
#   9.2-14.4  - clean  (SKL_PickLock, S_Fire, S_EvilNecro, s_morph, intcedit,
#                        MM_Loc - busy/high-contrast small icons, not noise)
#   32.4-71.5 - garbage (XP_Pip1-10, ScrllSlideB/T - real static)
# Wide gap, no overlap - 20.0 sits comfortably in it.
PIXEL_SOUP_ROUGHNESS_THRESHOLD = 20.0


def roughness(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    dx = np.abs(np.diff(arr, axis=1)).mean()
    dy = np.abs(np.diff(arr, axis=0)).mean()
    return (dx + dy) / 2


def run_esrgan(src_png: Path, dest_png: Path, model: str) -> None:
    args = [str(config.REALESRGAN_EXE), "-i", str(src_png), "-o", str(dest_png), "-s", str(HD_SCALE), "-n", model]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"realesrgan failed on {src_png}:\n{result.stdout}\n{result.stderr}")


def run_realcugan(src_png: Path, dest_png: Path) -> None:
    """-n -1 = conservative/no-denoise model - least alteration to source
    detail, confirmed the right choice for the dithered-icon/tiny-degenerate
    cases this is used for (denoise levels would smooth exactly the fine
    detail we're trying to preserve)."""
    args = [
        str(config.REALCUGAN_EXE), "-i", str(src_png), "-o", str(dest_png),
        "-s", str(HD_SCALE), "-n", "-1", "-m", str(config.REALCUGAN_MODEL_DIR),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"realcugan failed on {src_png}:\n{result.stdout}\n{result.stderr}")


# ncnn-vulkan's directory mode (-i indir -o outdir) loads the Vulkan device
# and model weights ONCE and reuses them for every image found in indir,
# instead of paying that fixed setup cost per subprocess launch. Confirmed
# by direct benchmark: 15 tiny (60x60) frames through 15 separate
# single-file launches took 101.8s (~6.8s/frame); the same 15 frames through
# one directory-mode launch took 6.9s (~0.46s/frame) - ~15x, because actual
# upscale compute on a frame this small is milliseconds and setup dominated.
# Confirmed on GPU telemetry too: the original all-single-launch batch sat
# at ~20% GPU utilization / ~45W the whole run (of a ~285W card) - compute
# was never the bottleneck, subprocess/model-load churn was.
#
# Directory mode is not safe to trust, though: after some inputs (a 6x90 or
# 8x40 frame in the x16 trial - not a clean size threshold, 5 and 12 px wide
# were fine) every later output in the same batch is another buffer read at
# the wrong stride, exit code 0, right size, not blank. Every batch output is
# therefore checked against its own input: box-downscaled back to 1x, the
# luma correlation with the source was >= 0.94 for every real upscale and
# <= 0.22 (or a flat image) for every garbage one. Mean abs diff doesn't
# separate them - dithered chainmail or a gradient vial legitimately differ
# by 15-24 once ESRGAN smooths the dither. Failures are redone single-file,
# which never showed the problem.
BATCH_OUTPUT_MIN_CORR = 0.5


def structural_corr(src: np.ndarray, out_small: np.ndarray, where: np.ndarray | None = None) -> float:
    """Luma correlation of two same-size RGB arrays (optionally only where
    `where`). A flat source can't be judged -> 1.0; a flat output of a
    non-flat source -> 0.0."""
    lum = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    a = src.astype(np.float32) @ lum
    b = out_small.astype(np.float32) @ lum
    if where is not None:
        a, b = a[where], b[where]
    # Too few pixels to judge: eye_candy has hundreds of 3x3-ish frames with
    # a handful of opaque pixels whose (correct) upscales correlate at
    # random, and each "failure" cost a single-file redo.
    if a.size < 64 or a.std() < 1.0:
        return 1.0
    if b.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def batch_output_corr(src_png: Path, out_png: Path) -> float:
    src = Image.open(src_png).convert("RGB")
    out = Image.open(out_png).convert("RGB")
    if out.size != (src.width * HD_SCALE, src.height * HD_SCALE):
        return -1.0
    return structural_corr(np.asarray(src), np.asarray(out.resize(src.size, Image.BOX)))


def verify_batch(src_dir: Path, dest_dir: Path, redo) -> None:
    redone = 0
    for src in sorted(src_dir.glob("*.png")):
        out = dest_dir / src.name
        if out.is_file() and batch_output_corr(src, out) >= BATCH_OUTPUT_MIN_CORR:
            continue
        redo(src, out)
        redone += 1
        corr = batch_output_corr(src, out) if out.is_file() else -1.0
        if corr < BATCH_OUTPUT_MIN_CORR:
            print(f"  warning: {out.name} still doesn't match its source after a single-file redo (corr {corr:.2f})")
    if redone:
        print(f"  {redone} corrupt batch output(s) in {dest_dir.name} redone single-file")


# The corruption follows small inputs: with every input edge-padded to at
# least this many px (outputs cropped back), the x16 trial batch that broke
# 15 of 49 frames came out clean - and the overnight scenery fix had most of
# its 222 frames redone single-file (~1.5 s each) before this.
BATCH_MIN_SIDE = 64


def run_padded_batch(src_dir: Path, dest_dir: Path, args_for) -> None:
    """Run one ncnn directory-mode upscale of src_dir into dest_dir with
    every input padded to BATCH_MIN_SIDE (replicated edges) and each output
    cropped back to 4x its input. args_for(in_dir, out_dir) -> argv."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    pad_in = dest_dir.parent / (dest_dir.name + "_padin")
    pad_out = dest_dir.parent / (dest_dir.name + "_padout")
    for d in (pad_in, pad_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    sizes: dict[str, tuple[int, int]] = {}
    for src in src_dir.glob("*.png"):
        with Image.open(src) as im:
            w, h = im.size
            if w >= BATCH_MIN_SIDE and h >= BATCH_MIN_SIDE:
                shutil.copyfile(src, pad_in / src.name)
                continue
            arr = np.asarray(im.convert("RGB"))
        sizes[src.name] = (w, h)
        arr = np.pad(arr, ((0, max(0, BATCH_MIN_SIDE - h)), (0, max(0, BATCH_MIN_SIDE - w)), (0, 0)), mode="edge")
        Image.fromarray(arr, "RGB").save(pad_in / src.name)
    result = subprocess.run(args_for(pad_in, pad_out), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ncnn batch failed on {src_dir}:\n{result.stdout}\n{result.stderr}")
    for out in pad_out.glob("*.png"):
        if out.name in sizes:
            w, h = sizes[out.name]
            with Image.open(out) as im:
                im.crop((0, 0, w * HD_SCALE, h * HD_SCALE)).save(dest_dir / out.name)
        else:
            shutil.move(str(out), str(dest_dir / out.name))
    shutil.rmtree(pad_in, ignore_errors=True)
    shutil.rmtree(pad_out, ignore_errors=True)


def run_esrgan_batch(src_dir: Path, dest_dir: Path, model: str) -> None:
    run_padded_batch(src_dir, dest_dir, lambda i, o: [
        str(config.REALESRGAN_EXE), "-i", str(i), "-o", str(o), "-s", str(HD_SCALE), "-n", model,
    ])
    verify_batch(src_dir, dest_dir, lambda s, d: run_esrgan(s, d, model))


# -j 1:1:1: with the default 1:2:2 directory mode writes garbage (noise,
# black, even another input's image under this name) for most frames of a
# mixed-size batch - 47 of 49 in the x16 trial - and still exits 0.
def run_realcugan_batch(src_dir: Path, dest_dir: Path) -> None:
    run_padded_batch(src_dir, dest_dir, lambda i, o: [
        str(config.REALCUGAN_EXE), "-i", str(i), "-o", str(o),
        "-s", str(HD_SCALE), "-n", "-1", "-m", str(config.REALCUGAN_MODEL_DIR), "-j", "1:1:1",
    ])
    verify_batch(src_dir, dest_dir, run_realcugan)


def is_blank_output(img: Image.Image) -> bool:
    """ncnn-vulkan can hit a transient GPU error ('vkQueueSubmit failed',
    'vkWaitForFences failed') and still exit 0, silently leaving the output
    at whatever the file already contained (zeroed/blank) instead of
    propagating the failure - confirmed via direct repro, and NOT specific
    to directory-batch mode (reproduced standalone on a single 38x10 frame
    too, so this is a pre-existing GPU/driver flake the pipeline never had a
    check for). A real upscaled frame always has some variance from model
    texture/noise even on flat-color source art, so exact-zero standard
    deviation is a safe, specific signal that the write silently failed."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return bool(arr.std() == 0.0)


def load_and_validate(path: Path, expected_size: tuple[int, int], label: str) -> Image.Image:
    if not path.is_file():
        raise RuntimeError(f"{label} did not produce {path}")
    img = Image.open(path).convert("RGB")
    if img.size != expected_size:
        raise RuntimeError(f"{label} produced wrong size {img.size} for {path} (expected {expected_size})")
    return img


def upscale_pre_lanczos(src_rgb: np.ndarray, w: int, h: int, bmp: Path, model: str) -> Image.Image:
    """Fallback for confirmed pixel-soup output: Lanczos-upscale the source
    4x FIRST (so ESRGAN sees a smooth ~4x-larger image instead of a handful
    of raw pixels - no more degenerate tiny-input pattern to trigger static),
    run ESRGAN x4 on that at its native trained scale, then Lanczos back down
    to the true 4x target. Confirmed on XP_Pip1/ScrllSlideB: real
    beveled/shaded detail instead of static, better than a plain flat
    Lanczos x4 of the original since ESRGAN still contributes real shading."""
    pre = Image.fromarray(src_rgb, "RGB").resize((w * HD_SCALE, h * HD_SCALE), Image.LANCZOS)
    pre_png = bmp.with_name(bmp.stem + "_pre.png")
    pre.save(pre_png)
    preesrgan_png = bmp.with_name(bmp.stem + "_preesrgan.png")
    run_esrgan(pre_png, preesrgan_png, model)
    big = Image.open(preesrgan_png).convert("RGB")
    return big.resize((w * HD_SCALE, h * HD_SCALE), Image.LANCZOS)


def read_bmp_palette(path: Path) -> list[int]:
    """Read the 256-entry BGRA color table SaveBMPS() always writes, and
    return it as a flat RGB list suitable for Image.putpalette().

    Deliberately avoids PIL's Image.open() here: opening `path` with PIL and
    then writing a new file over that same path later in the same process
    intermittently deadlocks/fails on Windows (OSError 22 / WinError 5),
    seemingly a Pillow file-handle lifecycle quirk rather than an external
    lock (a fresh process can rename/delete the file immediately after).
    """
    with open(path, "rb") as f:
        header = f.read(14 + 4)
        bi_size = struct.unpack_from("<I", header, 14)[0]
        f.seek(14 + bi_size)
        raw = f.read(256 * 4)
    rgb = []
    for i in range(256):
        b, g, r, _a = raw[i * 4 : i * 4 + 4]
        rgb.extend((r, g, b))
    return rgb


def read_bmp_dims(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        f.seek(14 + 4)
        w, h = struct.unpack("<ii", f.read(8))
    return w, h


def read_bmp_indices(path: Path) -> list[int]:
    """Read the raw 8bpp palette-index pixel data, top-to-bottom row-major
    (matching PIL's pixel order), so callers can locate specific-index
    pixels (e.g. the color-key background) before quantize() overwrites
    this file. See read_bmp_palette() for why this avoids PIL.
    """
    with open(path, "rb") as f:
        header = f.read(14 + 4)
        bi_size = struct.unpack_from("<I", header, 14)[0]
        f.seek(14)
        dib = f.read(bi_size)
        width, height = struct.unpack_from("<ii", dib, 4)
        height = abs(height)
        f.seek(14 + bi_size + 256 * 4)
        row_stride = ((width + 3) // 4) * 4
        data = f.read(row_stride * height)
    indices = [0] * (width * height)
    for row in range(height):
        src = data[row * row_stride : row * row_stride + width]
        dst_row = height - 1 - row  # BMP rows are stored bottom-up
        indices[dst_row * width : (dst_row + 1) * width] = src
    return indices


def cmd_hd_overlay(rel_path: str) -> Path:
    """Emit the full-resolution RGB BMP consumed directly by the engine's
    native-resolution GPU spike (tig_video_set_hd_overlay(), loaded via plain
    SDL_LoadBMP - not the palette-indexed ART/VFS path the rest of this
    pipeline targets). No palette quantization, no downsampling: this is the
    raw AI-upscaled frame, saved as-is.

    Only meaningful for single-frame, full-screen background art (currently
    just MainMenuBack) - the engine hook only ever reads frame 0, and only on
    the mainmenu's fullscreen (4:3) background path. Run 'hd' on the same
    rel_path first (it leaves the upscaled frame-0 PNG behind in the work
    dir as a side effect, even though its real output is the hd/art/ sidecar
    tree) - retired once A.9's overlay-unification lands
    (docs/GPU_BLITTER_ROADMAP.md Phase 8) and the main menu draws through the
    same hd/art/ path as everything else.
    """
    wd = work_dir_for(rel_path)
    bmps = frame_bmps(wd)
    if not bmps:
        raise RuntimeError(f"No frame BMPs found in {wd}; run 'unpack' first")
    hd_png = bmps[0].with_name(bmps[0].stem + "_hd.png")
    if not hd_png.exists():
        raise RuntimeError(f"Missing {hd_png}; run 'upscale' first")

    basename = Path(rel_path.replace("\\", "/")).with_suffix("").name
    dest = config.HD_OVERLAY_DIR / "menu" / f"{basename}_hd.bmp"
    dest.parent.mkdir(parents=True, exist_ok=True)

    Image.open(hd_png).convert("RGB").save(dest, "BMP")
    print(f"Wrote HD overlay {hd_png.name} -> {dest}")
    return dest


def cmd_hd_slides(force: bool = False) -> None:
    """Upscale the story-slide BMPs (slide_ui.c: death/chapter/credits
    slideshow) the same way as the main-menu background spike - straight
    ESRGAN x4 RGB, no unpack/palette/colour-key step needed since these are
    already loose full-screen BMPs, not .ART sprite sheets. Output uses the
    same flat "hd/<basename>_hd.bmp" convention slide_ui.c already tries
    (tig_video_set_hd_overlay()) before falling back to the vanilla BMP.
    """
    sources: dict[str, Path] = {bmp.stem.lower(): bmp for bmp in sorted(config.SLIDE_DIR.glob("*.bmp"))}

    skipped_collisions = []
    for bmp in sorted(config.SLIDE_DIR_VORMANTOWN.glob("*.bmp")):
        if bmp.stem.lower() in sources:
            skipped_collisions.append(bmp.name)
            continue
        sources[bmp.stem.lower()] = bmp
    if skipped_collisions:
        print(f"Skipping {len(skipped_collisions)} Vormantown slide(s) sharing a basename with the main campaign "
              f"(flat hd/ path can't hold both): {', '.join(skipped_collisions)}")

    tmp_dir = config.WORK_DIR / "_slides_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    for bmp in sorted(sources.values(), key=lambda p: p.stem.lower()):
        dest = config.HD_OVERLAY_DIR / "slides" / f"{bmp.stem}_hd.bmp"
        if dest.is_file() and not force:
            skipped += 1
            continue
        try:
            src_png = tmp_dir / f"{bmp.stem}.png"
            Image.open(bmp).convert("RGB").save(src_png)
            out_png = tmp_dir / f"{bmp.stem}_hd.png"
            run_esrgan(src_png, out_png, config.REALESRGAN_MODEL)
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.open(out_png).convert("RGB").save(dest, "BMP")
            print(f"  {bmp.name} -> hd/slides/{dest.name}")
            done += 1
        except Exception as e:
            print(f"  FAILED {bmp.name}: {e}")
            failed += 1

    print(f"Slides: {done} done, {skipped} skipped (already exist, use --force), {failed} failed")


def cmd_hd_splash(force: bool = False) -> None:
    """Upscale the map-load splash BMPs (gamelib_splash()) the same way as
    the main-menu background and story slides - straight ESRGAN RGB with
    config.REALESRGAN_MODEL (4xNomos8kSC, the general-purpose pick), no
    unpack/palette step needed since these are already loose full-screen
    BMPs. Output uses the same flat "hd/<basename>_hd.bmp" convention
    gamelib_splash() now tries (tig_video_set_hd_overlay()) before falling
    back to the vanilla BMP.
    """
    tmp_dir = config.WORK_DIR / "_splash_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    for bmp in sorted(config.SPLASH_DIR.glob("*.bmp")):
        dest = config.HD_OVERLAY_DIR / "splash" / f"{bmp.stem}_hd.bmp"
        if dest.is_file() and not force:
            skipped += 1
            continue
        try:
            src_png = tmp_dir / f"{bmp.stem}.png"
            Image.open(bmp).convert("RGB").save(src_png)
            out_png = tmp_dir / f"{bmp.stem}_hd.png"
            run_esrgan(src_png, out_png, config.REALESRGAN_MODEL)
            dest.parent.mkdir(parents=True, exist_ok=True)
            hd = Image.open(out_png).convert("RGB")
            if bmp.stem in config.SPLASH_FACES:
                import faces
                hd = faces.patch_splash_faces(bmp.stem, Image.open(bmp), hd)
            hd.save(dest, "BMP")
            print(f"  {bmp.name} -> hd/splash/{dest.name}")
            done += 1
        except Exception as e:
            print(f"  FAILED {bmp.name}: {e}")
            failed += 1

    print(f"Splash: {done} done, {skipped} skipped (already exist, use --force), {failed} failed")


# The splashes' "Loading Arcanum..." lettering came out of the upscale
# lumpy (#228): erased (inpainted) and drawn again. Round 8 pass 11: from
# the vanilla lettering itself instead of a substitute font (neither Goudy
# Bookletter nor Cormorant matched its weight/shapes) - the 1x text's
# coverage over its background, smoothly upscaled and cut with an
# anti-aliased threshold: the original letters with clean HD edges. Per
# splash: (drop shadow, lines of words); a word = (text, HD ink box x0, cap
# top or None, x1, baseline) on the upscale - where to erase and where the
# 1x coverage is read. "..." words get identical dots (a copy of their
# median dot); SPLASH_EVEN_WEIGHT words get every letter's stroke width
# evened out to the word's median (Splash2's NUM read heavier than ARCA).
SPLASH_TEXT = {
    "Splash1": (True, [
        [("Loading", 1825, 1340, 2192, 1421), ("Arcanum", 2224, 1338, 2631, 1421), ("...", 2648, None, 2732, 1421)],
    ]),
    "Splash2": (False, [
        [("Loading", 2121, 1171, 2436, 1219)],
        [("ARCANUM", 2236, 1264, 2672, 1315), ("...", 2689, None, 2744, 1315)],
    ]),
    "Splash3": (False, [
        [("Loading", 106, 83, 407, 131), ("Arcanum", 475, 84, 805, 131), ("...", 821, None, 872, 131)],
    ]),
}
# Pass 12 feedback (#51): replaced by the per-stroke evening below (every
# word of every splash) - the per-letter one left the A's hairline leg and
# M's diagonals thin next to their thick stems.
SPLASH_EVEN_WEIGHT: dict[str, list[str]] = {}
SPLASH_STROKE_GAIN = 0.25  # cut change per HD px of stroke radius off the word's median
SPLASH_STROKE_BASE = 0.3   # coverage cut the stroke skeleton is taken from (keeps hairlines whole)
SPLASH_STROKE_PRUNE = 2.0  # skeleton points thinner than this (HD px radius) are halo spurs
SPLASH_INK_SHARPNESS = 16  # sigmoid slope on the upscaled coverage (AA width); was 10, soft


SPLASH_GRAIN_SHIFT = 140  # HD rows: where the grain for a textured fill comes from (above)


def _splash_word_box(word: tuple, H: int, W: int) -> tuple[slice, slice]:
    _, x0, top, x1, base = word
    y0 = (top if top is not None else base - 60) - 16
    y1 = base + 50  # descenders
    return slice(max(0, y0), min(H, y1)), slice(max(0, x0 - 24), min(W, x1 + 16))


def _stroke_width(ink: np.ndarray) -> float:
    """Mean stroke width of a soft (0..1) letter: 2 x area / perimeter (a
    stroke of length L and width w has area L w and perimeter ~2 L), with
    the perimeter as the ink's total variation - sub-pixel, unlike a
    distance transform's whole-pixel steps."""
    gy, gx = np.gradient(ink)
    perimeter = float(np.hypot(gx, gy).sum())
    return 2.0 * float(ink.sum()) / perimeter if perimeter > 0 else 0.0


def cmd_hd_splash_text(only: str | None = None) -> None:
    """See SPLASH_TEXT. Rebuilt from work/_splash_text_originals/ (the
    upscale, kept on the first run) every time."""
    import cv2
    from scipy import ndimage

    k = SPLASH_INK_SHARPNESS
    for stem, (shadow, lines) in SPLASH_TEXT.items():
        if only is not None and only.lower() not in stem.lower():
            continue
        dest = config.HD_OVERLAY_DIR / "splash" / f"{stem}_hd.bmp"
        backup = config.WORK_DIR / "_splash_text_originals" / dest.name
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dest, backup)
        img = np.asarray(Image.open(backup).convert("RGB")).copy()
        H, W = img.shape[:2]
        van = np.asarray(Image.open(config.SPLASH_DIR / f"{stem}.bmp").convert("RGB")).astype(np.float32)
        s = W // van.shape[1]
        words = [w for line in lines for w in line]

        # erase: the ink around every word, grown a bit - on black anything
        # a little lighter than the box's surroundings (the dim anti-aliased
        # edges too), on the textured scene (Splash1, the one with a shadow)
        # only the bright, grey ink
        mask = np.zeros((H, W), bool)
        lum = img.mean(axis=2)
        sat = img.max(axis=2).astype(int) - img.min(axis=2)
        for word in words:
            box = _splash_word_box(word, H, W)
            ring = np.concatenate([lum[box][0], lum[box][-1], lum[box][:, 0], lum[box][:, -1]])
            if not shadow:  # plain black
                mask[box] |= (lum[box] > np.median(ring) + 22) & (sat[box] < 80)
            else:
                mask[box] |= (lum[box] > 150) & (sat[box] < 50)
        mask = ndimage.binary_dilation(mask, iterations=7)
        filled = cv2.inpaint(np.ascontiguousarray(img[..., ::-1]), mask.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)[..., ::-1].astype(np.float32)
        if shadow:  # textured scene: the smooth fill gets the grain of the ground above
            up = np.roll(img, SPLASH_GRAIN_SHIFT, axis=0).astype(np.float32)
            filled += up - ndimage.gaussian_filter(up, (4, 4, 0))
        m = ndimage.gaussian_filter(mask.astype(np.float32), 1.5)[..., None]
        out = img * (1 - m) + filled * m

        # the vanilla lettering's coverage (0 background .. 1 letter colour)
        vl = van.mean(axis=2)
        vs = van.max(axis=2) - van.min(axis=2)
        cov = np.zeros(vl.shape, np.float32)
        fg = []
        for word in words:
            by, bx = _splash_word_box(word, H, W)
            vb = (slice(by.start // s, by.stop // s + 1), slice(bx.start // s, bx.stop // s + 1))
            reg = vl[vb]
            ring = np.concatenate([reg[0], reg[-1], reg[:, 0], reg[:, -1]])
            bg, top = np.median(ring), np.percentile(reg, 99)
            c = np.clip((reg - bg - 6) / max(1.0, top - bg - 6), 0, 1)
            if shadow:  # textured: only the grey, bright letters
                c = c * (vs[vb] < 60)
            cov[vb] = np.maximum(cov[vb], c)
            fg.append(van[vb][c > 0.9])
        colour = np.concatenate(fg).mean(axis=0)

        big = np.asarray(Image.fromarray(cov, "F").resize((W, H), Image.BICUBIC))
        big = ndimage.gaussian_filter(big, 1.2)
        thresh = np.full((H, W), 0.5, np.float32)

        # even stroke weight: a letter heavier than the word's median gets
        # the (higher) threshold that thins it to the median; lighter ones
        # are left alone
        def soft(b: np.ndarray, t: float) -> np.ndarray:
            return 1 / (1 + np.exp(-(b - t) * k))

        # Pass 12 feedback (#51, "weight consistent for all letters"): per
        # stroke, every word - each pixel's cut is moved by the radius of
        # its nearest stroke centre (skeleton of the coverage cut at
        # SPLASH_STROKE_BASE, halo spurs under SPLASH_STROKE_PRUNE px
        # dropped) against the word's median: hairlines (the A's left leg,
        # M's diagonals) are cut lower and so thicken, heavy stems thin.
        # not on the textured scene (Splash1): its coverage holds the ground's
        # grain, a lower cut brought it up as specks
        if SPLASH_STROKE_GAIN > 0 and not shadow:
            from skimage.morphology import skeletonize
            for word in words:
                if word[0] == "...":
                    continue
                box = _splash_word_box(word, H, W)
                sub = big[box]
                shape = sub > SPLASH_STROKE_BASE
                dist = ndimage.distance_transform_edt(shape)
                skel = skeletonize(shape) & (dist >= SPLASH_STROKE_PRUNE)
                if not skel.any():
                    continue
                med = float(np.median(dist[skel]))
                radius = np.where(skel, dist, 0)
                _, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
                local = ndimage.median_filter(radius, size=5)[iy, ix]
                local = np.where(local > 0, local, radius[iy, ix])
                t = np.clip(0.5 + SPLASH_STROKE_GAIN * (local - med), 0.22, 0.8)
                thresh[box] = ndimage.gaussian_filter(t, 3)
                print(f"  {stem} {word[0]}: stroke radius median {med:.2f} HD px, cuts evened")

        for word in words:
            if word[0] not in SPLASH_EVEN_WEIGHT.get(stem, []):
                continue
            box = _splash_word_box(word, H, W)
            sub = big[box]
            # a letter = parts within 12 HD px of each other (M's strokes
            # come apart at its thin joins)
            labels, n = ndimage.label(ndimage.binary_dilation(sub > 0.5, iterations=6))
            regions = [labels == i for i in range(1, n + 1) if ((labels == i) & (sub > 0.5)).sum() > 200]
            widths = [_stroke_width(soft(sub, 0.5) * r) for r in regions]
            target = float(np.median(widths))
            for region, width in zip(regions, widths):
                x = np.nonzero(region)[1].mean() + box[1].start
                if width <= target * 1.03:
                    print(f"  {stem} {word[0]}: letter at x {x:.0f} width {width:.2f} (target {target:.2f}) kept")
                    continue
                best = min(np.arange(0.5, 0.85, 0.01), key=lambda t: abs(_stroke_width(soft(sub, t) * region) - target))
                thresh[box][region] = best
                print(f"  {stem} {word[0]}: letter at x {x:.0f} width {width:.2f} -> "
                      f"{_stroke_width(soft(sub, best) * region):.2f} (t {best:.2f}, target {target:.2f})")
            thresh[box] = ndimage.gaussian_filter(thresh[box], 4)  # no step between letters

        ink = 1 / (1 + np.exp(-(big - thresh) * k))
        ink[big < 0.12] = 0

        # identical dots: every dot of a "..." word replaced by its median
        # dot, on one shared centre line
        for word in words:
            if word[0] != "...":
                continue
            box = _splash_word_box(word, H, W)
            pad = 6
            sub = np.pad(ink[box], 3 * pad)  # room for the copies near the box edge
            labels, n = ndimage.label(sub > 0.5)
            # only blobs inside the word's own x-range (the box margin can
            # hold the previous letter's tail)
            first_x = word[1] - box[1].start + 3 * pad
            dots = [(labels == i) for i in range(1, n + 1)
                    if (labels == i).sum() > 20 and np.nonzero(labels == i)[1].mean() >= first_x - 4]
            if len(dots) < 2:
                continue
            areas = [d.sum() for d in dots]
            tmpl_i = int(np.argsort(areas)[len(areas) // 2])
            ys, xs = np.nonzero(dots[tmpl_i])
            ty0, ty1, tx0, tx1 = ys.min() - pad, ys.max() + pad + 1, xs.min() - pad, xs.max() + pad + 1
            tmpl = sub[ty0:ty1, tx0:tx1] * ndimage.binary_dilation(dots[tmpl_i], iterations=pad)[ty0:ty1, tx0:tx1]
            tcy, tcx = ys.mean() - ty0, xs.mean() - tx0
            cy = float(np.median([np.nonzero(d)[0].mean() for d in dots]))
            new_sub = sub * ~ndimage.binary_dilation(np.any(dots, axis=0), iterations=pad)
            for d in dots:
                cx = np.nonzero(d)[1].mean()
                oy, ox = int(round(cy - tcy)), int(round(cx - tcx))
                h, w = tmpl.shape
                new_sub[oy:oy + h, ox:ox + w] = np.maximum(new_sub[oy:oy + h, ox:ox + w], tmpl)
            ink[box] = new_sub[3 * pad:-3 * pad, 3 * pad:-3 * pad]
            print(f"  {stem}: {len(dots)} dots -> copies of dot {tmpl_i + 1}")

        if shadow:
            sh = np.roll(np.roll(ndimage.gaussian_filter(ink, 3) * 0.7, 3, 0), 3, 1)[..., None]
            out = out * (1 - sh)
        out = out * (1 - ink[..., None]) + colour * ink[..., None]
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB").save(dest, "BMP")
        print(f"{stem}: lettering redone from the vanilla shapes -> {dest.name}")


def cmd_hd_portraits(force: bool = False) -> None:
    """Upscale character portrait BMPs (portrait.c) with config.REALESRGAN_MODEL
    (4xNomos8kSC - user picked this over x4plus/x4plus-anime after a 3-model
    comparison on ELF1_b/HAM1_b/DWM1_b, see hd_pipeline/comparison/). Unlike
    the slides/splash "hd/<basename>_hd.bmp" overlay convention, portraits
    are drawn inline alongside other composited UI, so portrait_draw_func()
    just tries "hd/<vanilla path>" as a plain BMP - output preserves the
    vanilla "portrait/<name>[_b].bmp" relative layout under hd/, no "_hd"
    suffix.

    Race/gender template portraits (DWM1/ELF1/... - every one has both a
    small and a "_b" 128x128 source in vanilla) skip generating the small
    variant: portrait_draw_func() (2026-09-24) always tries the "_b" HD
    source first regardless of requested display size (looks great
    downscaled), only falling back to a plain-size HD source for named NPC
    portraits (NPCArr, NPCBane, ...) that have no "_b" companion in vanilla
    at all. Generating the small HD variant for a file that HAS a "_b"
    sibling would just be dead weight the engine never loads.

    Face restoration (faces.restore_portrait) was tried 2026-09-26 and
    rejected by the user - ESRGAN portraits stay (comparison/face_restore/).
    """
    tmp_dir = config.WORK_DIR / "_portrait_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = config.HD_OVERLAY_DIR / "portrait"

    done = skipped = failed = 0
    for bmp in sorted(config.PORTRAIT_DIR.glob("*.bmp")):
        if not bmp.stem.endswith("_b") and (bmp.with_name(f"{bmp.stem}_b.bmp")).is_file():
            skipped += 1
            continue
        dest = out_dir / bmp.name
        if dest.is_file() and not force:
            skipped += 1
            continue
        try:
            src_png = tmp_dir / f"{bmp.stem}.png"
            Image.open(bmp).convert("RGB").save(src_png)
            out_png = tmp_dir / f"{bmp.stem}_hd.png"
            run_esrgan(src_png, out_png, config.REALESRGAN_MODEL)
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.open(out_png).convert("RGB").save(dest, "BMP")
            print(f"  {bmp.name} -> hd/portrait/{dest.name}")
            done += 1
        except Exception as e:
            print(f"  FAILED {bmp.name}: {e}")
            failed += 1

    print(f"Portraits: {done} done, {skipped} skipped (already exist, use --force), {failed} failed")


def cmd_hd_movies(force: bool = False) -> None:
    """Upscale the intro/logo Bink videos (SierraLogo.bik, TroikaLogo.bik -
    the only .bik files anywhere in this install) frame-by-frame: extract
    every frame + the audio track, ESRGAN x4 each frame (directory-mode
    batch, same run_esrgan_batch() the main hd pipeline uses), re-encode to
    VP9/WebM (BSD-licensed libvpx, no GPL/patent-pool baggage) with the
    original audio, and write the result back out under the SAME ".bik"
    filename. arcanum-ce's bink_compat now decodes via FFmpeg content-
    probing rather than trusting the extension, so this just works without
    any engine-side change - see config.py's MOVIE_OUTPUT_DIR docstring for
    why that specific deploy path (not data/TIGCache/) is the correct one.
    """
    bik_files = sorted(config.MOVIE_DIR.glob("*.bik"))
    if not bik_files:
        print(f"No .bik files found in {config.MOVIE_DIR}")
        return

    config.MOVIE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    for bik in bik_files:
        dest = config.MOVIE_OUTPUT_DIR / bik.name
        if dest.is_file() and not force:
            skipped += 1
            continue
        try:
            tmp_dir = config.WORK_DIR / "_movies_tmp" / bik.stem
            frames_src = tmp_dir / "frames_src"
            frames_up = tmp_dir / "frames_up"
            for d in (frames_src, frames_up):
                if d.is_dir():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
            audio_wav = tmp_dir / "audio.wav"

            probe = subprocess.run(
                [str(config.FFPROBE_EXE), "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(bik)],
                capture_output=True, text=True,
            )
            fps_str = probe.stdout.strip() or "15/1"
            num, _, den = fps_str.partition("/")
            fps = float(num) / float(den or 1)

            run_ffmpeg = lambda args: subprocess.run(  # noqa: E731
                [str(config.FFMPEG_EXE), "-hide_banner", "-loglevel", "error", "-y", *args],
                capture_output=True, text=True,
            )

            result = run_ffmpeg(["-i", str(bik), str(frames_src / "frame_%06d.png")])
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg frame extract failed: {result.stderr}")

            has_audio = subprocess.run(
                [str(config.FFPROBE_EXE), "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=index", "-of", "csv=p=0", str(bik)],
                capture_output=True, text=True,
            ).stdout.strip() != ""
            if has_audio:
                result = run_ffmpeg(["-i", str(bik), "-vn", str(audio_wav)])
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg audio extract failed: {result.stderr}")

            run_esrgan_batch(frames_src, frames_up, config.MOVIE_MODEL)

            encode_args = ["-framerate", str(fps), "-i", str(frames_up / "frame_%06d.png")]
            if has_audio:
                encode_args += ["-i", str(audio_wav), "-c:a", "libopus", "-shortest"]
            # -f webm forced explicitly: ffmpeg can't infer a muxer from the
            # ".bik" extension (unknown to it) even though the content is a
            # normal WebM/VP9 container - the filename stays ".bik" because
            # arcanum-ce's bink_compat (first_party/bink_compat) opens by
            # content-probing, not by trusting the extension.
            # Pure 4x ESRGAN output, no downsample (user call: the jagged
            # letter edges visible full-screen are baked into the ORIGINAL
            # 800x400 source's own aliasing, not something to fix by
            # softening the remaster - a Lanczos-downsample-to-2x pass was
            # tried and did remove them, but was rejected in favor of
            # keeping full 4x detail).
            encode_args += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "30", "-f", "webm", str(dest)]
            result = run_ffmpeg(encode_args)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg re-encode failed: {result.stderr}")

            print(f"  {bik.name} -> {dest} ({len(list(frames_src.glob('*.png')))} frames, {fps:.2f}fps, "
                  f"audio={'yes' if has_audio else 'no'})")
            done += 1
        except Exception as e:
            print(f"  FAILED {bik.name}: {e}")
            failed += 1

    print(f"Movies: {done} done, {skipped} skipped (already exist, use --force), {failed} failed")


# ---------------------------------------------------------------------------
# HD PNG sidecars (Phase 8: native-res UI rendering)
#
# Contract with the engine (tig_art_hd_frame_texture() in art.c):
#   <ARCANUM_ROOT>/hd/art/<rel path of .ART, no extension>/r<rot>_f<frame>.png
# e.g. art/interface/IntTop.ART frame 0 -> hd/art/interface/IntTop/r0_f0.png.
# Truecolor RGBA, STRAIGHT (non-premultiplied) alpha, dimensions exactly
# 4x the vanilla frame. Alpha comes from the vanilla colour-key (palette
# index 0), not from ESRGAN. Palette 0 only. Rotation-major frame order:
# linear index i in the .ART = rot * num_frames + frame (art.c sub_51B710).
# The vanilla .ART stays authoritative for metrics/hit-tests; these PNGs are
# only ever drawn 1:1 into a 4x render target.
# ---------------------------------------------------------------------------

HD_SCALE = 4
# Frames below this area used to route to Real-CUGAN (assumed too little
# context for ESRGAN). cmd_hd no longer does: the "pixel soup" behind that
# rule was the ncnn directory-mode corruption (see verify_batch), and
# single-file ESRGAN matched or beat CUGAN on 60 small interface frames
# (comparison/small_frames/) - CUGAN kept dither as a mesh. One model for
# everything; CUGAN is left for FORCE_CUGAN_ASSETS and the soup fallback.
# Still used by the tile self-wrap route (whose composites are never small)
# and to find the frames that took the old route (hd-requeue).
HD_SMALL_FRAME_PX = 48


def hd_out_dir(rel_path: str) -> Path:
    rel = Path(rel_path.replace("\\", "/"))
    return config.HD_OVERLAY_DIR / rel.parent / rel.name.rsplit(".", 1)[0]


def read_ini_frame_count(ini_path: Path) -> tuple[int, bool]:
    """(num_frames, animated) from the .ini artconverter writes: 'frames:' is
    the linear frame count (already x8 for 8-rotation art), and 8-rotation
    art is recognisable by its 'frame N_R:' block labels."""
    text = ini_path.read_text(errors="replace")
    m = re.search(r"^frames:\s*(\d+)", text, re.M)
    if not m:
        raise RuntimeError(f"No 'frames:' line in {ini_path}")
    total = int(m.group(1))
    animated = re.search(r"^frame \d+_\d+:", text, re.M) is not None
    return (total // 8 if animated else total), animated


def hd_frame_bmps(wd: Path, basename: str, num_frames: int, animated: bool) -> list[tuple[int, int, Path]]:
    """[(rot, frame, bmp)] in linear file order (see SaveBMPS in artconverter.cpp)."""
    out = []
    total = num_frames * (8 if animated else 1)
    for i in range(total):
        name = f"{basename}_{i // 8}{i % 8}.bmp" if animated else f"{basename}_{i}.bmp"
        p = wd / name
        if not p.is_file():
            raise RuntimeError(f"Missing frame BMP {p}")
        out.append((i // num_frames, i % num_frames, p))
    return out


def inpaint_colorkey(rgb: np.ndarray, key: np.ndarray, max_iter: int = 64) -> np.ndarray:
    """Replace colour-key pixels with the nearest non-key colour (iterative
    8-neighbour flood) so the upscaler never sees the flat key colour and
    can't bleed it into edge pixels. Pure numpy - no scipy available."""
    img = rgb.astype(np.float32).copy()
    todo = key.copy()
    if not todo.any():
        return img
    if todo.all():
        return img
    shifts = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for _ in range(max_iter):
        if not todo.any():
            break
        acc = np.zeros_like(img)
        cnt = np.zeros(todo.shape, dtype=np.float32)
        for dy, dx in shifts:
            src_ok = np.roll(~todo, (dy, dx), axis=(0, 1))
            src_img = np.roll(img, (dy, dx), axis=(0, 1))
            # np.roll wraps; mask the wrapped rows/cols out.
            if dy == 1:
                src_ok[0, :] = False
            elif dy == -1:
                src_ok[-1, :] = False
            if dx == 1:
                src_ok[:, 0] = False
            elif dx == -1:
                src_ok[:, -1] = False
            take = todo & src_ok
            acc[take] += src_img[take]
            cnt[take] += 1
        filled = todo & (cnt > 0)
        img[filled] = acc[filled] / cnt[filled][:, None]
        todo &= ~filled
    if todo.any():
        img[todo] = img[~key].mean(axis=0)
    return img


# Tried a 3x3 median pre-pass before ESRGAN to fix palette-quantization grain
# on smooth-gradient button states (Social_But.ART frames 0/2 - confirmed
# real, neither GARBAGE_ROUGHNESS_THRESHOLD nor STRUCTURAL_DIFF_THRESHOLD
# caught it). Reverted: on small icons with multiple separate thin shapes
# close together (Ammo_Icon_Bullets.ART - a cluster of individual bullets a
# couple px apart at 37x24), the same median filter erases the thin dark
# gaps *between* objects before ESRGAN ever sees them, fusing distinct
# shapes into one blob - a much worse failure than the grain it fixed, and
# not reliably distinguishable from the good case ahead of time (tried two
# different pixel-statistics classifiers, both misranked the known-good vs
# known-bad calibration pair). Applying it unconditionally to the whole
# pipeline is a net loss; leaving the smooth-gradient grain case as a known,
# rare, manually-fixable issue (redo that one .ART individually) instead.


# Selective de-dither. ESRGAN (Nomos8kSC) reads a strict 1-px checkerboard
# dither as heavy noise and paints the whole frame over with a washed-out
# blur (NextBut's hover/pressed arrow: the entire button came out a pale
# smear); Real-CUGAN keeps it as a visible mesh (cncl_big's X). A light blur
# of just the dithered area first gives the solid gradient the dither was
# standing in for. Detection is deliberately strict - a pixel counts only if
# its 4 neighbours agree with each other, its diagonals agree with it, and
# the two differ - so noisy texture that merely looks dithered (chainmail,
# fur, bark) is left alone; only areas where such pixels are the majority
# get blurred (comparison/dedither/).
DEDITHER_SIGMA = 0.7
DEDITHER_SHARE = (0.3, 0.55)


def dither_weight(rgb: np.ndarray, key: np.ndarray) -> np.ndarray:
    """0..1 per pixel: how much of the neighbourhood is strict checkerboard."""
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    p = np.pad(lum, 1, mode="edge")
    n = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    d = np.stack([p[:-2, :-2], p[:-2, 2:], p[2:, :-2], p[2:, 2:]])
    contrast = np.abs(lum - n.mean(0))
    checker = (
        (contrast > 6.0)
        & (n.max(0) - n.min(0) < 0.5 * contrast)
        & (d.max(0) - d.min(0) < 0.5 * contrast)
        & (np.abs(lum - d.mean(0)) < 0.35 * contrast)
        & ~key
    )
    lo, hi = DEDITHER_SHARE
    share = gaussian_blur_2d(checker.astype(np.float32), 1.5)
    return np.clip((share - lo) / (hi - lo), 0.0, 1.0)


def dedither(rgb: np.ndarray, key: np.ndarray) -> np.ndarray:
    """uint8 RGB -> uint8 RGB with only its checkerboard-dithered areas blurred."""
    w = dither_weight(rgb, key)
    if w.max() == 0.0:
        return rgb
    w = w[..., None]
    blur = np.stack([gaussian_blur_2d(rgb[..., c].astype(np.float32), DEDITHER_SIGMA) for c in range(3)], axis=2)
    return np.clip(rgb * (1.0 - w) + blur * w + 0.5, 0, 255).astype(np.uint8)


def checker_average(rgb: np.ndarray, key: np.ndarray) -> np.ndarray:
    """uint8 RGB -> uint8 RGB with each strict-checkerboard pixel replaced by
    the mean of itself and its 4 neighbours' mean - exactly the 50/50 blend
    the dither stands for. Unlike dedither()'s Gaussian it leaves the shape
    edges of a dithered hover glow sharp (the whole-disc glows of the HUD
    round buttons came out as mush through dedither())."""
    w = dither_weight(rgb, key)
    if w.max() == 0.0:
        return rgb
    f = rgb.astype(np.float32)
    p = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="edge")
    kp = np.pad(key, 1, mode="edge")
    s = np.zeros_like(f)
    c = np.zeros(f.shape[:2], np.float32)
    for n, k in ((p[:-2, 1:-1], kp[:-2, 1:-1]), (p[2:, 1:-1], kp[2:, 1:-1]), (p[1:-1, :-2], kp[1:-1, :-2]), (p[1:-1, 2:], kp[1:-1, 2:])):
        s += n * (~k)[..., None]
        c += ~k
    avg = 0.5 * f + 0.5 * s / np.maximum(c, 1)[..., None]
    w = w[..., None]
    return np.clip(f * (1.0 - w) + avg * w + 0.5, 0, 255).astype(np.uint8)


# Categories whose sidecars get smooth_alpha() (the AA pass) as they are
# written. Not wall/roof/facade/tile: those pieces butt against each other,
# and a softened edge on both sides of a joint would show as a seam.
ALPHA_SMOOTH_CATEGORIES = {
    "interface", "item", "critter", "monster", "unique_npc", "scenery",
    "eye_candy", "container", "portal", "light",
}


def alpha_backup_path(dest: Path) -> Path:
    rel = dest.relative_to(config.HD_OVERLAY_DIR / "art")
    return config.WORK_DIR / "_alpha_originals" / rel


def write_sidecar(rel_path: str, dest: Path, rgba: Image.Image) -> None:
    """Save one frame's raw RGBA sidecar, applying the AA pass (smooth_alpha)
    where its category takes it. The raw frame then goes to
    work/_alpha_originals/ (what hd-smooth-alpha reruns start from); any
    stale backup of a frame that isn't smoothed is dropped."""
    category = Path(rel_path.replace("\\", "/")).parts[1]
    arr = np.asarray(rgba.convert("RGBA"))
    alpha = arr[..., 3]
    matte = dest.parent in {hd_out_dir(piece) for piece, _, _ in BACKGROUND_MATTE_PIECES}
    backup = alpha_backup_path(dest)
    if category in ALPHA_SMOOTH_CATEGORIES and not matte and alpha.min() < 255 and alpha.max() > 0:
        backup.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, "RGBA").save(backup, "PNG", compress_level=6)
        out = arr.copy()
        out[..., 3] = np.clip(smooth_alpha(alpha / 255.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(out, "RGBA").save(dest, "PNG", compress_level=6)
    else:
        if backup.exists():
            backup.unlink()
        Image.fromarray(arr, "RGBA").save(dest, "PNG", compress_level=6)


def render_bar(current: int, total: int, width: int = 16) -> str:
    filled = width if not total else min(width, round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def cmd_hd(rel_path: str, model: str | None = None, force: bool = False, quiet: bool = False, progress_cb=None) -> Path:
    """Unpack -> per-frame: inpaint key, ESRGAN x4, alpha from key mask ->
    write RGBA PNG sidecars under hd/art/. Does not touch data/ or the .ART."""
    out_dir = hd_out_dir(rel_path)
    if out_dir.is_dir() and any(out_dir.glob("r*_f*.png")) and not force:
        if not quiet:
            print(f"Skipping {rel_path}: {out_dir} already populated (use --force)")
        return out_dir

    wd = cmd_unpack(rel_path, quiet=quiet)
    basename = Path(rel_path.replace("\\", "/")).name.rsplit(".", 1)[0]
    ini_path = wd / (basename + ".ini")
    num_frames, animated = read_ini_frame_count(ini_path)
    frames = hd_frame_bmps(wd, basename, num_frames, animated)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("r*_f*.png"):
        old.unlink()

    force_cugan = rel_path.replace("\\", "/") in config.FORCE_CUGAN_ASSETS
    esrgan_model = model or config.REALESRGAN_MODEL

    # Pass 1: stage every frame's colour-key-inpainted source into one of two
    # per-file directories (grouped by which upscaler it needs), instead of
    # calling out to ncnn-vulkan per frame - see run_esrgan_batch/
    # run_realcugan_batch for why (one directory-mode launch per group
    # instead of one launch per frame, ~15x fewer subprocess/model-load
    # round trips). Everything goes to ESRGAN (de-dithered first) except
    # FORCE_CUGAN_ASSETS; see HD_SMALL_FRAME_PX and dedither().
    esrgan_in, esrgan_out = wd / "_esrgan_in", wd / "_esrgan_out"
    cugan_in, cugan_out = wd / "_cugan_in", wd / "_cugan_out"
    esrgan_in.mkdir(exist_ok=True)
    cugan_in.mkdir(exist_ok=True)

    metas = []
    for rot, frame, bmp in frames:
        w, h = read_bmp_dims(bmp)
        h = abs(h)
        indices = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
        palette = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
        key = indices == 0
        rgb = palette[indices]

        src_rgb = np.clip(inpaint_colorkey(rgb, key), 0, 255).astype(np.uint8)
        route = "cugan" if force_cugan else "esrgan"
        if route == "esrgan":
            src_rgb = dedither(src_rgb, key)
        key_name = f"{rot}_{frame}.png"
        staging_dir = cugan_in if route == "cugan" else esrgan_in
        src_png = staging_dir / key_name
        Image.fromarray(src_rgb, "RGB").save(src_png)

        # NB: spatial uniformity, not "std()==0 over the whole HxWxC array" -
        # that mixes the spatial and channel axes, so a flat 1x1/1x2 dummy
        # frame in any non-grayscale colour (R != G != B) was never flagged
        # uniform even though every pixel is identical, sending it down the
        # blank-output-retry-then-fail path for content that can't produce
        # real upscaler variance in the first place.
        flat = src_rgb.reshape(-1, src_rgb.shape[-1])
        metas.append(dict(
            rot=rot, frame=frame, bmp=bmp, w=w, h=h, key=key, route=route, key_name=key_name,
            src_uniform=bool(np.all(flat == flat[0])),
        ))

    if any(m["route"] == "esrgan" for m in metas):
        run_esrgan_batch(esrgan_in, esrgan_out, esrgan_model)
    if any(m["route"] == "cugan" for m in metas):
        run_realcugan_batch(cugan_in, cugan_out)

    # Pass 2: pull each frame's batch output back out, apply the same
    # per-frame safety nets as before (blank-output retry, pixel-soup ->
    # Real-CUGAN fallback), then alpha + save.
    for i, m in enumerate(metas, 1):
        rot, frame, bmp, w, h, key, route, key_name, src_uniform = (
            m["rot"], m["frame"], m["bmp"], m["w"], m["h"], m["key"], m["route"], m["key_name"], m["src_uniform"],
        )
        expected_size = (w * HD_SCALE, h * HD_SCALE)
        pixel_soup_fallback = False

        if route == "cugan":
            hd = load_and_validate(cugan_out / key_name, expected_size, "realcugan batch")
            # A uniform source (e.g. PixelDummy.ART - a literal 1x2 solid-black
            # placeholder) legitimately upscales to a uniform output - only
            # treat exact-zero variance as a failed GPU write when the source
            # itself wasn't already flat.
            if not src_uniform and is_blank_output(hd):
                retry_png = bmp.with_name(bmp.stem + "_hd_retry.png")
                run_realcugan(cugan_in / key_name, retry_png)
                hd = load_and_validate(retry_png, expected_size, "realcugan retry")
                if is_blank_output(hd):
                    raise RuntimeError(f"realcugan produced blank output twice for {bmp}")
        else:
            hd = load_and_validate(esrgan_out / key_name, expected_size, "realesrgan batch")
            if not src_uniform and is_blank_output(hd):
                retry_png = bmp.with_name(bmp.stem + "_hd_retry.png")
                run_esrgan(esrgan_in / key_name, retry_png, esrgan_model)
                hd = load_and_validate(retry_png, expected_size, "realesrgan retry")
                if is_blank_output(hd):
                    raise RuntimeError(f"realesrgan produced blank output twice for {bmp}")
            pixel_soup_fallback = roughness(hd) > PIXEL_SOUP_ROUGHNESS_THRESHOLD
            if pixel_soup_fallback:
                # Real-CUGAN beats the Lanczos+ESRGANx4 hybrid here too - confirmed
                # on a 50-sample sweep (ELM-FIRE-Wal/Generic-Smoke/etc: hybrid still
                # desaturates/blobs the glow, CUGAN keeps it clean). Also doubles as
                # a safety net for tile-race glitches under batch GPU contention
                # (I_BeautyMedalion: clean under light load, pixel soup only under
                # heavy concurrent load) - roughness re-check below still applies.
                cugan_png = bmp.with_name(bmp.stem + "_hd_cugan.png")
                run_realcugan(esrgan_in / key_name, cugan_png)
                hd = load_and_validate(cugan_png, expected_size, "realcugan fallback")
                if is_blank_output(hd):
                    raise RuntimeError(f"realcugan fallback produced blank output for {bmp}")
                if not quiet:
                    print(f"  Pixel soup from direct ESRGAN on {bmp.name}, used Real-CUGAN fallback instead")

        # Alpha: hq4x edge-directed upscale of the binary key mask (not a
        # blur/SDF - pattern-matches the local 3x3 neighborhood like the
        # classic pixel-art scalers, so it converts a jagged staircase into
        # a clean diagonal run instead of softening it). Confirmed better
        # than bicubic and a from-scratch SDF-distance-field approach across
        # both a large plate (125x142 native) and small buttons (11x15,
        # 32x32) - SDF's fixed-width blur band melted small icons, hq4x held
        # up at every size tested since it has no absolute-pixel-width
        # parameter to mismatch against source resolution.
        mask_rgb = Image.fromarray(np.where(key, 0, 255).astype(np.uint8), "L").convert("RGB")
        alpha_rgb = hqx.hq4x(mask_rgb)
        if alpha_rgb.size != expected_size:
            raise RuntimeError(f"hq4x produced wrong size {alpha_rgb.size} for {bmp} (expected {expected_size})")
        alpha = alpha_rgb.convert("L")
        rgba = hd.copy()
        rgba.putalpha(alpha)
        dest = out_dir / f"r{rot}_f{frame}.png"
        write_sidecar(rel_path, dest, rgba)
        if not quiet:
            model_label = "realcugan" if (route == "cugan" or pixel_soup_fallback) else esrgan_model
            print(f"  {bmp.name} -> {dest.relative_to(config.HD_OVERLAY_DIR)} ({rgba.size[0]}x{rgba.size[1]}, {model_label})")
        if progress_cb is not None:
            progress_cb(i, len(frames))

    return out_dir


def cmd_hd_tile_selfwrap(model: str | None = None) -> None:
    """Reprocess every ground `tile` ART (art/tile/*.ART, 1255 files, each a
    single rotation/frame) using a self-wrap composite instead of the
    independent per-file upscaling they already got earlier this session.
    Ground tiles are a genuine self-repeating grid (SectorTileList.art_ids
    [4096] - the same small texture really tiles against itself at runtime,
    unlike wall/facade/roof's individually-placed objects), so tiling the
    SAME source image 3x3, upscaling that composite once, and cropping just
    the center cell gives the model real (self-)neighbour context on all
    four edges - the same "avoid independent-file seam risk" reasoning as
    the townmap/worldmap composite technique, just with a synthetic
    self-composite instead of a real map layout.

    ALWAYS overwrites hd/art/tile/<name>/r{rot}_f{frame}.png - this replaces
    the earlier independent-upscale output, it does not skip-if-exists.
    """
    esrgan_model = model or config.REALESRGAN_MODEL
    rel_paths = find_category_files("tile")
    if not rel_paths:
        raise RuntimeError("No .ART files found for art/tile/")

    tmp_dir = config.WORK_DIR / "_tile_selfwrap_tmp"
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    esrgan_in, esrgan_out = tmp_dir / "_esrgan_in", tmp_dir / "_esrgan_out"
    cugan_in, cugan_out = tmp_dir / "_cugan_in", tmp_dir / "_cugan_out"
    esrgan_in.mkdir(parents=True, exist_ok=True)
    cugan_in.mkdir(parents=True, exist_ok=True)

    # Pass 1: unpack every tile, inpaint its single frame, build the 3x3
    # self-wrap composite, stage it for one combined batch upscale (same
    # directory-mode-avoids-per-launch-setup-cost reasoning as cmd_hd()'s
    # per-file batching - see run_esrgan_batch's comment - just spanning all
    # 1255 files' frames in one go instead of one file's rotations).
    metas = []
    print(f"Tile self-wrap: staging {len(rel_paths)} tile(s)...", flush=True)
    for rel_path in rel_paths:
        wd = cmd_unpack(rel_path, quiet=True)
        basename = Path(rel_path.replace("\\", "/")).name.rsplit(".", 1)[0]
        ini_path = wd / (basename + ".ini")
        num_frames, animated = read_ini_frame_count(ini_path)
        frames = hd_frame_bmps(wd, basename, num_frames, animated)

        force_cugan = rel_path.replace("\\", "/") in config.FORCE_CUGAN_ASSETS
        for rot, frame, bmp in frames:
            w, h = read_bmp_dims(bmp)
            h = abs(h)
            indices = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
            palette = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
            key = indices == 0
            rgb = palette[indices]
            src_rgb = np.clip(inpaint_colorkey(rgb, key), 0, 255).astype(np.uint8)

            tiled = np.tile(src_rgb, (3, 3, 1))
            is_small = w * h < HD_SMALL_FRAME_PX * HD_SMALL_FRAME_PX
            route = "cugan" if (force_cugan or is_small) else "esrgan"
            key_name = f"{basename}_r{rot}_f{frame}.png"
            staging_dir = cugan_in if route == "cugan" else esrgan_in
            Image.fromarray(tiled, "RGB").save(staging_dir / key_name)

            metas.append(dict(
                rel_path=rel_path, rot=rot, frame=frame, w=w, h=h, key=key,
                route=route, key_name=key_name, bmp=bmp,
            ))

    if any(m["route"] == "esrgan" for m in metas):
        n = sum(1 for m in metas if m["route"] == "esrgan")
        print(f"Tile self-wrap: running realesrgan batch on {n} composite(s)...", flush=True)
        run_esrgan_batch(esrgan_in, esrgan_out, esrgan_model)
    if any(m["route"] == "cugan" for m in metas):
        n = sum(1 for m in metas if m["route"] == "cugan")
        print(f"Tile self-wrap: running realcugan batch on {n} composite(s)...", flush=True)
        run_realcugan_batch(cugan_in, cugan_out)

    # Pass 2: crop the center cell out of each upscaled composite, apply
    # alpha from the single tile's own colour-key mask (same hq4x technique
    # as cmd_hd() - the mask only needs the real tile's own edges, not the
    # synthetic self-wrap composite), always overwrite the existing output.
    ok = bad = 0
    for i, m in enumerate(metas, 1):
        rel_path, rot, frame, w, h, key, route, key_name, bmp = (
            m["rel_path"], m["rot"], m["frame"], m["w"], m["h"], m["key"], m["route"], m["key_name"], m["bmp"],
        )
        expected_tiled_size = (w * 3 * HD_SCALE, h * 3 * HD_SCALE)
        try:
            if route == "cugan":
                hd_tiled = load_and_validate(cugan_out / key_name, expected_tiled_size, "realcugan batch (tile self-wrap)")
                if is_blank_output(hd_tiled):
                    retry_png = (cugan_in / key_name).with_name(Path(key_name).stem + "_retry.png")
                    run_realcugan(cugan_in / key_name, retry_png)
                    hd_tiled = load_and_validate(retry_png, expected_tiled_size, "realcugan retry (tile self-wrap)")
                    if is_blank_output(hd_tiled):
                        raise RuntimeError(f"realcugan produced blank output twice for {bmp}")
            else:
                hd_tiled = load_and_validate(esrgan_out / key_name, expected_tiled_size, "realesrgan batch (tile self-wrap)")
                if is_blank_output(hd_tiled):
                    retry_png = (esrgan_in / key_name).with_name(Path(key_name).stem + "_retry.png")
                    run_esrgan(esrgan_in / key_name, retry_png, esrgan_model)
                    hd_tiled = load_and_validate(retry_png, expected_tiled_size, "realesrgan retry (tile self-wrap)")
                    if is_blank_output(hd_tiled):
                        raise RuntimeError(f"realesrgan produced blank output twice for {bmp}")

            box = (w * HD_SCALE, h * HD_SCALE, 2 * w * HD_SCALE, 2 * h * HD_SCALE)
            hd = hd_tiled.crop(box)

            mask_rgb = Image.fromarray(np.where(key, 0, 255).astype(np.uint8), "L").convert("RGB")
            alpha_rgb = hqx.hq4x(mask_rgb)
            expected_size = (w * HD_SCALE, h * HD_SCALE)
            if alpha_rgb.size != expected_size:
                raise RuntimeError(f"hq4x produced wrong size {alpha_rgb.size} for {bmp} (expected {expected_size})")
            alpha = alpha_rgb.convert("L")
            rgba = hd.copy()
            rgba.putalpha(alpha)

            out_dir = hd_out_dir(rel_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"r{rot}_f{frame}.png"
            rgba.save(dest, "PNG", optimize=True)
            ok += 1
            print(f"[{i}/{len(metas)}] {rel_path} -> {dest.relative_to(config.HD_OVERLAY_DIR)}", flush=True)
        except Exception as e:
            bad += 1
            print(f"[{i}/{len(metas)}] FAILED: {rel_path}: {e}", flush=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Tile self-wrap: {ok} succeeded, {bad} failed out of {len(metas)}.", flush=True)


def read_townmap_info(tmi_path: Path) -> dict:
    """Parse a 48-byte TownMapInfo (arcanum-ce src/game/townmap.h): 8x int32,
    then int64 loc, float32 scale, int32 padding. Confirmed against Ashbury's
    real .tmi (width=20480 height=10240 num_hor=num_vert=80 scale=0.25 ->
    width/num_hor*scale = 64, matching its actual tile bitmaps exactly)."""
    data = tmi_path.read_bytes()
    version, _field_4, _map, width, height, num_hor_tiles, num_vert_tiles, _pad1c, _loc, scale, _pad2c = \
        struct.unpack("<8i q f i", data)
    if version != 2:
        raise RuntimeError(f"unexpected TownMapInfo version {version} in {tmi_path}")
    return dict(width=width, height=height, num_hor_tiles=num_hor_tiles, num_vert_tiles=num_vert_tiles, scale=scale)


def cmd_hd_townmap_one(town_dir: Path, esrgan_model: str) -> Path:
    """Assemble one town's populated automap tiles (cropped to their bounding
    box, not the full nominal grid - most towns only use a fraction of it)
    into a single composite, upscale that ONE image, then slice the result
    back into per-tile HD BMPs. Avoids the independent-per-tile seam problem
    entirely: the model sees real neighbour pixels at every former tile
    boundary, because the composite genuinely reconstructs the town's real
    layout (index = col + num_hor_tiles*row, a plain row-major grid, not the
    diagonal/staggered layout iso world tiles use)."""
    town = town_dir.name
    info = read_townmap_info(town_dir / f"{town}.tmi")
    num_hor_tiles = info["num_hor_tiles"]

    existing: dict[int, Path] = {}
    tile_w = tile_h = None
    for p in sorted(town_dir.glob(f"{town}??????.bmp")):
        m = re.fullmatch(re.escape(town) + r"(\d{6})\.bmp", p.name)
        if m is None:
            continue
        idx = int(m.group(1))
        existing[idx] = p
        if tile_w is None:
            tile_w, tile_h = read_bmp_dims(p)
            tile_h = abs(tile_h)

    if not existing:
        raise RuntimeError(f"no tile bmps found under {town_dir}")

    cols = [idx % num_hor_tiles for idx in existing]
    rows = [idx // num_hor_tiles for idx in existing]
    min_col, min_row = min(cols), min(rows)
    grid_w = max(cols) - min_col + 1
    grid_h = max(rows) - min_row + 1

    composite = Image.new("RGB", (grid_w * tile_w, grid_h * tile_h))
    for idx, p in existing.items():
        col = idx % num_hor_tiles - min_col
        row = idx // num_hor_tiles - min_row
        composite.paste(Image.open(p).convert("RGB"), (col * tile_w, row * tile_h))

    tmp_dir = config.WORK_DIR / "_townmap_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src_png = tmp_dir / f"{town}.png"
    out_png = tmp_dir / f"{town}_hd.png"
    composite.save(src_png)
    run_esrgan(src_png, out_png, esrgan_model)

    hd_composite = load_and_validate(out_png, (grid_w * tile_w * HD_SCALE, grid_h * tile_h * HD_SCALE), "realesrgan (townmap composite)")

    out_dir = config.TOWNMAP_OUTPUT_DIR / town
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, p in existing.items():
        col = idx % num_hor_tiles - min_col
        row = idx // num_hor_tiles - min_row
        box = (col * tile_w * HD_SCALE, row * tile_h * HD_SCALE,
            (col + 1) * tile_w * HD_SCALE, (row + 1) * tile_h * HD_SCALE)
        hd_composite.crop(box).save(out_dir / p.name, "BMP")

    src_png.unlink(missing_ok=True)
    out_png.unlink(missing_ok=True)

    return out_dir


def cmd_hd_townmap(name: str | None = None, force: bool = False, model: str | None = None) -> None:
    """Batch driver for cmd_hd_townmap_one() across every town under
    config.TOWNMAP_ROOTS (Arcanum + Vormantown). A Vormantown town whose name
    collides with a base-game one is skipped - same defensive rule
    cmd_hd_slides() uses for its Vormantown collisions, since the engine's
    flat "hd/townmap/<name>/" path can't disambiguate by module either."""
    esrgan_model = model or config.REALESRGAN_MODEL

    seen_names: set[str] = set()
    towns: list[Path] = []
    for root in config.TOWNMAP_ROOTS:
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not (d / f"{d.name}.tmi").is_file():
                continue
            if d.name in seen_names:
                print(f"  Skipping {root.name}/{d.name}: name collides with an already-queued town")
                continue
            seen_names.add(d.name)
            towns.append(d)

    if name is not None:
        towns = [d for d in towns if d.name == name]
        if not towns:
            raise RuntimeError(f"No town named '{name}' found under any of config.TOWNMAP_ROOTS")

    log_name = "hd_townmap"
    done = set() if force else load_done_set(log_name)
    pending = [d for d in towns if d.name not in done]
    skipped_done = len(towns) - len(pending)
    print(f"HD townmap: {len(towns)} total, {skipped_done} already done, {len(pending)} queued.", flush=True)

    ok = bad = 0
    for town_dir in pending:
        town = town_dir.name
        try:
            out_dir = cmd_hd_townmap_one(town_dir, esrgan_model)
            print(f"[{ok + bad + 1}/{len(pending)}] {town} -> {out_dir.relative_to(config.HD_OVERLAY_DIR)} ({len(list(town_dir.glob(f'{town}??????.bmp')))} tiles)")
            ok += 1
            with open(batch_log_dir() / f"{log_name}.done.txt", "a", encoding="utf-8") as f:
                f.write(town + "\n")
        except Exception as e:
            bad += 1
            print(f"[{ok + bad}/{len(pending)}] FAILED: {town}: {e}")
            with open(batch_log_dir() / f"{log_name}.failed.txt", "a", encoding="utf-8") as f:
                f.write(f"{town}\t{e}\n")

    print(f"Townmap: {ok} done, {skipped_done} skipped (already done, use --force), {bad} failed")


def cmd_hd_worldmap_tiles(esrgan_model: str) -> Path:
    """Composite the overworld's SmallMapChunks grid into one image, upscale
    once, slice back - the same composite-upscale-slice technique as
    cmd_hd_townmap_one(), generalized to the single base-game overworld map
    (config.WORLDMAP_GRID/ROOT/TILE_BASENAME, from WorldMap.mes). Avoids the
    same independent-per-tile seam problem townmap had, for the same reason:
    the model sees real neighbour pixels at every former tile boundary."""
    root = config.WORLDMAP_ROOT
    num_hor, _num_vert = config.WORLDMAP_GRID
    basename = config.WORLDMAP_TILE_BASENAME

    existing: dict[int, Path] = {}
    tile_w = tile_h = None
    for p in sorted(root.glob(f"{basename}*.bmp")):
        m = re.fullmatch(re.escape(basename) + r"(\d{3})\.bmp", p.name)
        if m is None:
            continue
        idx = int(m.group(1)) - 1  # vanilla filenames are 1-indexed; normalize to 0-indexed row-major
        existing[idx] = p
        if tile_w is None:
            tile_w, tile_h = read_bmp_dims(p)
            tile_h = abs(tile_h)

    if not existing:
        raise RuntimeError(f"no worldmap tile bmps found under {root}")

    cols = [idx % num_hor for idx in existing]
    rows = [idx // num_hor for idx in existing]
    min_col, min_row = min(cols), min(rows)
    grid_w = max(cols) - min_col + 1
    grid_h = max(rows) - min_row + 1

    composite = Image.new("RGB", (grid_w * tile_w, grid_h * tile_h))
    for idx, p in existing.items():
        col = idx % num_hor - min_col
        row = idx // num_hor - min_row
        composite.paste(Image.open(p).convert("RGB"), (col * tile_w, row * tile_h))

    tmp_dir = config.WORK_DIR / "_worldmap_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src_png = tmp_dir / "worldmap.png"
    out_png = tmp_dir / "worldmap_hd.png"
    composite.save(src_png)
    run_esrgan(src_png, out_png, esrgan_model)

    hd_composite = load_and_validate(out_png, (grid_w * tile_w * HD_SCALE, grid_h * tile_h * HD_SCALE), "realesrgan (worldmap composite)")

    out_dir = config.WORLDMAP_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, p in existing.items():
        col = idx % num_hor - min_col
        row = idx // num_hor - min_row
        box = (col * tile_w * HD_SCALE, row * tile_h * HD_SCALE,
            (col + 1) * tile_w * HD_SCALE, (row + 1) * tile_h * HD_SCALE)
        hd_composite.crop(box).save(out_dir / p.name, "BMP")

    src_png.unlink(missing_ok=True)
    out_png.unlink(missing_ok=True)

    return out_dir


def cmd_hd_worldmap_zoomed(esrgan_model: str) -> Path | None:
    """Best-effort standalone upscale of Map_Zoomed.bmp (the single full-
    overworld overview image, WorldMap.mes's ZoomedName key) - no composite
    needed, it's already one image. Low priority: sub_565230 (wmap_ui.c)
    blits it through a *scaling* tig_video_buffer_blit into a small on-screen
    pane, so the extra source resolution here is close to imperceptible on
    screen - included anyway since it's cheap and the machinery already
    exists (same straight-ESRGAN pattern as cmd_hd_slides/cmd_hd_splash).
    The place names are then drawn again with a real font (maplabels.py) -
    ESRGAN garbles 6 px letters."""
    src = config.WORLDMAP_ROOT / f"{config.WORLDMAP_ZOOMED_BASENAME}.bmp"
    if not src.is_file():
        return None
    tmp_dir = config.WORK_DIR / "_worldmap_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src_png = tmp_dir / "zoomed.png"
    out_png = tmp_dir / "zoomed_hd.png"
    Image.open(src).convert("RGB").save(src_png)
    run_esrgan(src_png, out_png, esrgan_model)
    dest = config.WORLDMAP_OUTPUT_DIR / f"{config.WORLDMAP_ZOOMED_BASENAME}.bmp"
    dest.parent.mkdir(parents=True, exist_ok=True)
    import maplabels
    maplabels.reletter(Image.open(src), Image.open(out_png)).save(dest, "BMP")
    src_png.unlink(missing_ok=True)
    out_png.unlink(missing_ok=True)
    return dest


def cmd_hd_worldmap(force: bool = False, model: str | None = None) -> None:
    """Batch driver for the overworld: tiles + Map_Zoomed. Single base-game
    map only - no separate Vormantown WorldMap exists (that module has no
    overworld, confirmed via find), so unlike cmd_hd_townmap() this isn't a
    per-town loop."""
    esrgan_model = model or config.REALESRGAN_MODEL
    log_name = "hd_worldmap"

    if not force and load_done_set(log_name):
        print("WorldMap: already done (use --force to regenerate)")
        return

    try:
        tiles_dir = cmd_hd_worldmap_tiles(esrgan_model)
        print(f"WorldMap tiles -> {tiles_dir.relative_to(config.HD_OVERLAY_DIR)}")
        zoomed = cmd_hd_worldmap_zoomed(esrgan_model)
        if zoomed:
            print(f"WorldMap zoomed overview -> {zoomed.relative_to(config.HD_OVERLAY_DIR)}")
        with open(batch_log_dir() / f"{log_name}.done.txt", "a", encoding="utf-8") as f:
            f.write("WorldMap\n")
        print("WorldMap: done")
    except Exception as e:
        print(f"WorldMap: FAILED: {e}")
        with open(batch_log_dir() / f"{log_name}.failed.txt", "a", encoding="utf-8") as f:
            f.write(f"WorldMap\t{e}\n")


def cmd_hd_batch(category: str, model: str | None = None, limit: int | None = None, force: bool = False, file_list: Path | None = None, workers: int = 1) -> None:
    all_files = load_file_list(file_list) if file_list else find_category_files(category)
    if not all_files:
        source = str(file_list) if file_list else f"art/{category}/"
        raise RuntimeError(f"No .ART files found for {source}")
    log_name = f"hd_{category}"
    done = set() if force else load_done_set(log_name)
    pending = [f for f in all_files if f not in done]
    skipped_done = len(all_files) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    print(f"HD batch '{category}': {len(all_files)} total, {skipped_done} already done, {len(pending)} queued ({workers} worker{'s' if workers != 1 else ''}).", flush=True)

    ok = bad = 0
    completed = 0
    print_lock = threading.Lock()

    def run_one(rel_path: str) -> tuple[str, Exception | None]:
        last_bucket = -1

        def progress_cb(i: int, total: int) -> None:
            nonlocal last_bucket
            # Big animated sprites (monster/critter, up to 100+ frames) can
            # take minutes per file - a whole file finishing is too coarse a
            # unit to prove the batch is alive. ~10 buckets per file gives a
            # real moving bar without spamming tiny 1-4 frame icons (those
            # just print once, at completion, same as before).
            if total <= 8:
                return
            bucket = i * 10 // total
            if bucket == last_bucket and i != total:
                return
            last_bucket = bucket
            with print_lock:
                print(f"  {render_bar(i, total)} {i}/{total} frames - {rel_path}", flush=True)

        try:
            cmd_hd(rel_path, model=model, force=force, quiet=True, progress_cb=progress_cb)
            return rel_path, None
        except Exception as e:
            return rel_path, e

    if workers <= 1:
        for rel_path in pending:
            completed += 1
            _, err = run_one(rel_path)
            if err is None:
                append_done(log_name, rel_path)
                ok += 1
                print(f"[{completed}/{len(pending)}] {rel_path}", flush=True)
            else:
                print(f"[{completed}/{len(pending)}] FAILED: {rel_path}: {err}", flush=True)
                append_failed(log_name, rel_path, str(err))
                bad += 1
    else:
        # Each cmd_hd() spends nearly all its time inside a blocking
        # subprocess.run() (ncnn-vulkan), which releases the GIL while
        # waiting - real threads, not just async, genuinely overlap here.
        # ncnn-vulkan instances share the one GPU but run concurrently fine
        # at this frame size; work dirs are keyed by .ART basename
        # (work_dir_for), so two in-flight files never share a directory
        # unless two categories have an identically-named .ART (not the
        # case for interface/item today).
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, rel_path): rel_path for rel_path in pending}
            for future in as_completed(futures):
                rel_path, err = future.result()
                with print_lock:
                    completed += 1
                    if err is None:
                        append_done(log_name, rel_path)
                        ok += 1
                        print(f"[{completed}/{len(pending)}] {rel_path}", flush=True)
                    else:
                        append_failed(log_name, rel_path, str(err))
                        bad += 1
                        print(f"[{completed}/{len(pending)}] FAILED: {rel_path}: {err}", flush=True)

    print(f"HD batch '{category}' finished: {ok} succeeded, {bad} failed, {skipped_done} skipped. Logs: {batch_log_dir() / (log_name + '.done.txt')}", flush=True)


def find_category_files(category: str) -> list[str]:
    """Every .ART file under art/<category>/ across all EXTRACTED_DAT_ROOTS,
    deduped by rel_path (relative to its own root) in root-priority order -
    same resolution order find_source_art() itself uses, so a file present
    under multiple roots is only queued once.

    Special pseudo-category "tig-root": loose *.ART sitting directly under
    an art/ root (not inside any category subfolder) - e.g. tig.dat's
    mouse.ART (the cursor) and 15 UI chrome files (button.ART, TileStamp.ART,
    etc.). These are real, in-game-used ART files that every other category
    batch silently skips, since they walk art/<category>/ subfolders only.
    Excluded on purpose: BadArt.ART (arcanum1/art root, the missing-texture
    placeholder - not worth converting) and morph15font.ART (a bitmap font
    glyph sheet - fonts are explicitly deferred to the very end of the HD
    pipeline work, handled together with the TTF project, not bundled in
    here just because it happens to sit in the same loose-file root).
    """
    _TIG_ROOT_EXCLUDE = {"BadArt.ART", "morph15font.ART"}
    if category == "tig-root":
        seen: set[str] = set()
        rel_paths: list[str] = []
        for root in config.EXTRACTED_DAT_ROOTS:
            art_dir = root / "art"
            if not art_dir.is_dir():
                continue
            for p in sorted(art_dir.iterdir()):
                if p.is_file() and p.suffix.lower() == ".art" and p.name not in _TIG_ROOT_EXCLUDE:
                    rel = p.relative_to(root).as_posix()
                    if rel not in seen:
                        seen.add(rel)
                        rel_paths.append(rel)
        return rel_paths

    seen: set[str] = set()
    rel_paths: list[str] = []
    for root in config.EXTRACTED_DAT_ROOTS:
        cat_dir = root / "art" / category
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() == ".art":
                rel = p.relative_to(root).as_posix()
                if rel not in seen:
                    seen.add(rel)
                    rel_paths.append(rel)
    return rel_paths


def batch_log_dir() -> Path:
    d = config.WORK_DIR / "batch_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_done_set(category: str) -> set[str]:
    log = batch_log_dir() / f"{category}.done.txt"
    if not log.exists():
        return set()
    return set(log.read_text(encoding="utf-8").splitlines())


def append_done(category: str, rel_path: str) -> None:
    with open(batch_log_dir() / f"{category}.done.txt", "a", encoding="utf-8") as f:
        f.write(rel_path + "\n")


def append_failed(category: str, rel_path: str, error: str) -> None:
    with open(batch_log_dir() / f"{category}.failed.txt", "a", encoding="utf-8") as f:
        f.write(f"{rel_path}\t{error}\n")


def load_file_list(list_path: Path) -> list[str]:
    lines = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


# Alpha edge smoothing (playtest round 4: "rings and circular buttons are far
# from perfect circles"). hq4x straightens diagonals, but a curve in the
# 1-bit colour-key mask is a staircase of 1-px steps, which comes out as
# 4-px steps at HD. A small Gaussian blur at 4x (sigma well under one
# source pixel) averages the steps into a slope, then a smoothstep
# re-sharpens it to a ~2 px anti-aliased edge. Straight edges stay put;
# square corners round by ~1 HD px, invisible at display scale.
ALPHA_SMOOTH_SIGMA = 2.5
ALPHA_SMOOTH_EDGE = (0.3, 0.7)


def gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, edge-padded, pure numpy (no scipy here)."""
    radius = max(1, int(round(sigma * 3)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    padded = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for i, k in enumerate(kernel):
        out += k * padded[:, i:i + img.shape[1]]
    padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
    out2 = np.zeros_like(out)
    for i, k in enumerate(kernel):
        out2 += k * padded[i:i + img.shape[0], :]
    return out2


def smooth_alpha(alpha: np.ndarray) -> np.ndarray:
    """0..1 float alpha -> de-staircased 0..1 float alpha."""
    lo, hi = ALPHA_SMOOTH_EDGE
    t = np.clip((gaussian_blur_2d(alpha.astype(np.float32), ALPHA_SMOOTH_SIGMA) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smooth_alpha_file(png: Path) -> bool:
    backup = alpha_backup_path(png)
    source = backup if backup.exists() else png
    with Image.open(source) as im:
        rgba = np.asarray(im.convert("RGBA"))
    alpha = rgba[..., 3]
    if alpha.min() == 255 or alpha.max() == 0:
        return False
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(png, backup)
    out = rgba.copy()
    out[..., 3] = np.clip(smooth_alpha(alpha / 255.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(png, "PNG", compress_level=6)
    return True


def cmd_hd_smooth_alpha(category: str = "interface", workers: int = 8) -> None:
    """Apply smooth_alpha() to every existing hd/art/<category>/ sidecar
    that has real transparency. Colour is untouched (no re-upscale). The
    first run keeps the originals in work/_alpha_originals/ and every run
    starts from them, so it is safe to rerun or retune. New sidecars get
    this as they're written (write_sidecar, ALPHA_SMOOTH_CATEGORIES)."""
    from concurrent.futures import ProcessPoolExecutor
    root = config.HD_OVERLAY_DIR / "art" / category
    # hd-background-matte smooths its own pieces (it regenerates them from
    # their originals, so a smoothed copy here would go stale).
    matte_dirs = {hd_out_dir(piece) for piece, _, _ in BACKGROUND_MATTE_PIECES}
    pngs = [p for p in sorted(root.rglob("r*_f*.png")) if p.parent not in matte_dirs]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, smoothed in enumerate(pool.map(_smooth_alpha_file, pngs, chunksize=64), 1):
            done += smoothed
            if i % 5000 == 0:
                print(f"  {render_bar(i, len(pngs))} {i}/{len(pngs)}", flush=True)
    print(f"hd-smooth-alpha {category}: smoothed {done}, skipped {len(pngs) - done} (opaque/empty)", flush=True)


# --- hd-scan: find sidecars written from corrupt ncnn batch output ---------
#
# Everything generated before verify_batch() existed went through ncnn
# directory mode unchecked (see BATCH_OUTPUT_MIN_CORR). A frame is flagged
# when its sidecar, box-downscaled to 1x, correlates with the vanilla frame
# below that same threshold over the vanilla frame's opaque pixels (key
# pixels are inpainted before upscaling, so they're no reference). --fix
# redoes just those frames single-file, through the same route cmd_hd takes.

def _scan_art(rel: str) -> tuple[str, int, list[tuple[int, int, float]], str | None]:
    """(rel, frames checked, [(rot, frame, corr) below threshold], error)."""
    import tempfile
    out_dir = hd_out_dir(rel)
    bad: list[tuple[int, int, float]] = []
    checked = 0
    try:
        with tempfile.TemporaryDirectory(dir=config.WORK_DIR / "_scan_tmp") as tmp:
            basename = Path(rel).name.rsplit(".", 1)[0]
            run_art_converter(find_source_art(rel), Path(tmp) / basename)
            n, anim = read_ini_frame_count(Path(tmp) / (basename + ".ini"))
            for rot, frame, bmp in hd_frame_bmps(Path(tmp), basename, n, anim):
                png = out_dir / f"r{rot}_f{frame}.png"
                if not png.is_file():
                    continue
                w, h = read_bmp_dims(bmp)
                h = abs(h)
                idx = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
                opaque = idx != 0
                if not opaque.any():
                    continue
                checked += 1
                pal = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
                with Image.open(png) as im:
                    hd = im.convert("RGB")
                if hd.size != (w * HD_SCALE, h * HD_SCALE):
                    bad.append((rot, frame, float("inf")))
                    continue
                corr = structural_corr(pal[idx], np.asarray(hd.resize((w, h), Image.BOX)), opaque)
                if corr < BATCH_OUTPUT_MIN_CORR:
                    bad.append((rot, frame, corr))
    except Exception as e:  # report and keep scanning
        return rel, checked, bad, str(e)
    return rel, checked, bad, None


def _hd_art_rel_paths(category: str) -> list[str]:
    """Source rel_paths of the .ARTs that have a sidecar dir under hd/art/<category>/."""
    return [rel for rel in find_category_files(category) if hd_out_dir(rel).is_dir()]


def cmd_hd_scan(categories: list[str], workers: int = 8, fix: bool = False, model: str | None = None) -> None:
    (config.WORK_DIR / "_scan_tmp").mkdir(parents=True, exist_ok=True)
    report = batch_log_dir() / "hd_scan.txt"
    for category in categories:
        rels = _hd_art_rel_paths(category)
        print(f"hd-scan {category}: {len(rels)} art(s), {workers} worker(s)...", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        checked = 0
        bad_by_art: dict[str, list[tuple[int, int, float]]] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool, open(report, "a", encoding="utf-8") as log:
            for i, (rel, n, bad, err) in enumerate(pool.map(_scan_art, rels, chunksize=4), 1):
                checked += n
                if err:
                    log.write(f"ERROR|{rel}|{err.splitlines()[0] if err else ''}\n")
                for rot, frame, corr in bad:
                    log.write(f"BAD|{rel}|r{rot}_f{frame}|corr {corr:.2f}\n")
                if bad:
                    bad_by_art[rel] = bad
                if i % 200 == 0 or i == len(rels):
                    nbad = sum(len(b) for b in bad_by_art.values())
                    print(f"  {render_bar(i, len(rels))} {i}/{len(rels)} arts, {checked} frames, {nbad} bad", flush=True)
        nbad = sum(len(b) for b in bad_by_art.values())
        print(f"hd-scan {category}: {nbad} bad frame(s) in {len(bad_by_art)} art(s) (list in {report})", flush=True)
        if fix and bad_by_art:
            n = regenerate_frames([(rel, [(r, f) for r, f, _ in bad]) for rel, bad in bad_by_art.items()], model=model)
            print(f"hd-scan {category}: redid {n} frame(s)", flush=True)


REGEN_CHUNK_FRAMES = 1500


def regenerate_frames(jobs: list[tuple[str, list[tuple[int, int]] | None]], model: str | None = None) -> int:
    """Regenerate frames exactly as cmd_hd now would (ESRGAN after
    dedither(), FORCE_CUGAN_ASSETS through Real-CUGAN, soup fallback,
    hq4x alpha, write_sidecar's AA pass), for [(rel, [(rot, frame)] or None
    for all)], many arts per verified directory-mode launch. Returns the
    number of frames written."""
    esrgan_model = model or config.REALESRGAN_MODEL
    root = config.WORK_DIR / "_regen"
    written = 0

    def flush(staged: list[dict]) -> int:
        if not staged:
            return 0
        e_in, c_in = root / "e_in", root / "c_in"
        if any(s["route"] == "esrgan" for s in staged):
            run_esrgan_batch(e_in, root / "e_out", esrgan_model)
        if any(s["route"] == "cugan" for s in staged):
            run_realcugan_batch(c_in, root / "c_out")
        for s in staged:
            name = s["name"]
            if s["route"] == "cugan":
                out = root / "c_out" / name
            else:
                out = root / "e_out" / name
                if roughness(Image.open(out)) > PIXEL_SOUP_ROUGHNESS_THRESHOLD:
                    out = root / "soup" / name
                    out.parent.mkdir(exist_ok=True)
                    run_realcugan(e_in / name, out)
            hd = load_and_validate(out, (s["w"] * HD_SCALE, s["h"] * HD_SCALE), "regenerate")
            mask_rgb = Image.fromarray(np.where(s["key"], 0, 255).astype(np.uint8), "L").convert("RGB")
            hd.putalpha(hqx.hq4x(mask_rgb).convert("L"))
            write_sidecar(s["rel"], s["dest"], hd)
        return len(staged)

    def reset() -> None:
        if root.exists():
            shutil.rmtree(root)
        for d in ("e_in", "c_in", "art"):
            (root / d).mkdir(parents=True)

    reset()
    staged: list[dict] = []
    for ai, (rel, frames) in enumerate(jobs):
        try:
            art_dir = root / "art" / f"{ai:06d}"
            art_dir.mkdir()
            basename = Path(rel.replace("\\", "/")).name.rsplit(".", 1)[0]
            run_art_converter(find_source_art(rel), art_dir / basename)
            n, anim = read_ini_frame_count(art_dir / (basename + ".ini"))
            by_key = {(r, f): p for r, f, p in hd_frame_bmps(art_dir, basename, n, anim)}
        except Exception as e:
            print(f"  regenerate: skipped {rel}: {e}", flush=True)
            continue
        force_cugan = rel.replace("\\", "/") in config.FORCE_CUGAN_ASSETS
        out_dir = hd_out_dir(rel)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rot, frame in frames if frames is not None else list(by_key):
            bmp = by_key[(rot, frame)]
            w, h = read_bmp_dims(bmp)
            h = abs(h)
            idx = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
            pal = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
            key = idx == 0
            src = np.clip(inpaint_colorkey(pal[idx], key), 0, 255).astype(np.uint8)
            route = "cugan" if force_cugan else "esrgan"
            if route == "esrgan":
                src = dedither(src, key)
            name = f"{len(staged):06d}.png"
            Image.fromarray(src, "RGB").save(root / ("c_in" if route == "cugan" else "e_in") / name)
            staged.append(dict(rel=rel, dest=out_dir / f"r{rot}_f{frame}.png", w=w, h=h, key=key, route=route, name=name))
        if len(staged) >= REGEN_CHUNK_FRAMES:
            written += flush(staged)
            print(f"  regenerate: {written} frame(s) written, {ai + 1}/{len(jobs)} art(s)", flush=True)
            staged = []
            reset()
    written += flush(staged)
    shutil.rmtree(root, ignore_errors=True)
    return written


def redo_hd_frames(rel: str, frames: list[tuple[int, int]] | None = None, model: str | None = None) -> None:
    """Regenerate some (default: all) frames of one .ART - see regenerate_frames."""
    n = regenerate_frames([(rel, frames)], model=model)
    print(f"  redid {n} frame(s) of {rel}", flush=True)


# --- hd-palettes: sidecars for an art's extra palettes ---------------------
#
# Many item arts (robes, chain/leather/plate armour, some swords) carry 2-4
# palettes; the art id selects one (tig_art_id_palette_get). The BMPs the
# converter writes always use palette 0, so those items had no HD sidecar
# for their other colours and the engine refused them (tig_art_hd_blit:
# palette != 0 -> 1x; playtest round 5, robe icon). The .ini lists every
# palette as 256 "RRGGBB00"-ish tokens, each byte's two hex digits written
# low nibble first (checked against the BMP palette: 768/768 channels).
# Sidecars go to r<rot>_f<frame>_p<N>.png next to palette 0's.

def read_ini_palettes(ini_path: Path) -> list[np.ndarray]:
    text = ini_path.read_text(errors="replace")
    parts = re.split(r"^palette (\d+):\s*$", text, flags=re.M)
    out = []
    for i in range(1, len(parts) - 1, 2):
        toks = parts[i + 1].split()[:256]
        if len(toks) < 256:
            break
        out.append(np.array([[int(t[j:j + 2][::-1], 16) for j in (0, 2, 4)] for t in toks], dtype=np.uint8))
    return out


def cmd_hd_palettes(categories: list[str], model: str | None = None, force: bool = False) -> None:
    esrgan_model = model or config.REALESRGAN_MODEL
    root = config.WORK_DIR / "_palettes"
    for category in categories:
        if root.exists():
            shutil.rmtree(root)
        (root / "in").mkdir(parents=True)
        staged = []
        for rel in find_category_files(category):
            out_dir = hd_out_dir(rel)
            if not out_dir.is_dir():
                continue
            try:
                wd = cmd_unpack(rel, quiet=True)
                basename = Path(rel).name.rsplit(".", 1)[0]
                palettes = read_ini_palettes(wd / (basename + ".ini"))
                if len(palettes) < 2:
                    continue
                n, anim = read_ini_frame_count(wd / (basename + ".ini"))
                frames = hd_frame_bmps(wd, basename, n, anim)
            except Exception as e:
                print(f"  hd-palettes: skipped {rel}: {e}", flush=True)
                continue
            for p in range(1, len(palettes)):
                for rot, frame, bmp in frames:
                    dest = out_dir / f"r{rot}_f{frame}_p{p}.png"
                    if dest.exists() and not force:
                        continue
                    w, h = read_bmp_dims(bmp)
                    h = abs(h)
                    idx = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
                    key = idx == 0
                    if key.all():
                        continue
                    src = np.clip(inpaint_colorkey(palettes[p][idx], key), 0, 255).astype(np.uint8)
                    name = f"{len(staged):06d}.png"
                    Image.fromarray(dedither(src, key), "RGB").save(root / "in" / name)
                    staged.append(dict(rel=rel, dest=dest, w=w, h=h, key=key, name=name))
        print(f"hd-palettes {category}: {len(staged)} palette frame(s) to upscale", flush=True)
        if not staged:
            continue
        run_esrgan_batch(root / "in", root / "out", esrgan_model)
        for s in staged:
            out = root / "out" / s["name"]
            if roughness(Image.open(out)) > PIXEL_SOUP_ROUGHNESS_THRESHOLD:
                out = root / "soup" / s["name"]
                out.parent.mkdir(exist_ok=True)
                run_realcugan(root / "in" / s["name"], out)
            hd = load_and_validate(out, (s["w"] * HD_SCALE, s["h"] * HD_SCALE), "hd-palettes")
            mask_rgb = Image.fromarray(np.where(s["key"], 0, 255).astype(np.uint8), "L").convert("RGB")
            hd.putalpha(hqx.hq4x(mask_rgb).convert("L"))
            write_sidecar(s["rel"], s["dest"], hd)
        print(f"hd-palettes {category}: wrote {len(staged)} sidecar(s)", flush=True)
    shutil.rmtree(root, ignore_errors=True)


# --- hd-scroll-thumb: the scrollbar thumb, upscaled as one stack ----------
#
# scrollbar_ui.c draws the thumb as ScrllSlideT (11x5), then ScrllSlideM1
# (11x1) once per pixel row, then ScrllSlideB (11x7), edge to edge. Upscaled
# one by one, every 1-px row became its own 4-px ESRGAN strip, so the thumb
# showed a seam per row (playtest round 2 item 8). Here the stack T + M1 x N
# + B is upscaled as one image and cut back apart; the M1 sidecar is a row
# from the middle (M1 above and below it, as when drawn), and the AA pass is
# applied to the whole stack before slicing so no joint gets a softened edge.
SCROLL_THUMB_PIECES = ("art/interface/ScrllSlideT.ART", "art/interface/ScrllSlideM1.ART", "art/interface/ScrllSlideB.ART")
SCROLL_THUMB_REPEAT = 24


def _frame_rgb_key(rel: str) -> tuple[np.ndarray, np.ndarray]:
    wd = cmd_unpack(rel, quiet=True)
    basename = Path(rel).name.rsplit(".", 1)[0]
    n, anim = read_ini_frame_count(wd / (basename + ".ini"))
    _, _, bmp = hd_frame_bmps(wd, basename, n, anim)[0]
    w, h = read_bmp_dims(bmp)
    h = abs(h)
    idx = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
    pal = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
    return pal[idx], idx == 0


def cmd_hd_scroll_thumb(model: str | None = None) -> None:
    rels = [next(r for r in find_category_files("interface") if r.lower() == p.lower()) for p in SCROLL_THUMB_PIECES]
    (top, top_key), (mid, mid_key), (bot, bot_key) = (_frame_rgb_key(r) for r in rels)
    rgb = np.concatenate([top] + [mid] * SCROLL_THUMB_REPEAT + [bot])
    key = np.concatenate([top_key] + [mid_key] * SCROLL_THUMB_REPEAT + [bot_key])
    src = dedither(np.clip(inpaint_colorkey(rgb, key), 0, 255).astype(np.uint8), key)
    stage = config.WORK_DIR / "_scroll_thumb"
    stage.mkdir(parents=True, exist_ok=True)
    Image.fromarray(src, "RGB").save(stage / "in.png")
    run_esrgan(stage / "in.png", stage / "out.png", model or config.REALESRGAN_MODEL)
    hd = np.asarray(load_and_validate(stage / "out.png", (src.shape[1] * HD_SCALE, src.shape[0] * HD_SCALE), "hd-scroll-thumb"))
    mask_rgb = Image.fromarray(np.where(key, 0, 255).astype(np.uint8), "L").convert("RGB")
    alpha = np.asarray(hqx.hq4x(mask_rgb).convert("L"), dtype=np.float32) / 255.0
    if alpha.min() < 1.0:
        alpha = smooth_alpha(alpha)
    rgba = np.dstack([hd, np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)])
    s = HD_SCALE
    t_h, m_h = top.shape[0] * s, mid.shape[0] * s
    mid_row = t_h + (SCROLL_THUMB_REPEAT // 2) * m_h
    slices = [rgba[:t_h], rgba[mid_row:mid_row + m_h], rgba[-bot.shape[0] * s:]]
    for rel, piece in zip(rels, slices):
        dest = hd_out_dir(rel) / "r0_f0.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(piece), "RGBA").save(dest, "PNG", compress_level=6)
        backup = alpha_backup_path(dest)
        if backup.exists():
            backup.unlink()
        print(f"hd-scroll-thumb: wrote {dest.relative_to(config.HD_OVERLAY_DIR)} ({piece.shape[1]}x{piece.shape[0]})")


# --- hd-requeue: frames whose route changed under the one-model policy ----

def _requeue_art(rel: str) -> tuple[str, list[tuple[int, int]], str | None]:
    """Frames of one .ART that cmd_hd would now upscale differently: those
    the old size rule sent to Real-CUGAN, and those dedither() changes."""
    import tempfile
    out_dir = hd_out_dir(rel)
    todo: list[tuple[int, int]] = []
    if rel.replace("\\", "/") in config.FORCE_CUGAN_ASSETS:
        return rel, todo, None
    # Built as one stack by hd-scroll-thumb; a standalone redo would bring
    # the seams back.
    if rel.replace("\\", "/").lower() in {p.lower() for p in SCROLL_THUMB_PIECES}:
        return rel, todo, None
    try:
        with tempfile.TemporaryDirectory(dir=config.WORK_DIR / "_scan_tmp") as tmp:
            basename = Path(rel).name.rsplit(".", 1)[0]
            run_art_converter(find_source_art(rel), Path(tmp) / basename)
            n, anim = read_ini_frame_count(Path(tmp) / (basename + ".ini"))
            for rot, frame, bmp in hd_frame_bmps(Path(tmp), basename, n, anim):
                if not (out_dir / f"r{rot}_f{frame}.png").is_file():
                    continue
                w, h = read_bmp_dims(bmp)
                h = abs(h)
                idx = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
                key = idx == 0
                if key.all():
                    continue
                if w * h < HD_SMALL_FRAME_PX * HD_SMALL_FRAME_PX:
                    todo.append((rot, frame))
                    continue
                pal = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
                if dither_weight(pal[idx], key).max() > 0.0:
                    todo.append((rot, frame))
    except Exception as e:
        return rel, todo, str(e)
    return rel, todo, None


def cmd_hd_requeue(categories: list[str], workers: int = 8, dry_run: bool = False, model: str | None = None) -> None:
    from concurrent.futures import ProcessPoolExecutor
    (config.WORK_DIR / "_scan_tmp").mkdir(parents=True, exist_ok=True)
    for category in categories:
        rels = _hd_art_rel_paths(category)
        jobs: list[tuple[str, list[tuple[int, int]]]] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rel, todo, err in pool.map(_requeue_art, rels, chunksize=4):
                if err:
                    print(f"  requeue: {rel}: {err.splitlines()[0]}", flush=True)
                if todo:
                    jobs.append((rel, todo))
        nframes = sum(len(t) for _, t in jobs)
        print(f"hd-requeue {category}: {nframes} frame(s) in {len(jobs)} art(s) take the new route", flush=True)
        with open(batch_log_dir() / f"hd_requeue_{category}.txt", "w", encoding="utf-8") as f:
            for rel, todo in jobs:
                f.write(f"{rel}|{' '.join(f'r{r}_f{fr}' for r, fr in todo)}\n")
        if not dry_run and jobs:
            n = regenerate_frames(jobs, model=model)
            print(f"hd-requeue {category}: regenerated {n} frame(s)", flush=True)


# Interface pieces whose vanilla art has a copy of the background they sit
# on baked into their opaque corners (the PC-lens rings: square art, round
# ring, wood outside it). Upscaled on their own, those corners come out as a
# blurrier, differently-coloured square that no longer matches the (also
# upscaled) background around them - playtest round 4 #62. (piece, the
# background it is drawn on, top-left of the piece in that background)
BACKGROUND_MATTE_PIECES = [
    ("art/interface/SaveLoadPCLens.ART", "art/interface/SaveLoadBackground.ART", (84, 10)),
    ("art/interface/OptionsPCLens.ART", "art/interface/OptionsMenuBack.ART", (84, 67)),
]

# Max per-channel difference for "this piece pixel is background": the
# piece re-quantized the same wood to its own palette (measured <= 6).
BACKGROUND_MATTE_TOLERANCE = 12
BACKGROUND_MATTE_EDGE_TOLERANCE = 48


def vanilla_frame_rgb(rel_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(rgb HxWx3, palette indices HxW) of frame 0 of a vanilla .ART."""
    wd = cmd_unpack(rel_path, quiet=True)
    bmp = frame_bmps(wd)[0]
    w, h = read_bmp_dims(bmp)
    h = abs(h)
    palette = np.array(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
    indices = np.array(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
    return palette[indices], indices


def cmd_hd_background_matte() -> None:
    """Combine -> upscale -> decompose, for the background part only: the
    region of each BACKGROUND_MATTE_PIECES piece that is really background
    (flood-filled from the piece's border across pixels matching the vanilla
    background within tolerance, so it stops at the ring) is replaced in the
    piece's HD sidecar by the matching crop of the background's HD sidecar.
    The HD background is one upscale of the whole screen, so the corners
    then line up with it exactly. Overwrites hd/art/<piece>/r0_f0.png; the
    first run keeps the original upscale in work/_matte_originals/ and every
    run starts from that, so it is safe to rerun."""
    for piece_rel, bg_rel, (ox, oy) in BACKGROUND_MATTE_PIECES:
        piece_rgb, piece_idx = vanilla_frame_rgb(piece_rel)
        bg_rgb, _ = vanilla_frame_rgb(bg_rel)
        ph, pw = piece_idx.shape

        diff = np.abs(bg_rgb[oy:oy + ph, ox:ox + pw].astype(int) - piece_rgb.astype(int)).max(axis=2)
        candidate = (diff <= BACKGROUND_MATTE_TOLERANCE) & (piece_idx != 0)

        # Flood fill from the border through candidate pixels (4-neighbour).
        region = np.zeros_like(candidate)
        region[0, :] = candidate[0, :]
        region[-1, :] = candidate[-1, :]
        region[:, 0] |= candidate[:, 0]
        region[:, -1] |= candidate[:, -1]
        while True:
            grown = region.copy()
            grown[1:, :] |= region[:-1, :]
            grown[:-1, :] |= region[1:, :]
            grown[:, 1:] |= region[:, :-1]
            grown[:, :-1] |= region[:, 1:]
            grown &= candidate
            if np.array_equal(grown, region):
                break
            region = grown

        # One more vanilla pixel into the anti-aliased band where the ring
        # blends into the wood (looser tolerance, so the gold ring itself
        # stays) - otherwise a thin stepped strip of the old upscale remains.
        edge = region.copy()
        edge[1:, :] |= region[:-1, :]
        edge[:-1, :] |= region[1:, :]
        edge[:, 1:] |= region[:, :-1]
        edge[:, :-1] |= region[:, 1:]
        region |= edge & (diff <= BACKGROUND_MATTE_EDGE_TOLERANCE) & (piece_idx != 0)

        piece_png = hd_out_dir(piece_rel) / "r0_f0.png"
        bg_png = hd_out_dir(bg_rel) / "r0_f0.png"
        # Outside the per-art work dir: cmd_unpack() wipes that every run.
        backup = config.WORK_DIR / "_matte_originals" / f"{work_dir_for(piece_rel).name}.png"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copyfile(piece_png, backup)

        hd_piece = np.asarray(Image.open(backup).convert("RGBA"), dtype=np.float32)
        hd_bg = np.asarray(Image.open(bg_png).convert("RGBA"), dtype=np.float32)
        scale = hd_piece.shape[1] // pw
        if hd_piece.shape[0] != ph * scale or hd_bg.shape[1] != bg_rgb.shape[1] * scale:
            raise RuntimeError(f"{piece_rel}: unexpected HD sizes {hd_piece.shape} / {hd_bg.shape}")

        crop = hd_bg[oy * scale:(oy + ph) * scale, ox * scale:(ox + pw) * scale]

        # 4x nearest mask, de-staircased like the alpha (smooth_alpha), so
        # the seam around the ring is a smooth curve rather than 4-px steps
        # and isn't a hard cut between the two upscales.
        weight = smooth_alpha(np.kron(region.astype(np.float32), np.ones((scale, scale), dtype=np.float32)))
        weight = weight[..., None]

        out = hd_piece * (1.0 - weight) + crop * weight
        piece_alpha = smooth_alpha(hd_piece[..., 3] / 255.0) * 255.0
        out[..., 3] = np.maximum(piece_alpha, weight[..., 0] * 255.0)
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA").save(piece_png)
        print(f"{piece_rel}: {int(region.sum())} of {ph * pw} px taken from {bg_rel} at ({ox},{oy}) -> {piece_png}")


# --- hd-compose: compose -> upscale -> decompose from engine captures -------
#
# The engine (arcanum-ce, window.c tig_window_hd_capture) writes
# <game>/hd_capture/manifest.txt plus one BMP per placement when a
# hd_capture/ folder exists: the window's 1x pixels under each interface
# frame (16 px margin) the first time it's blitted at a position. Every run
# first merges that folder into work/_captures/ (so the game folder can be
# wiped between sessions without losing anything), then each captured frame
# is composed over its underlay and the composite is upscaled in context
# (ESRGAN sees the ring against what it's drawn on, instead of a lone sprite
# against an inpainted edge colour). The frame is cut back out of the
# upscale with an alpha that is itself an ESRGAN upscale of the frame's mask
# (smooth curves, unlike hq4x's 45-degree-only smoothing).
#
# Frames up to COMPOSE_DOUBLE_MAX px go through the upscaler twice (16x) and
# are Lanczos-downsampled back to 4x: at 4x the source's 1 px stair steps are
# 4 px features the second pass turns into real curves, so rings and round
# buttons come out round (playtest round 5).
#
# No background matte here: making frame pixels that equal their capture's
# underlay transparent broke every button whose states are drawn over each
# other (stuck hover frames, invisible click frames). Pieces that need it
# are handled by hd-background-matte (BACKGROUND_MATTE_PIECES) and skipped.

HD_CAPTURE_DIR = config.ARCANUM_ROOT / "hd_capture"
CAPTURE_ARCHIVE_DIR = config.WORK_DIR / "_captures"
# 0 = off. The x16 -> x4 double pass (was 160) lost to plain x4 in the
# user's 49-sample comparison (comparison/x16_trial/, 2026-09-26).
COMPOSE_DOUBLE_MAX = 0
COMPOSE_MASK_EDGE = (0.3, 0.7)
BLT_FLIP_X = 0x1
BLT_FLIP_Y = 0x2
WINDOW_TRANSPARENT = 0x1


def sync_capture_archive() -> Path:
    """Merge <game>/hd_capture/ into work/_captures/; returns the archive's
    manifest path."""
    CAPTURE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_manifest = CAPTURE_ARCHIVE_DIR / "manifest.txt"
    lines = archive_manifest.read_text(errors="replace").splitlines() if archive_manifest.is_file() else []
    known = set(lines)
    added = copied = 0
    live_manifest = HD_CAPTURE_DIR / "manifest.txt"
    if live_manifest.is_file():
        for bmp in HD_CAPTURE_DIR.glob("*.bmp"):
            dest = CAPTURE_ARCHIVE_DIR / bmp.name
            if not dest.exists() or dest.stat().st_size != bmp.stat().st_size:
                shutil.copyfile(bmp, dest)
                copied += 1
        for line in live_manifest.read_text(errors="replace").splitlines():
            if line and line not in known:
                known.add(line)
                lines.append(line)
                added += 1
        archive_manifest.write_text("".join(line + "\n" for line in lines))
    print(f"captures: {len(lines)} placement line(s) archived in {CAPTURE_ARCHIVE_DIR} (+{added} lines, {copied} BMPs from {HD_CAPTURE_DIR})")
    return archive_manifest


def read_capture_manifest() -> dict[tuple[str, int, int], list[dict]]:
    """{(art rel_path, rot, frame): [placement, ...]} in capture order, from
    the merged archive (see sync_capture_archive)."""
    manifest = sync_capture_archive()
    if not manifest.is_file():
        raise RuntimeError(f"No captures yet - create {HD_CAPTURE_DIR} and play the screens first")

    def rect(s: str) -> tuple[int, int, int, int]:
        x, y, w, h = (int(v) for v in s.split(","))
        return x, y, w, h

    out: dict[tuple[str, int, int], list[dict]] = {}
    for line in manifest.read_text(errors="replace").splitlines():
        parts = line.strip().split("|")
        if len(parts) != 11:
            continue
        art, rot, frame, flags, _win, src, dst, cap, bmp, win_flags, win_key = parts
        rel = art.replace("\\", "/")
        key = int(win_key, 16) if int(win_flags, 16) & WINDOW_TRANSPARENT else None
        out.setdefault((rel, int(rot), int(frame)), []).append(dict(
            flags=int(flags, 16), src=rect(src), dst=rect(dst), cap=rect(cap), bmp=CAPTURE_ARCHIVE_DIR / bmp,
            key=None if key is None else ((key >> 16) & 255, (key >> 8) & 255, key & 255),
        ))
    return out


# Per-art ESRGAN model exceptions to config.REALESRGAN_MODEL (user picks from
# comparison/button_models/, round 8): thin details the default smears.
BUTTON_MODEL = {
    "Char_Plus": "remacri-4x",
    "Char_Minus": "remacri-4x",
}
_COLLEGES = ("Air Conveyance Divination Earth EvilNecro Fire Force GoodNecro Mental Meta "
             "Morph Nature Phantasm Summoning Temporal Water").split()
# Round 8, second remacri pass (user picked from comparison/remacri_audit/ S,
# F, D sheets): college circles + spell-level squares, the four big HUD
# buttons (were x4plus), round HUD buttons, discipline buttons/tabs,
# logbook tabs.
REMACRI_UI = (
    ["S_Air", "S_Conveyance", "S_Divination", "S_Earth", "S_EvilNecro", "S_Fire", "S_Forc",
     "S_GoodNecro", "S_Mental", "S_Meta", "s_morph", "S_Nature", "s_phantasm", "S_Summoning",
     "S_Temporal", "S_Water"]
    + [f"S_{c}{n}" for c in _COLLEGES for n in range(1, 6)]
    + ("Char_But Char_ON Invn_But Invn_ON Log_But Log_ON TMap_But TMap_ON WMap_But WMap_ON "
       "Combat_But Combat_Button Anatomical_But Chemistry_But Electrical_But Explosives_But "
       "GunSmithy_But Mechanical_But Smithy_But Therapeutics_But Technological_But Social_But "
       "Thieving_But Anatomical_Tab Chemistry_Tab Electrical_Tab Explosives_Tab GunSmithy_Tab "
       "Mechanical_Tab Smithy_Tab Theraputics_Tab Tab_BackGrnd Tab_BlessCurse Tab_EgoInjure "
       "Tab_Keys Tab_Note Tab_Quest Tab_Rep").split()
)
BUTTON_MODEL.update({name: "remacri-4x" for name in REMACRI_UI})
# Back to x4plus after the in-game check (remacri stripes their hatched
# fills): charedit skill category buttons (key / gear / swap) and the four
# big Skills/Spells/Schematics/common-skills buttons (round 8, twice).
BUTTON_MODEL.update({name: "realesrgan-x4plus" for name in
                     ("Thieving_But", "Technological_But", "Social_But",
                      "Skills_Button", "Spells_Button", "Schematics_Button", "char_Common_Skills",
                      # pass 4: the rest of the charedit row, to match the HUD ones
                      "char_Tech_Skills", "char_Spells_Skills", "char_Schem_Skills")})
# Round 8 remacri pass (user: remacri wins on small assets - checked in game,
# then fixed one by one): A = small arrow / +- controls, B = small symbol
# icons, C = cursors. See docs/ROUND8_PLAN.md.
REMACRI_SMALL = (
    # A
    "SP_Plus SP_Minus Char_HTFTPlus Char_HTFTMinus OldChar_Plus OldChar_Minus "
    "SkilAddBut SkilMinusBut SpellTech_Add SpellTech_Minus Big_Grn_L Big_Grn_R "
    "PageTurn_L PageTurn_R Schm_LArrow Schm_RArrow WrtnBookLArro WrtnBookRArro "
    "SldrButt_L_Arrow SldrButt_R_Arrow Sm_RightArrow Scrll_DWN Scrll_UP ScrllDWN "
    "ScrllUP Follow_Scroll_dwn Follow_Scroll_dwn_OFF Follow_Scroll_up "
    "Follow_Scroll_up_OFF FollowerCycleLeft FollowerCycleRight MPCycleLeftButton "
    "MPCycleRightButton M_UpBut M_DnBut MultiPlay_UP MultiPlay_DWN Cursor_UP "
    "Cursor_DWN BookmarkButt UnBookmarkButt PrivMes_ClsBut EndTurn_But "
    "MPly_AddBut MPly_KickBut MPRefreshButton FateBut "
    # B
    "MM_Chest MM_Cross MM_Loc MM_LocNew MM_Note MM_Ques MM_Skull MM_WayP "
    "MMB_Chest MMB_Note MMB_Ques MMB_Skull AP_Green AP_Orange AP_Red "
    "comm_sk_small tech_sk_small SKL_Combine SKL_Conceal SKL_Heal SKL_PickLock "
    "SKL_pickpocket SKL_Repair SKL_silentmove SKL_Traps Pen_Cover Pen_Injury "
    "Pen_Light Pen_MSR Pen_Perception Pen_Range Ammo_Icon_Arrows "
    "Ammo_Icon_Bullets Ammo_Icon_Charges Ammo_Icon_Fuel Ammo_Icon_Gold "
    "Ammo_Icon_Mana BlockedShot Magic-Tech-Penalty Item_Dam XP_Pip1 XP_Pip2 "
    "XP_Pip3 XP_Pip4 XP_Pip5 XP_Pip6 XP_Pip7 XP_Pip8 XP_Pip9 XP_Pip10 HKTshON "
    "HKTshOFF MT_Apt "
    # C
    "cursor battlecur skillcur spellcur TechCur Cursor-Called-Arm "
    "Cursor-Called-Head Cursor-Called-Leg CURSOR-Identify-Item cur_del Scroll-0 "
    "Scroll-1 Scroll-2 Scroll-3 Scroll-4 Scroll-5 Scroll-6 Scroll-7 Scroll_not"
).split()
BUTTON_MODEL.update({name: "remacri-4x" for name in REMACRI_SMALL})
# Round 8 pass 4: the dialog text toggle (nav bar) read blurry in x4plus.
BUTTON_MODEL["TextToggle"] = "remacri-4x"


BUTTON_MIN_SIDE = 64


def cmd_hd_buttons(names: list[str], model: str | None = None) -> None:
    """Re-upscale interface buttons one frame at a time with checker_average()
    instead of dedither() (dithered hover/press glows), without hd-compose's
    in-context step. Each frame is its own verified ESRGAN run (no directory
    batch). Previous sidecars go to work/_button_originals/ once."""
    esrgan_model = model or config.REALESRGAN_MODEL
    stage = config.WORK_DIR / "_buttons"
    stage.mkdir(parents=True, exist_ok=True)
    for name in names:
        rel = name if "/" in name else f"art/interface/{name}.ART"
        wd = cmd_unpack(rel, quiet=True)
        basename = Path(rel).name.rsplit(".", 1)[0]
        esrgan_model = model or BUTTON_MODEL.get(basename, config.REALESRGAN_MODEL)
        num_frames, animated = read_ini_frame_count(wd / (basename + ".ini"))
        out_dir = hd_out_dir(rel)
        backup = config.WORK_DIR / "_button_originals" / out_dir.relative_to(config.HD_OVERLAY_DIR)
        if out_dir.is_dir() and not backup.exists():
            shutil.copytree(out_dir, backup)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rot, frame, bmp in hd_frame_bmps(wd, basename, num_frames, animated):
            w, h = read_bmp_dims(bmp)
            h = abs(h)
            indices = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
            palette = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
            key = indices == 0
            src = np.clip(inpaint_colorkey(palette[indices], key), 0, 255).astype(np.uint8)
            src = checker_average(src, key)
            src_png = stage / f"{basename}_{rot}_{frame}.png"
            hd_png = stage / f"{basename}_{rot}_{frame}_x4.png"
            # ncnn returns noise for tiny inputs (remacri on the 38x10
            # XP pips, 9x9 map markers): edge-pad to BUTTON_MIN_SIDE, crop after.
            pw, ph = max(0, BUTTON_MIN_SIDE - w), max(0, BUTTON_MIN_SIDE - h)
            padded = np.pad(src, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)), mode="edge")
            Image.fromarray(padded, "RGB").save(src_png)
            run_esrgan(src_png, hd_png, esrgan_model)
            hd = load_and_validate(hd_png, (padded.shape[1] * HD_SCALE, padded.shape[0] * HD_SCALE), "hd-buttons")
            x0, y0 = pw // 2 * HD_SCALE, ph // 2 * HD_SCALE
            hd = hd.crop((x0, y0, x0 + w * HD_SCALE, y0 + h * HD_SCALE))
            flat = src.reshape(-1, 3)
            if not np.all(flat == flat[0]) and is_blank_output(hd):
                raise RuntimeError(f"realesrgan produced blank output for {bmp}")
            corr = structural_corr(src, np.asarray(hd.convert("RGB").resize((w, h), Image.BOX)), ~key)
            if corr < BATCH_OUTPUT_MIN_CORR:
                raise RuntimeError(f"realesrgan output doesn't match {bmp} (corr {corr:.2f})")
            mask_rgb = Image.fromarray(np.where(key, 0, 255).astype(np.uint8), "L").convert("RGB")
            rgba = hd.convert("RGB")
            rgba.putalpha(hqx.hq4x(mask_rgb).convert("L"))
            dest = out_dir / f"r{rot}_f{frame}.png"
            write_sidecar(rel, dest, rgba)
            print(f"  {bmp.name} -> {dest.relative_to(config.HD_OVERLAY_DIR)}")


# The UI text face (Flare12/14, pork12, Garmond9, Nick16, Euph30 - Grenze
# until round 8 pass 6). Change it here: (file in fonts/vanilla/, variable-
# font weight or None, download URL used when the file is missing).
MAIN_FONT = ("Outfit[wght].ttf", 400,
             "https://github.com/google/fonts/raw/main/ofl/outfit/Outfit%5Bwght%5D.ttf")
MAIN_FONT_ARTS = [
    "art/interface/Euph30Font.ART",
    "art/interface/Flare12Font.ART",
    "art/interface/Flare14Font.ART",
    "art/interface/Garmond9Font.ART",
    "art/interface/Nick16Font.ART",
    "art/interface/pork12font.art",
]
# Download URLs for fonts not kept in the repo (fetched on first use).
FONT_URLS = {MAIN_FONT[0]: MAIN_FONT[2]}

# Vanilla bitmap font art -> (TTF/OTF in fonts/vanilla/, variable-font weight
# or None). See fonts/vanilla/README.md for where each came from.
FONT_TTF = {
    "art/interface/arial10font.art": ("arial.ttf", None),
    "art/interface/ArialB12Font.ART": ("arialbd.ttf", None),
    "art/interface/BookmanOldBold18Font.ART": ("texgyrebonum-bold.otf", None),
    "art/interface/casablanca16font.art": ("IMFeENrm28P.ttf", None),
    # Round 8 pass 9 (#14/#28): schematic discipline title + description in a
    # typewriter face, "Special Elite" (Google Fonts, Astigmatic). Exclusive
    # to schematic_ui.c (not in MAIN_FONT_ARTS), so it's safe to remap
    # directly rather than needing a SchemDescFont-style art alias.
    "art/interface/CasablancaAntique30Font.ART": ("SpecialElite-Regular.ttf", None),
    "art/interface/SchemDescFont.ART": ("SpecialElite-Regular.ttf", None),
    # Round 8 pass 9 (#19): Save/Load Game screen's save names in a header-
    # ish weight, not the rest-of-UI Outfit 400 (MAIN_FONT). Exclusive to
    # mainmenu_ui.c's two save/load list+preview fonts, so a SaveLoadListFont
    # alias (like SchemDescFont/LogbookFont) rather than a MAIN_FONT_ARTS
    # entry, which would apply 400 everywhere Flare12Font.ART is used.
    # Pass 12 feedback (#47): "make save games list forced bold" - 700.
    "art/interface/SaveLoadListFont.ART": ("Outfit[wght].ttf", 700),
    # CharStatsFont: drawn as a face (FACE_EXTRA_ARTS, round 8 pass 11 #65);
    # this cell fit only backs the 1x/fallback glyphs.
    "art/interface/CharStatsFont.ART": ("Outfit[wght].ttf", 400),
    "art/interface/ClarendonBLK18Font.ART": ("Coustard-Black.ttf", None),
    "art/interface/Cloister18Font.ART": ("CloisterBlack.ttf", None),
    "art/interface/Comic12Font.ART": ("comic.ttf", None),
    "art/interface/Courier10Font.ART": ("cour.ttf", None),
    "art/interface/Elga12Font.ART": ("CrimsonPro[wght].ttf", 800),
    "art/interface/LogbookFont.ART": ("JimNightshade-Regular.ttf", None),
    "art/interface/Garmond6Font.ART": ("EBGaramond.ttf", 600),
    "art/interface/Garmond8Font.ART": ("EBGaramond.ttf", 600),
    "art/interface/Icons17Font.ART": ("MORPHEUS.TTF", None),
    "art/interface/Icons32Font.ART": ("MORPHEUS.TTF", None),
    "art/interface/NewsIconsFont.ART": (None, None),
    "art/interface/BookImagesFont.ART": (None, None),
    "art/interface/rollerfont.art": ("arial.ttf", None),
    "art/interface/Georgia30Font.ART": ("georgia.ttf", None),
    "art/interface/LatinXCN30Font.ART": ("StintUltraCondensed-Regular.ttf", None),
    "art/interface/morph15font.art": ("MORPHEUS.TTF", None),
    "art/interface/Morph30Font.ART": ("MORPHEUS.TTF", None),
    "art/morph15font.ART": ("MORPHEUS.TTF", None),
    "art/interface/NewTimes16Font.ART": ("times.ttf", None),
    "art/interface/Pepper20Font.ART": ("Fondamento-Italic.ttf", None),
    "art/interface/Swiss921Font.ART": ("Anton-Regular.ttf", None),
    "art/interface/Zurich16Font.ART": ("ArchivoNarrow[wght].ttf", 700),
    "art/interface/Zurich20Font.ART": ("ArchivoNarrow[wght].ttf", 700),
}
FONT_TTF.update({rel: MAIN_FONT[:2] for rel in MAIN_FONT_ARTS})
# Fonts fitted uniformly (see _fit_glyph `uniform`): one x scale and the
# shared baseline for every letter. CasablancaAntique30Font/SchemDescFont
# (round 8 pass 10 #14/#28/#44/#60): per-glyph fitting was re-squeezing
# individual letters back toward the (narrow) alias source's own ink width
# even with FONT_XSCALE=1.0 overriding the font-wide measurement - uniform
# mode skips that per-glyph override entirely.
FONT_UNIFORM: set[str] = set(MAIN_FONT_ARTS) | {
    "art/interface/CasablancaAntique30Font.ART",
    "art/interface/SchemDescFont.ART",
    # CharStatsFont's alias source (morph15font.art) is a decorative
    # blackletter face with irregular per-glyph ink boxes/baselines -
    # per-glyph fitting Outfit's plain letterforms against that produced
    # jumping baselines and overlapping letters (round 8 pass 10 #54).
    "art/interface/CharStatsFont.ART",
}


def ensure_font(ttf: str) -> Path:
    """fonts/vanilla/<ttf>, downloaded from FONT_URLS if missing."""
    path = FONT_DIR / ttf
    if not path.exists() and ttf in FONT_URLS:
        import urllib.request
        print(f"downloading {ttf} from {FONT_URLS[ttf]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FONT_URLS[ttf]) as r:
            path.write_bytes(r.read())
    return path
# Fonts whose vanilla lower case is drawn as capitals.
FONT_CAPS_ONLY = {"art/interface/LatinXCN30Font.ART"}
# Glyphs that are pictures, not letters (Icons17/32Font's digits are the
# tech discipline icons, BookImagesFont is the book maps): ESRGAN-upscaled
# from the vanilla coverage instead. A string lists the characters; an int
# takes every glyph at least that wide. Fonts with no TTF (None) upscale
# their remaining glyphs smoothly.
FONT_PICTO = {
    "art/interface/Icons17Font.ART": "0123456789",
    "art/interface/Icons32Font.ART": "0123456789:",
    "art/interface/NewsIconsFont.ART": 12,
    "art/interface/BookImagesFont.ART": 12,
    "art/interface/rollerfont.art": "0123456789",
}


def _is_picto(rel: str, ch: str | None, width: int) -> bool:
    picto = FONT_PICTO.get(rel)
    if isinstance(picto, int):
        return width >= picto
    return picto is not None and ch is not None and ch in picto
# Font arts that don't exist in vanilla: a copy of another font art (same
# cells and advances) the engine loads from the game's data/ folder, so one
# UI can have its own glyphs. LogbookFont = the logbook body in a
# handwriting (logbook_ui.c, interface art 4000 in name.c). SchemDescFont =
# the schematic screen's discipline description in a typewriter face
# (schematic_ui.c, interface art 4003 in name.c, round 8 pass 9 #14/#28).
FONT_ALIAS = {
    "art/interface/LogbookFont.ART": "art/interface/Flare12Font.ART",
    "art/interface/SchemDescFont.ART": "art/interface/Flare12Font.ART",
    "art/interface/SaveLoadListFont.ART": "art/interface/Flare12Font.ART",
    "art/interface/CharStatsFont.ART": "art/interface/morph15font.art",
}

# Stroke thinning in HD px for single-weight fonts that render too heavy;
# negative emboldens (hairline scripts).
FONT_THIN: dict[str, float] = {"art/interface/LogbookFont.ART": -0.5}
# Cap height as a fraction of the vanilla 'H' (fonts with tall loops that
# would not fit the vanilla cells otherwise).
FONT_CAP: dict[str, float] = {
    # Special Elite is wider per unit cap height than its alias sources'
    # (Flare12Font/CasablancaAntique30Font) vanilla cells - _fit_glyph hard
    # clips a glyph's rendered width to the cell (gw = min(cw, ...)), so
    # even at x_scale 1.0 + uniform fit, letters were still getting clipped
    # back down. Shrinking the point size (not the cell) is what actually
    # avoids that clip - "changing only font size" (round 8 pass 10 #60).
    "art/interface/CasablancaAntique30Font.ART": 0.8,
    "art/interface/SchemDescFont.ART": 0.8,
}
# Glyph size after the fit, about each glyph's ink centre and the baseline
# (layout unchanged): "2 pt smaller" Grenze (user, round 8: 12 -> 10, 14 -> 12).
FONT_SCALE: dict[str, float] = {
    "art/interface/Flare12Font.ART": 10 / 12,
    "art/interface/Flare14Font.ART": 12 / 14,
    "art/interface/pork12font.art": 10 / 12,
}
# Fonts placed with the font-wide scale only (see _fit_glyph).
FONT_FREE_FIT: set[str] = {"art/interface/LogbookFont.ART"}
# Per-font x scale override, skipping the letter-width-ratio measurement
# below: for an aliased font swapped to an unrelated TTF (FONT_ALIAS), that
# ratio is measured against the ALIAS SOURCE's letter widths, not the new
# TTF's own - e.g. Special Elite (a typewriter face) matched against
# Flare12Font/CasablancaAntique30Font's much narrower vanilla letters came
# out visibly squeezed (round 8 pass 9/10 #14/#28/#44 - "we're changing only
# font size, not squeezing it horizontally to fit"). 1.0 keeps the TTF's own
# natural proportions; layout (advances/wrapping/centring) still follows the
# vanilla cell per cmd_hd_fonts' docstring, unaffected by this.
FONT_XSCALE: dict[str, float] = {
    "art/interface/CasablancaAntique30Font.ART": 1.0,
    "art/interface/SchemDescFont.ART": 1.0,
}


def _thin(cov: np.ndarray, r: float, keep: float) -> np.ndarray:
    """Erode anti-aliased coverage by up to r px per side, but never thin a
    stroke below 2*keep px wide (hairlines and serifs survive; only the
    heavy stems lose weight). Works on the signed distance to the 0.5
    contour; the result has a ~1 HD px soft edge (4 supersampled px)."""
    from scipy import ndimage
    inside = cov >= 0.5
    din = ndimage.distance_transform_edt(inside)
    d = np.where(inside, din - 0.5, 0.5 - ndimage.distance_transform_edt(~inside))
    if r < 0:  # embolden: grow every stroke by -r
        return np.clip((d - r) / 4.0 + 0.5, 0.0, 1.0)
    # each pixel's stroke half-width: the inside distance at its nearest
    # medial-axis (ridge) pixel
    ridge = inside & (din >= ndimage.maximum_filter(din, size=3))
    if not ridge.any():
        return cov
    _, (iy, ix) = ndimage.distance_transform_edt(~ridge, return_indices=True)
    half = din[iy, ix]
    r_eff = np.clip(half - keep, 0.0, r)
    return np.clip((d - r_eff) / 4.0 + 0.5, 0.0, 1.0) * (cov > 0)


FONT_DIR = Path(__file__).resolve().parent / "fonts" / "vanilla"


def _font_frames(wd: Path, basename: str) -> list[dict]:
    """Every glyph frame of an unpacked font art: coverage (0..1, from the
    grey palette; index 0 is the key) and size."""
    num_frames, _ = read_ini_frame_count(wd / (basename + ".ini"))
    frames = []
    for _, frame, bmp in hd_frame_bmps(wd, basename, num_frames, False):
        w, h = read_bmp_dims(bmp)
        h = abs(h)
        idx = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
        pal = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
        cov = np.where(idx == 0, 0.0, pal[idx].max(axis=2) / 255.0)
        frames.append(dict(frame=frame, w=w, h=h, cov=cov))
    return frames


def _font_advances(ini: Path) -> dict[int, int]:
    """Frame -> advance (hot x, "center_x" in the unpacked ini)."""
    adv, cur = {}, None
    for line in ini.read_text(errors="replace").splitlines():
        m = re.match(r"frame (\d+):", line)
        if m:
            cur = int(m.group(1))
            continue
        m = re.match(r"center_x: (-?\d+)", line)
        if m and cur is not None:
            adv[cur] = int(m.group(1))
    return adv


def _ink_box(a: np.ndarray, thresh: float = 0.25):
    ys, xs = np.nonzero(a > thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _glyph_char(frame: int) -> str | None:
    # tig_font: frame = byte - 31, text is Windows-1252.
    try:
        return bytes([frame + 31]).decode("cp1252")
    except (UnicodeDecodeError, ValueError):
        return None


def _scale_glyph(cell: np.ndarray, f: float, base: int) -> np.ndarray:
    """Resize a cell's ink by f about its ink box's centre column and the
    baseline row `base` (supersampled px)."""
    box = _ink_box(cell, 0.02)
    if box is None:
        return cell
    x0, y0, x1, y1 = box
    g = Image.fromarray((cell[y0:y1, x0:x1] * 255).astype(np.uint8), "L")
    gw, gh = max(1, int(round(g.width * f))), max(1, int(round(g.height * f)))
    g = np.asarray(g.resize((gw, gh), Image.LANCZOS), dtype=np.float32) / 255.0
    nx = int(round((x0 + x1) / 2 - gw / 2))
    ny = int(round(base - (base - y0) * f))
    out = np.zeros_like(cell)
    H, W = cell.shape
    sx0, sy0 = max(0, -nx), max(0, -ny)
    dx0, dy0 = max(0, nx), max(0, ny)
    w, h = min(gw - sx0, W - dx0), min(gh - sy0, H - dy0)
    if w > 0 and h > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = g[sy0:sy0 + h, sx0:sx0 + w]
    return out


def _squash_rows(canvas: np.ndarray, top: int, h: int, bl: int) -> np.ndarray:
    """Rows [top, top + h) of `canvas`; ink above / below that is squashed
    in, separately above and below the baseline row `bl`."""
    ys = np.nonzero(canvas.max(axis=1) > 0.02)[0]
    out = canvas[top:top + h].copy()
    if len(ys) == 0:
        return out
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    if y0 < top and bl > top:
        part = Image.fromarray((canvas[y0:bl] * 255).astype(np.uint8), "L")
        part = part.resize((canvas.shape[1], bl - top), Image.LANCZOS)
        out[:bl - top] = np.asarray(part, dtype=np.float32) / 255.0
    if y1 > top + h and bl < top + h:
        part = Image.fromarray((canvas[bl:y1] * 255).astype(np.uint8), "L")
        part = part.resize((canvas.shape[1], top + h - bl), Image.LANCZOS)
        out[bl - top:] = np.asarray(part, dtype=np.float32) / 255.0
    return out


def _fit_glyph(a: np.ndarray, base: int, vb, ch: str, x_scale: float, by: int, cw: int, chh: int, k: int,
               pen: int | None = None, uniform: bool = False, pad: int = 0):
    """Place one TTF-rendered glyph (coverage `a`, baseline row `base`) in
    its cell (cw x chh, baseline row `by`, all in supersampled px; k =
    vanilla px -> supersampled px) so its ink lands where the vanilla ink
    (box `vb`, vanilla px) was. Width: the vanilla ink width, within
    0.8-1.25 of the font-wide x scale (thin glyphs keep the font scale),
    centred on the vanilla ink. Height: the font's size on the shared
    baseline, unless that misses the vanilla ink by more than 1.5 vanilla
    px at the top or bottom, or it is a digit (old-style figures ->
    lining): then fitted to the vanilla ink box (0.8-1.25). With `pen`
    (the TTF pen column in `a`): no per-glyph fit, the font-wide scale on
    the baseline with the pen at the cell's left edge, as the TTF lays it
    out (fonts nothing like vanilla's, e.g. a joined handwriting whose
    strokes must meet). `uniform` (FONT_UNIFORM): the font-wide x scale
    and the shared baseline for every letter, centred on the vanilla ink
    (no per-glyph width / height fit - that made letters jump and change
    width, #225); digits still get the height fit. Whatever still
    overflows the cell is squashed above / below the baseline separately.
    Returns (coverage, x0, y0) or None."""
    tc = _ink_box(a, 0.02)
    tb = _ink_box(a, 0.5) or tc
    if tc is None:
        return None
    vx0, vy0, vx1, vy1 = (v * k for v in vb)
    vy0, vy1 = vy0 + pad, vy1 + pad  # cell padded above by `pad` rows
    tw, th = max(1, tb[2] - tb[0]), max(1, tb[3] - tb[1])
    free = pen is not None
    sx = x_scale
    if vb[2] - vb[0] >= 3 and not free and not uniform:
        sx = x_scale * float(np.clip((vx1 - vx0) / (tw * x_scale), 0.8, 1.25))
    sy = 1.0
    oy = by - (base - tb[1])  # cell row of the ink-box top on the shared baseline
    tol = 1.5 * k
    off = abs(oy - vy0) > tol or abs(oy + th - vy1) > tol
    if not free and vb[3] - vb[1] >= 2 and (ch.isdigit() or (off and not uniform)):
        want = (vy1 - vy0) / th
        sy = float(np.clip(want, 0.8, 1.25))
        oy = vy0 if sy == want else vy1 - th * sy
    ox = (tb[0] - pen) * sx if free else (vx0 + vx1) / 2 - tw * sx / 2

    g = Image.fromarray((a[tc[1]:tc[3], tc[0]:tc[2]] * 255).astype(np.uint8), "L")
    gw = min(cw, max(1, int(round(g.width * sx))))
    x0 = int(round(ox + (tc[0] - tb[0]) * sx))
    x0 = min(max(x0, 0), cw - gw)
    y0 = int(round(oy + (tc[1] - tb[1]) * sy))
    asc = min(max(base - tc[1], 0), g.height)  # rows above the baseline
    desc = g.height - asc
    a_h, d_h = int(round(asc * sy)), int(round(desc * sy))
    bl = y0 + a_h  # cell row of the baseline
    if y0 < 0 or bl + d_h > chh:
        a_h, d_h = min(a_h, max(bl, 0)), min(d_h, max(chh - bl, 0))
        y0 = bl - a_h
    parts = []
    if asc > 0 and a_h > 0:
        parts.append(g.crop((0, 0, g.width, asc)).resize((gw, a_h), Image.LANCZOS))
    if desc > 0 and d_h > 0:
        parts.append(g.crop((0, asc, g.width, g.height)).resize((gw, d_h), Image.LANCZOS))
    if not parts:
        return None
    return np.concatenate([np.asarray(q, dtype=np.float32) / 255.0 for q in parts], axis=0), x0, y0


# PC lens ring overlays (the round view in charedit/logbook/inventory/...,
# intgame_pc_lens_redraw()): 89x89 squares, rim + wood corners around a
# transparent inscribed-circle hole the view shows through. The upscaled
# 1x hole edge is a staircase (#121); replace it with an analytic,
# anti-aliased circle. (The corners' mismatch with the panel, #110/#111, is
# handled in the engine - intgame_pc_lens_hd_corners().) Also the world
# map's Nav_Cvr, whose round hole (the NavButton socket) had the same
# staircase plus a dark fringe (#120). Value: radius percentile of the
# hole's edge pixels to put the circle at (higher trims a dark fringe).
LENS_RINGS = {
    "PCWinCvr": 50, "Char_PCC": 50, "Lns_Bart": 50, "Lns_Loot": 50, "Lns_Papr": 50,
    "OptionsPCLens": 50, "SaveLoadPCLens": 50,
    "Nav_Cvr": 97,
}


# Inventory-panel lenses (inven_ui.c: Lns_Papr over PDoll at (11, 9),
# Lns_Bart / Lns_Loot over Barter / Loot at (16, 17)): the ring is half in
# the panel (outside the 89x89 box) and half in the lens art's corners, and
# the two were upscaled separately - two rings that don't meet, and a square
# seam around the box (round 8 pass 11 #35). The vanilla panel + lens
# composite around the box is upscaled once (LENS_CONTEXT_MARGIN 1x px of
# context); the lens sidecar's colour becomes that upscale's box (its alpha
# - the analytic hole, LENS_RINGS - is kept) and the panel sidecar gets the
# upscale around the box, feathered into its own pixels over the outer
# LENS_CONTEXT_FEATHER 1x px of the margin. Originals: work/_lens_context_originals/.
LENS_CONTEXT = [
    ("PDoll", "Lns_Papr", 11, 9),
    ("Barter", "Lns_Bart", 16, 17),
    ("Barter_Follower", "Lns_Bart", 16, 17),
    ("Loot", "Lns_Loot", 16, 17),
]
LENS_CONTEXT_MARGIN = 20
LENS_CONTEXT_FEATHER = 8


def cmd_hd_lens_context(only: str | None = None) -> None:
    stage = config.WORK_DIR / "_lens_context"
    stage.mkdir(parents=True, exist_ok=True)
    s = HD_SCALE
    m = LENS_CONTEXT_MARGIN
    lens_done: set[str] = set()
    for panel, lens, lx, ly in LENS_CONTEXT:
        if only is not None and only.lower() not in (panel + lens).lower():
            continue
        prel, lrel = f"art/interface/{panel}.ART", f"art/interface/{lens}.ART"
        ppath, lpath = hd_out_dir(prel) / "r0_f0.png", hd_out_dir(lrel) / "r0_f0.png"
        if not ppath.exists() or not lpath.exists():
            print(f"{panel}/{lens}: no sidecar, skipped")
            continue
        for path in (ppath, lpath):
            backup = config.WORK_DIR / "_lens_context_originals" / path.parent.name / path.name
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, backup)

        def vanilla(rel: str) -> tuple[np.ndarray, np.ndarray]:
            wd = cmd_unpack(rel, quiet=True)
            bmp = sorted(wd.glob("*_0.bmp"))[0]
            w, h = read_bmp_dims(bmp)
            h = abs(h)
            idx = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
            pal = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
            return pal[idx], idx == 0

        prgb, pkey = vanilla(prel)
        lrgb, lkey = vanilla(lrel)
        comp = prgb.copy()
        key = pkey.copy()
        box = comp[ly:ly + lrgb.shape[0], lx:lx + lrgb.shape[1]]
        box[~lkey] = lrgb[~lkey]
        key[ly:ly + lrgb.shape[0], lx:lx + lrgb.shape[1]] &= lkey
        # keyed holes (Loot, Barter_Follower) would bleed their key colour
        # into the ring
        comp = np.clip(inpaint_colorkey(comp, key), 0, 255).astype(np.uint8)
        x0, y0 = max(0, lx - m), max(0, ly - m)
        x1, y1 = min(comp.shape[1], lx + lrgb.shape[1] + m), min(comp.shape[0], ly + lrgb.shape[0] + m)
        crop = comp[y0:y1, x0:x1]
        src_png, hd_png = stage / f"{panel}.png", stage / f"{panel}_x4.png"
        Image.fromarray(np.ascontiguousarray(crop), "RGB").save(src_png)
        run_esrgan(src_png, hd_png, config.REALESRGAN_MODEL)
        up = np.asarray(load_and_validate(hd_png, (crop.shape[1] * s, crop.shape[0] * s), "hd-lens-context")
                        .convert("RGB")).astype(np.float32)
        corr = structural_corr(crop, np.asarray(Image.fromarray(up.astype(np.uint8)).resize(crop.shape[1::-1], Image.BOX)))
        if corr < BATCH_OUTPUT_MIN_CORR:
            raise RuntimeError(f"{panel}: upscale doesn't match its source (corr {corr:.2f})")

        # the panel: the upscale around the box, feathered at the margin's
        # outer edge (image borders count as inside)
        pan = np.asarray(Image.open(config.WORK_DIR / "_lens_context_originals" / ppath.parent.name / ppath.name)
                         .convert("RGBA")).astype(np.float32)
        hh, ww = up.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32) + 0.5
        f = LENS_CONTEXT_FEATHER * s
        dl = xx if x0 > 0 else np.full_like(xx, f)
        dt = yy if y0 > 0 else np.full_like(yy, f)
        dr = ww - xx if x1 < prgb.shape[1] else np.full_like(xx, f)
        db = hh - yy if y1 < prgb.shape[0] else np.full_like(yy, f)
        wgt = np.clip(np.minimum(np.minimum(dl, dr), np.minimum(dt, db)) / f, 0, 1)[..., None]
        region = pan[y0 * s:y1 * s, x0 * s:x1 * s, :3]
        pan[y0 * s:y1 * s, x0 * s:x1 * s, :3] = region * (1 - wgt) + up * wgt
        Image.fromarray(np.clip(pan + 0.5, 0, 255).astype(np.uint8), "RGBA").save(ppath)

        # the lens: colour from the upscale's box, its own (analytic) alpha
        if lens not in lens_done:
            lens_done.add(lens)
            la = np.asarray(Image.open(config.WORK_DIR / "_lens_context_originals" / lpath.parent.name / lpath.name)
                            .convert("RGBA")).copy()
            bx, by = (lx - x0) * s, (ly - y0) * s
            la[..., :3] = np.clip(up[by:by + la.shape[0], bx:bx + la.shape[1]] + 0.5, 0, 255).astype(np.uint8)
            Image.fromarray(la, "RGBA").save(lpath)
        print(f"{panel}/{lens}: ring upscaled in context (corr {corr:.2f})")


# Round 8 pass 12 (#41): the character sheet's HP/fatigue -/+ buttons used
# the stat ovals' Char_Minus/Char_Plus (their own brass bezel, not the olive
# knobs painted beside the heart/cross circles). The engine now draws
# Char_HTFTMinus/Plus (770/771, 21x21, unused in vanilla) there instead;
# their HD sidecars are the panel's own knob (Char_Maint, HP row, cut to its
# disc) with the red sign of the matching Char_Minus/Char_Plus frame on it.
# name -> (sign art, 1x box x on Char_Maint, knob centre in the HD box).
HTFT_KNOBS = {
    "Char_HTFTMinus": ("Char_Minus", 408, (46.0, 44.4)),
    "Char_HTFTPlus": ("Char_Plus", 464, (43.8, 44.2)),
}
HTFT_KNOB_Y = 143      # 1x box y (HP row; the fatigue row's knobs match)
HTFT_KNOB_RADIUS = 33.0  # HD px, the knob's outer brass edge
# frame -> sign frame drawn (vanilla: 0 up, 1 down, 2 hover, 3 disabled);
# the pressed one is the bold hover sign brightened (its own glow frame
# doubled on the knob), disabled is the bare knob.
HTFT_SIGN_FRAMES = {0: (0, 1.0), 1: (2, 1.3), 2: (2, 1.0), 3: None}


def cmd_hd_htft_knob(only: str | None = None) -> None:
    root = config.HD_OVERLAY_DIR / "art" / "interface"
    s = 4
    bg = np.asarray(Image.open(root / "Char_Maint" / "r0_f0.png").convert("RGB")).astype(np.float32)
    for name, (sign_art, bx, (kx, ky)) in HTFT_KNOBS.items():
        if only is not None and only.lower() not in name.lower():
            continue
        size = 21 * s
        crop = bg[HTFT_KNOB_Y * s:HTFT_KNOB_Y * s + size, bx * s:bx * s + size]
        yy, xx = np.mgrid[0:size, 0:size]
        alpha = np.clip((HTFT_KNOB_RADIUS - np.hypot(xx - kx, yy - ky)) / 2.0 + 0.5, 0, 1)
        # Pass 12 feedback (#60/#61, "+ moving a bit on hover"): the sign
        # frames are registered alike, so all of them take the up frame's
        # offset - each one centred on its own centroid moved the glowing
        # hover sign by a few px.
        offset = None
        for frame, sign in sorted(HTFT_SIGN_FRAMES.items(), key=lambda kv: kv[1] is None or kv[1][0] != 0):
            out = crop.copy()
            if sign is not None:
                sf, gain = sign
                im = np.asarray(Image.open(root / sign_art / f"r0_f{sf}.png").convert("RGBA")).astype(np.float32)
                red = im[..., 0] - (im[..., 1] + im[..., 2]) / 2
                w = np.clip((red - 50) / 70, 0, 1) * (im[..., 3] / 255)
                c = (im.shape[1] - 1) / 2
                iy, ix = np.mgrid[0:im.shape[0], 0:im.shape[1]]
                w *= np.hypot(ix - c, iy - c) < 24  # the sign, not the bezel's tints
                if offset is None:
                    ys, xs = np.nonzero(w > 0.5)
                    offset = (int(round(ky - ys.mean())), int(round(kx - xs.mean())))
                oy, ox = offset
                rgb = np.clip(im[..., :3] * gain, 0, 255)
                for y, x in zip(*np.nonzero(w > 0)):
                    ty, tx = y + oy, x + ox
                    if 0 <= ty < size and 0 <= tx < size:
                        out[ty, tx] = out[ty, tx] * (1 - w[y, x]) + rgb[y, x] * w[y, x]
            dest = root / name / f"r0_f{frame}.png"
            backup = config.WORK_DIR / "_htft_originals" / name / dest.name
            if dest.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dest, backup)
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.dstack([out, alpha * 255]).clip(0, 255).astype(np.uint8), "RGBA").save(dest)
        print(f"{name}: knob + {sign_art} sign -> {root / name}")


def _fit_circle(ex: np.ndarray, ey: np.ndarray) -> tuple[float, float]:
    """Algebraic circle fit (x^2 + y^2 + D x + E y + F = 0) -> centre."""
    m = np.stack([ex, ey, np.ones_like(ex)], 1)
    sol, *_ = np.linalg.lstsq(m, -(ex * ex + ey * ey), rcond=None)
    return float(-sol[0] / 2.0), float(-sol[1] / 2.0)


# Outer silhouettes with a dark, bumpy 1x outline that upscaled into a
# serrated edge (#185 world map nav bar): the edge facing the outside is
# pulled in by `erode` HD px and re-smoothed (Gaussian `sigma` + smoothstep);
# a round lens hole (LENS_RINGS) is left as it is. name -> (erode, sigma).
OUTER_SMOOTH = {"Nav_Cvr": (5.0, 4.5)}
# name -> (first, last+1 HD row) where the ring stands alone above the bar.
OUTER_RING_CUT = {"Nav_Cvr": (14, 40)}


def cmd_hd_outer_smooth(only: str | None = None) -> None:
    from scipy import ndimage
    for name, (erode, sigma) in OUTER_SMOOTH.items():
        if only is not None and only.lower() not in name.lower():
            continue
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        backup = config.WORK_DIR / "_outer_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        for src in sorted(backup.glob("*.png")):
            im = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
            h, w = im.shape[:2]
            a = im[..., 3] / 255.0
            yy, xx = np.mgrid[:h, :w].astype(np.float32)
            labels, _ = ndimage.label(a < 0.5)
            border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
            outside = np.isin(labels, list(border))
            keep = np.zeros_like(outside)
            if name in LENS_RINGS:  # the hole: everything near its circle stays
                hole = labels == labels[h // 2, w // 2]
                op = a >= 0.5
                nb = ndimage.binary_dilation(op) & hole
                if nb.sum() >= 16:
                    cx, cy = _fit_circle(xx[nb] + 0.5, yy[nb] + 0.5)
                    r = float(np.percentile(np.hypot(xx[nb] + 0.5 - cx, yy[nb] + 0.5 - cy), 97))
                    keep = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) < r + 12
                    outside &= ~keep
            d = ndimage.distance_transform_edt(~outside)
            t = np.clip((gaussian_blur_2d(np.minimum(a, np.clip(d - erode, 0, 1)), sigma) - 0.3) / 0.4, 0, 1)
            na = t * t * (3 - 2 * t)
            near = (d < erode + 4 * sigma) & ~keep
            im[..., 3] = np.where(near, na, a) * 255.0
            if name in OUTER_RING_CUT and keep.any():
                # Vanilla has a nub on top of the ring (1x rows 0-1, #207):
                # cut everything above the ring's outer circle, fitted to
                # its free-standing flanks (above where the bar joins).
                top, join = OUTER_RING_CUT[name]
                rs = []
                for y in range(top, join):
                    xs = np.nonzero(im[y, :, 3] >= 128)[0]
                    xs = xs[np.abs(xs + 0.5 - cx) < r + 40]
                    if len(xs):
                        rs += [np.hypot(xs[0] + 0.5 - cx, y + 0.5 - cy), np.hypot(xs[-1] + 0.5 - cx, y + 0.5 - cy)]
                if rs:
                    ro = float(np.median(rs))
                    dd = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
                    zone = (yy < join) & (np.abs(xx + 0.5 - cx) < ro + 2)
                    im[..., 3] = np.where(zone, im[..., 3] * np.clip(ro - dd + 0.5, 0, 1), im[..., 3])
                    print(f"{name}/{src.name}: cut above ring r={ro:.1f}")
            Image.fromarray(np.clip(im + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
            print(f"{name}/{src.name}: outer edge smoothed")


def lens_ring_alpha(rgba: np.ndarray, pct: float = 50) -> tuple[np.ndarray, float]:
    """New alpha for a ring sidecar with a round transparent hole (the
    component under the image centre): the circle is least-squares fitted
    to the hole's edge pixels, its radius set at their `pct` percentile;
    within a band around it an analytic circle with a 1 px soft edge,
    everywhere else the original alpha. Returns (alpha 0..255, radius)."""
    from scipy import ndimage
    h, w = rgba.shape[:2]
    a = rgba[..., 3].astype(np.float32) / 255.0
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    labels, _ = ndimage.label(a < 0.5)
    hole = labels == labels[h // 2, w // 2]
    op = a >= 0.5
    nb = np.zeros_like(op)
    nb[1:, :] |= op[:-1, :]
    nb[:-1, :] |= op[1:, :]
    nb[:, 1:] |= op[:, :-1]
    nb[:, :-1] |= op[:, 1:]
    edge = hole & nb
    if edge.sum() < 16:
        return rgba[..., 3].copy(), 0.0
    cx, cy = _fit_circle(xx[edge] + 0.5, yy[edge] + 0.5)
    d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    r_in = float(np.percentile(d[edge], pct))
    band = np.abs(d - r_in) < 6.0
    alpha = np.where(band, np.clip(d - r_in + 0.5, 0, 1), np.where(hole, 0.0, a))
    return (alpha * 255.0 + 0.5).astype(np.uint8), r_in


# Round buttons whose vanilla art is a dark disc with a clipped crescent of
# rim on one side (it sits in a socket drawn by the panel under it): the
# sidecar is cut to the disc so only the socket's own ring shows around it.
# name -> (centre x, centre y, radius) in vanilla px.
DISC_MASKS = {
    "lilgrnbut": (10.625, 11.125, 11.0),  # HUD fate / sleep buttons in IntTop
    # Loot window's take-all button: its own blotchy dark wood recess stood
    # out on Loot_PD's plain HD wood (round 8 pass 11 #13) - keep the gold
    # ring (outer r ~18.4) plus a thin shadow.
    "TakeAllButt": (24.3125, 24.125, 19.0),
}


def cmd_hd_disc_mask(only: str | None = None) -> None:
    """Cut DISC_MASKS buttons' sidecars to a circle (1 HD px feather). The
    unmasked sidecars are kept in work/_disc_originals/ and always used as
    the input, so reruns don't compound."""
    for name, (cx, cy, r) in DISC_MASKS.items():
        if only is not None and only.lower() not in name.lower():
            continue
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        backup = config.WORK_DIR / "_disc_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        for src in sorted(backup.glob("*.png")):
            a = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
            yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]] + 0.5
            d = np.hypot(xx - cx * HD_SCALE, yy - cy * HD_SCALE)
            a[..., 3] *= np.clip((r * HD_SCALE - d) / 2 + 0.5, 0, 1)
            Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
        print(f"{name}: disc mask r={r} at ({cx}, {cy})")


# Schematic drawings (rules/schematic.mes "Drawing" entries) fill
# Schematic_Base's transparent hole at (240, 146) exactly. Each has its own
# copy of the paper, whose tone never quite matches the base's around the
# hole - a hard colour step along the drawing's edges (round 8 pass 11 #15,
# vanilla has it too). The base is shared, so each drawing is corrected
# instead. Target tone: the base paper around the hole (SCHEM_TONE_BAND HD
# px deep per side, smoothed along the edge; only a few rows of paper below
# the hole before the torn edge) filled harmonically (Laplace) across the
# hole. The drawing's own paper tone: a wide normalized blur over its paper
# pixels only (the drawn object and grid lines masked out). The difference
# is added to every pixel - edges meet the base, and the drawing's own
# brighter middle follows the surrounding paper too (pass 11 #37: "inner
# rect a little brighter"). RGB only, low-frequency, so detail is kept. Alpha is forced
# opaque: the eight SchemTitle_* pages had ~230 alpha on their last row and
# column, which showed the black behind the hole as a thin frame.
# Originals are kept in work/_schem_originals/ and always used as the input.
SCHEM_HOLE = (240, 146)
SCHEM_TONE_BAND = {"left": 40, "right": 40, "top": 40, "bottom": 8}
SCHEM_TONE_SIGMA = 40
SCHEM_TONE_GRID = 8  # Laplace fill solved on a 1/8 grid, then upsampled
SCHEM_TONE_RAMP = 160
SCHEM_TONE_EDGE = 6  # HD px the leftover edge tone is measured over (#49)


def _schematic_drawing_arts() -> list[str]:
    import re
    mes = None
    names = None
    for root in config.EXTRACTED_DAT_ROOTS:
        if mes is None and (root / "rules" / "schematic.mes").exists():
            mes = (root / "rules" / "schematic.mes").read_text(encoding="latin-1")
        if names is None and (root / "art" / "interface" / "interface.mes").exists():
            names = dict((int(a), b) for a, b in re.findall(
                r"\{(\d+)\}\{([^}]*)\}", (root / "art" / "interface" / "interface.mes").read_text(encoding="latin-1")))
    # entry base + 2 is the drawing's interface art number (SCHEMATIC_F_ART_NUM)
    arts = set()
    for num, value in re.findall(r"\{(\d+)\}\{(\d+)\}", mes):
        if int(num) % 10 == 2 and int(value) in names:
            arts.add(names[int(value)].rsplit(".", 1)[0])
    return sorted(arts)


def _harmonic_fill(left: np.ndarray, right: np.ndarray, top: np.ndarray, bottom: np.ndarray,
                   gh: int, gw: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(gh, gw, 3) grid whose border is the four edge profiles (sampled at
    rows/cols) and whose interior is their Laplace (harmonic) fill."""
    field = np.zeros((gh, gw, 3), np.float32)
    field[:, 0] = left[rows]
    field[:, -1] = right[rows]
    field[0, :] = top[cols]
    field[-1, :] = bottom[cols]
    for cy, cx in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        field[cy, cx] = (field[cy, cx] + (left if cx == 0 else right)[rows[cy]]) / 2
    inner = field[1:-1, 1:-1]
    inner[:] = np.concatenate([field[[0, -1]].reshape(-1, 3), field[:, [0, -1]].reshape(-1, 3)]).mean(axis=0)
    for _ in range(4000):
        inner[:] = (field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:]) / 4
    return field


# Round 8 pass 12 (#40, "tiny black frame" around the schematic drawings):
# Schematic_Base's hole for the drawing (1x 240,146 295x225) has a 1-HD-px
# ring of alpha ~200 just outside the drawing's rect, which the black under
# the window showed through. Made opaque (its colour is already the paper's).
SCHEM_BASE_HOLE = (240, 146, 295, 225)


def cmd_hd_schem_base_edge() -> None:
    path = config.HD_OVERLAY_DIR / "art" / "interface" / "Schematic_Base" / "r0_f0.png"
    im = np.asarray(Image.open(path).convert("RGBA")).copy()
    s = im.shape[1] // 800
    x, y, w, h = (v * s for v in SCHEM_BASE_HOLE)
    ring = np.zeros(im.shape[:2], bool)
    ring[y - 3:y + h + 3, x - 3:x + w + 3] = True
    ring[y:y + h, x:x + w] = False  # under the drawing: left alone
    fixed = ring & (im[..., 3] < 255)
    im[fixed, 3] = 255
    Image.fromarray(im, "RGBA").save(path)
    print(f"Schematic_Base: {int(fixed.sum())} hole-edge px made opaque")


# Round 8 pass 12 feedback (#47, "why don't we have a chain there as in
# other scroll bars"): the loot/barter scroll tracks have a gold chain painted
# into their panels, the Save/Load list's track (SaveLoadBackground, the
# scrollbar at 1x 213,111 12x232) is bare wood. One period (two links) of the
# Loot panel's chain is cut out (alpha from its brightness over the black
# track), tiled down the track between the arrow buttons with a soft shadow.
# Idempotent: works from a backup of the original sidecar.
SAVELOAD_CHAIN_SRC = (1336, 1364, 1040)  # Loot HD x0, x1, first row of the period
SAVELOAD_CHAIN_PERIOD = 105  # HD rows, best pixel autocorrelation (two links)
# art -> (1x centre x of the chain = the thumb's, 1x rows a little under
# both arrow buttons). Pass 12 feedback #52: the character sheet's scheme
# list (Scheme_Rot, scrollbar 209,58 17x255) too.
# Pass 13 feedback #89: rows exactly between the arrow buttons (up arrow 15
# rows, down arrow 14; the chain showed past the arrows' slanted sides).
SAVELOAD_CHAIN_TARGETS = {
    "SaveLoadBackground": (218.5, (111 + 15, 111 + 232 - 14)),
    "Scheme_Rot": (217.5, (58 + 15, 58 + 255 - 14)),
}


def _scroll_chain_period():
    """One period of the Loot panel's vanilla chain: (alpha, colour, ink columns)."""
    backup = config.WORK_DIR / "_scroll_chain_originals" / "Loot"
    src = backup if backup.exists() else hd_out_dir("art/interface/Loot.ART")
    loot = np.asarray(Image.open(src / "r0_f0.png").convert("RGB")).astype(np.float32)
    x0, x1, y = SAVELOAD_CHAIN_SRC
    seg = loot[y:y + SAVELOAD_CHAIN_PERIOD, x0:x1]
    alpha = np.clip((seg.max(axis=2) - 8) / 30, 0, 1)
    color = np.clip(np.where(alpha[..., None] > 0.02, seg / np.maximum(alpha[..., None], 1e-3), 0), 0, 255)
    ink = np.nonzero(alpha.max(axis=0) > 0.2)[0]
    return alpha, color, ink


def _paint_scroll_chain(bg: np.ndarray, x_c: float, y_top: int, y_bottom: int) -> tuple[int, int, int]:
    """Tile the chain over `bg` (float RGBA, HD) centred on 1x x_c, 1x rows y_top..y_bottom."""
    from scipy import ndimage

    alpha, color, ink = _scroll_chain_period()
    shadow_src = ndimage.gaussian_filter(alpha, 3)
    s = HD_SCALE
    y0, y1 = y_top * s, y_bottom * s
    h = y1 - y0
    reps = h // SAVELOAD_CHAIN_PERIOD + 1
    a = np.tile(alpha, (reps, 1))[:h]
    c = np.tile(color, (reps, 1, 1))[:h]
    cx = int(round(x_c * s - (ink[0] + ink[-1]) / 2))
    region = bg[y0:y1, cx:cx + a.shape[1], :3]
    shadow = np.roll(np.tile(shadow_src, (reps, 1))[:h], (3, 3), axis=(0, 1)) * 0.6
    region *= 1 - shadow[..., None]
    region[:] = region * (1 - a[..., None]) + c * a[..., None]
    return cx, y0, y1


def cmd_hd_saveload_chain() -> None:
    for name, (x_c, (y_top, y_bottom)) in SAVELOAD_CHAIN_TARGETS.items():
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        backup = config.WORK_DIR / "_saveload_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        bg = np.asarray(Image.open(backup / "r0_f0.png").convert("RGBA")).astype(np.float32)
        cx, y0, y1 = _paint_scroll_chain(bg, x_c, y_top, y_bottom)
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGBA").save(out_dir / "r0_f0.png")
        print(f"{name}: chain at x {cx}.., y {y0}..{y1}")


# Pass 13 feedback #89: the loot/barter panels' own chains ran under both
# arrow buttons (showing past their slanted sides) and sat ~1 1x px left of
# the arrows' centre. The vanilla chain is painted out (each row filled with
# the median of the track either side of it) and repainted between the
# arrows, centred on them. Scrollbars (inven_ui.c): loot 330,136 17x256,
# barter 330,168 17x224 (Barter_Follower = barter with cycle buttons);
# arrows 11 wide centred in the rect, up 15 rows, down 14.
# art -> (scrollbar y, height, HD rows holding the vanilla chain)
SCROLL_CHAIN_PANELS = {
    "Loot": (136, 256, (590, 1520)),
    "Barter": (168, 224, (676, 1564)),
    "Barter_Follower": (168, 224, (718, 1520)),
}
SCROLL_CHAIN_ERASE_X = (1334, 1368)  # HD columns of the vanilla chain (+ margin)
SCROLL_CHAIN_SIDE_X = ((1316, 1330), (1370, 1384))  # clean track either side
SCROLL_CHAIN_BAR_X = 330  # 1x, all three
SCROLL_CHAIN_BAR_W = 17


def cmd_hd_scroll_chains() -> None:
    backups = config.WORK_DIR / "_scroll_chain_originals"
    # Loot's backup first: it is the chain source for every target.
    for name in SCROLL_CHAIN_PANELS:
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        if not (backups / name).exists():
            shutil.copytree(out_dir, backups / name)
    x_c = SCROLL_CHAIN_BAR_X + (SCROLL_CHAIN_BAR_W - 11) // 2 + 11 / 2
    for name, (bar_y, bar_h, (ey0, ey1)) in SCROLL_CHAIN_PANELS.items():
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        bg = np.asarray(Image.open(backups / name / "r0_f0.png").convert("RGBA")).astype(np.float32)
        ex0, ex1 = SCROLL_CHAIN_ERASE_X
        (l0, l1), (r0, r1) = SCROLL_CHAIN_SIDE_X
        side = np.concatenate([bg[ey0:ey1, l0:l1, :3], bg[ey0:ey1, r0:r1, :3]], axis=1)
        bg[ey0:ey1, ex0:ex1, :3] = np.median(side, axis=1)[:, None, :]
        cx, y0, y1 = _paint_scroll_chain(bg, x_c, bar_y + 15, bar_y + bar_h - 14)
        Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGBA").save(out_dir / "r0_f0.png")
        print(f"{name}: chain erased HD rows {ey0}..{ey1}, repainted centre {x_c} (1x), x {cx}.., y {y0}..{y1}")


# Round 8 pass 12 feedback (#48): the worldmap's bottom plate (MapMain, with
# Nav_Cvr over its top half at 1x 294,341) has two pill frames whose dark
# inner groove sits ~16 HD px from the outer edge at the top but ~8 at the
# bottom. Everything inside each pill but its outer 5 HD px is moved up
# NAV_PILL_SHIFT px, so both rims are ~12. Pills are stadiums (HD x0, y0,
# x1, y1). Idempotent: works from backups of both sidecars.
NAV_PILLS = [(1184, 1444, 1458, 1566), (1716, 1444, 1990, 1566)]
NAV_PILL_SHIFT = 4
NAV_CVR_POS = (294, 341)  # 1x, wmap_ui.c wmap_ui_nav_cvr_frame (382 - window y 41)


# Pass 13 #62: character creation's arrow buttons (portrait Big_Grn_L/R,
# gender/race/background MPCycleLeft/RightButton). The upscales kept the
# vanilla's dithered checker in the lit arrows, two different reds, and the
# big ones' clipped crescent of rim on one side. Rebuilt: one disc (frame
# 0's, arrow inpainted) for every frame so nothing shifts on hover, vector
# arrows (1x polygons traced from the vanilla) in shared colours, and the
# big discs cut round and centred on CreateCharacterBase's sockets.
# name -> (arrow polygon (1x px edges), disc mask (cx, cy, r) 1x or None,
#          content shift (dx, dy) 1x)
CYCLE_ARROWS = {
    "Big_Grn_L": ([(8.0, 15.5), (14.8, 8.7), (14.8, 12.0), (22.0, 12.0), (22.0, 19.0),
                   (14.8, 19.0), (14.8, 22.3)], (15.0, 14.8, 15.6), (0.0, -0.7)),
    "Big_Grn_R": ([(23.0, 15.5), (16.2, 8.7), (16.2, 12.0), (9.0, 12.0), (9.0, 19.0),
                   (16.2, 19.0), (16.2, 22.3)], (16.0, 14.8, 15.6), (1.0, -0.7)),
    # own partial gold ring clashed with the pill's socket ring: cut inside it
    "MPCycleLeftButton": ([(8.0, 11.5), (14.0, 5.8), (14.0, 17.2)], (11.05, 11.3, 10.4), (-0.45, -0.2)),
    "MPCycleRightButton": ([(16.0, 11.5), (10.0, 5.8), (10.0, 17.2)], (11.85, 11.1, 10.4), (0.35, -0.4)),
}
# frame -> (top colour, bottom colour) of the arrow's vertical gradient
CYCLE_ARROW_COLORS = {
    0: ((170, 18, 36), (112, 4, 22)),    # idle
    1: ((222, 52, 50), (160, 22, 28)),   # pressed
    2: ((250, 84, 74), (196, 34, 36)),   # hover
}


def cmd_hd_cycle_arrows(only: str | None = None) -> None:
    import cv2
    from PIL import ImageDraw
    from scipy import ndimage

    s = HD_SCALE
    ss = 4  # supersampling for the polygon
    for name, (poly, disc, (dx, dy)) in CYCLE_ARROWS.items():
        if only is not None and only.lower() not in name.lower():
            continue
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        backup = config.WORK_DIR / "_cycle_arrow_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        base = np.asarray(Image.open(backup / "r0_f0.png").convert("RGBA")).astype(np.float32)
        h, w = base.shape[:2]

        # the old arrow (red-dominant pixels, grown) inpainted out of the disc
        rgb = base[..., :3]
        red = (rgb[..., 0] > rgb[..., 1] + 35) & (rgb[..., 0] > 60) & (base[..., 3] > 128)
        red = ndimage.binary_dilation(red, iterations=5)
        clean = cv2.inpaint(np.ascontiguousarray(rgb.clip(0, 255).astype(np.uint8)),
                            red.astype(np.uint8) * 255, 8, cv2.INPAINT_TELEA).astype(np.float32)
        disc_im = np.dstack([clean, base[..., 3]])

        if dx or dy:
            m = np.float32([[1, 0, dx * s], [0, 1, dy * s]])
            disc_im = cv2.warpAffine(disc_im, m, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        if disc is not None:
            cx, cy, r = disc
            yy, xx = np.mgrid[0:h, 0:w] + 0.5
            d = np.hypot(xx - cx * s, yy - cy * s)
            disc_im[..., 3] = np.minimum(disc_im[..., 3], np.clip(r * s - d + 0.5, 0, 1) * 255)

        # arrow coverage, supersampled
        big = Image.new("L", (w * ss, h * ss), 0)
        ImageDraw.Draw(big).polygon([((x + dx) * s * ss, (y + dy) * s * ss) for x, y in poly], fill=255)
        cov = np.asarray(big.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
        ys = [(y + dy) * s for _, y in poly]
        t = np.clip((np.arange(h)[:, None] - min(ys)) / max(1.0, max(ys) - min(ys)), 0, 1)
        # bevel: lit along the top-left edge, shaded along the bottom-right
        inner = ndimage.gaussian_filter(cov, 2.0)
        gy, gx = np.gradient(inner)
        bevel = np.clip(-(gx + gy) * 6.0, -1, 1) * cov
        # the arrow sits in a recess: a soft dark halo under it
        shadow = ndimage.gaussian_filter(ndimage.shift(cov, (1.5, 1.5), order=1), 2.0) * 0.55

        for f, (top, bot) in CYCLE_ARROW_COLORS.items():
            top_c, bot_c = np.array(top, np.float32), np.array(bot, np.float32)
            col = top_c[None, None] * (1 - t[..., None]) + bot_c[None, None] * t[..., None]
            col = col + np.where(bevel[..., None] > 0, (255 - col) * bevel[..., None] * 0.45,
                                 col * bevel[..., None] * 0.5)
            out = disc_im.copy()
            out[..., :3] *= (1 - shadow[..., None])
            out[..., :3] = out[..., :3] * (1 - cov[..., None]) + col * cov[..., None]
            Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA").save(out_dir / f"r0_f{f}.png")
        print(f"{name}: disc + vector arrow, {len(CYCLE_ARROW_COLORS)} frames")


def cmd_hd_nav_pill_rim() -> None:
    dirs = {}
    for name in ("MapMain", "Nav_Cvr"):
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        backup = config.WORK_DIR / "_nav_pill_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        dirs[name] = (out_dir, Image.open(backup / "r0_f0.png").convert("RGBA"))
    bg = dirs["MapMain"][1]
    nav = dirs["Nav_Cvr"][1]
    nx, ny = NAV_CVR_POS[0] * HD_SCALE, NAV_CVR_POS[1] * HD_SCALE
    comp = bg.copy()
    comp.alpha_composite(nav, (nx, ny))
    a = np.asarray(comp).copy()
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    moved = np.roll(a, -NAV_PILL_SHIFT, axis=0)
    mask = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in NAV_PILLS:
        x0, y0, x1, y1 = x0 + 5, y0 + 5, x1 - 5, y1 - 5
        r = (y1 - y0) / 2
        dx = np.maximum(np.maximum(x0 + r - xx, xx - (x1 - r)), 0)
        mask |= dx ** 2 + (yy - (y0 + y1) / 2) ** 2 <= r * r
    a[mask] = moved[mask]
    out_bg = np.asarray(bg).copy()
    out_bg[mask] = a[mask]
    Image.fromarray(out_bg, "RGBA").save(dirs["MapMain"][0] / "r0_f0.png")
    n = np.asarray(nav).copy()
    nh, nw = n.shape[:2]
    sub_mask = mask[ny:ny + nh, nx:nx + nw] & (n[..., 3] > 0)
    n[..., :3][sub_mask] = a[ny:ny + nh, nx:nx + nw, :3][sub_mask]
    Image.fromarray(n, "RGBA").save(dirs["Nav_Cvr"][0] / "r0_f0.png")
    print(f"MapMain/Nav_Cvr: {int(mask.sum())} pill px moved up {NAV_PILL_SHIFT}")


# Round 8 pass 12 feedback (#59): the charedit Skills_Window's four gauges.
# - The glass tube (1x x 59..179, the liquid's span) only had its cylinder
#   in HD rows 17..57 of each slot; below it the "reservoir" read as erased
#   (near-black). It is vanilla's own dark glass: now kept as upscaled with
#   vanilla's on-screen dark lift applied (#94).
# - The 1..5 under it were 3-4 px blobs no model can read: each cell
#   (24 px, dividers at 1x x 59 + 24 k, strip rows 115..122 of slot 0) has
#   the old digit inpainted away and the digit drawn in SKILL_GAUGE_FONT,
#   dark engraved ink with a light lower-right edge.
# Slots are 66 px apart from y 87. Idempotent: works from a backup.
SKILL_GAUGE_SLOTS = [87 + 66 * k for k in range(4)]
SKILL_GAUGE_GLASS_X = (59, 179)
SKILL_GAUGE_CELLS_X = 59
SKILL_GAUGE_STRIP_Y = (115, 123)  # 1x rows of the wooden strip, slot 0
SKILL_GAUGE_FONT = ("texgyrebonum-bold.otf", 26)  # HD px
SKILL_GAUGE_INK = (58, 34, 18)
SKILL_GAUGE_RAIL_ROWS = (86, 107)  # HD rows below each slot's top: the rail under the tube
SKILL_GAUGE_RAIL_X = (214, 293)    # HD x: left bracket .. where the vanilla rail starts
SKILL_GAUGE_GLASS_ROWS = (12, 88)  # HD rows below each slot's top: the glass (1x 90..108)
SKILL_GAUGE_LIQUID_EDGE_ALPHA = 0.35  # liquid's alpha at its top/bottom edge (#97)
SKILL_GAUGE_LIQUID_EDGE_ROWS = 5      # 1x rows from the edge to full alpha
SKILL_GAUGE_GLASS_LIFT = 30.0    # added at black, fading out by lum 110 (see #94)
# vanilla on screen: mean glass level of 1x rows 90.. of slot 0 (user capture, #94)
SKILL_GAUGE_GLASS_PROFILE = (90, [54, 55, 52, 76, 94, 101, 102, 87, 64, 45, 43, 40, 30, 32, 32, 28, 17, 8, 20])


def cmd_hd_skill_gauge() -> None:
    import cv2
    from PIL import ImageDraw, ImageFont

    out_dir = hd_out_dir("art/interface/Skills_Window.ART")
    backup = config.WORK_DIR / "_skill_gauge_originals" / "Skills_Window"
    if not backup.exists():
        shutil.copytree(out_dir, backup)
    s = HD_SCALE
    im = np.asarray(Image.open(backup / "r0_f0.png").convert("RGBA")).astype(np.float32)
    x0, x1 = SKILL_GAUGE_GLASS_X[0] * s, SKILL_GAUGE_GLASS_X[1] * s
    font = ImageFont.truetype(str(ensure_font(SKILL_GAUGE_FONT[0])), SKILL_GAUGE_FONT[1])
    for slot in SKILL_GAUGE_SLOTS:
        y = slot * s
        # pass 13 #94: the tube is the vanilla glass as upscaled - its
        # reflection, dark middle and textured lower glass give the cylinder
        # its volume (stretching rows of it, #59/#92, flattened that). Vanilla
        # on screen lifts the dark tones (black ~30, 16 -> 45, 90 -> 100),
        # which is what shows the lower glass; the same lift is applied here,
        # then each row is scaled to vanilla's on-screen mean
        # (SKILL_GAUGE_GLASS_PROFILE: highlight, bottom shadow line).
        g0, g1 = SKILL_GAUGE_GLASS_ROWS
        glass = im[y + g0:y + g1, x0:x1, :3]
        lum = glass.mean(axis=2, keepdims=True)
        lifted = glass + SKILL_GAUGE_GLASS_LIFT * np.clip(1 - lum / 110.0, 0, 1)
        p0, prof = SKILL_GAUGE_GLASS_PROFILE
        centers = [(p0 + i - 87 + 0.5) * s for i in range(len(prof))]
        mid = x0 + (x1 - x0) // 8, x1 - (x1 - x0) // 4  # clear of the end caps
        cur = lifted[:, mid[0] - x0:mid[1] - x0].mean(axis=(1, 2))
        want = np.interp(np.arange(g0, g1) + 0.5, centers, prof)
        gain = want / np.maximum(cur, 1)
        # smooth the gain over the 1x row so the texture keeps its own detail
        gain = np.convolve(np.pad(gain, s // 2, mode="edge"), np.ones(s) / s, mode="valid")[:g1 - g0]
        lifted = lifted * np.clip(gain, 0.2, 2.0)[:, None, None]
        rows = np.arange(g0, g1)
        cols = np.arange(x0, x1)
        w = (np.clip(np.minimum(rows - g0, g1 - 1 - rows) / 3.0, 0, 1)[:, None, None]
             * np.clip(np.minimum(cols - x0, x1 - 1 - cols) / 6.0, 0, 1)[None, :, None])
        im[y + g0:y + g1, x0:x1, :3] = glass * (1 - w) + lifted * w

        # pass 13 #65: the brass rail under the tube starts ~20 px short of
        # the left bracket (vanilla too) - a dark gap above the "1". Carry
        # the rail on to the bracket, copied from just right of its end.
        r0, r1 = SKILL_GAUGE_RAIL_ROWS
        d0, d1 = SKILL_GAUGE_RAIL_X
        src = im[y + r0:y + r1, d1 + 8:d1 + 8 + (d1 - d0)].copy()
        rw = (np.clip(np.minimum(np.arange(r1 - r0), r1 - r0 - 1 - np.arange(r1 - r0)) / 3.0, 0, 1)[:, None, None]
              * np.clip(np.minimum(np.arange(d1 - d0) / 3.0, (d1 - d0 - 1 - np.arange(d1 - d0)) / 4.0 + 0.5), 0, 1)[None, :, None])
        im[y + r0:y + r1, d0:d1] = im[y + r0:y + r1, d0:d1] * (1 - rw) + src * rw

        sy0 = (SKILL_GAUGE_STRIP_Y[0] - 87 + slot) * s
        sy1 = (SKILL_GAUGE_STRIP_Y[1] - 87 + slot) * s
        rgb = np.ascontiguousarray(im[..., :3].clip(0, 255).astype(np.uint8))
        for k in range(5):
            cx = (SKILL_GAUGE_CELLS_X + 24 * k + 12) * s
            mask = np.zeros(rgb.shape[:2], np.uint8)
            mask[sy0 + 3:sy1 - 3, cx - 16:cx + 16] = 255
            rgb = cv2.inpaint(rgb, mask, 6, cv2.INPAINT_TELEA)
        im[..., :3] = rgb
        layer = Image.new("RGBA", (im.shape[1], im.shape[0]), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        cy = (sy0 + sy1) / 2
        for k in range(5):
            cx = (SKILL_GAUGE_CELLS_X + 24 * k + 12) * s
            dr.text((cx + 1.5, cy + 1.5), str(k + 1), font=font, fill=(235, 200, 140, 110), anchor="mm")
            dr.text((cx, cy), str(k + 1), font=font, fill=SKILL_GAUGE_INK + (255,), anchor="mm")
        base = Image.fromarray(im.clip(0, 255).astype(np.uint8), "RGBA")
        base.alpha_composite(layer)
        im = np.asarray(base).astype(np.float32)
    Image.fromarray(im.clip(0, 255).astype(np.uint8), "RGBA").save(out_dir / "r0_f0.png")
    print(f"Skills_Window: {len(SKILL_GAUGE_SLOTS)} gauges - glass filled, 1..5 redrawn")

    # pass 13 #97: the liquid (SkilGauge, charedit draws its 1x rows 3..23)
    # read as a flat opaque block over the glass. Its top and bottom rows
    # fade (alpha SKILL_GAUGE_LIQUID_EDGE_ALPHA at the edge, full
    # SKILL_GAUGE_LIQUID_EDGE_ROWS 1x rows in), so the tube's shading shows
    # through them.
    liq_dir = hd_out_dir("art/interface/SkilGauge.ART")
    liq_backup = config.WORK_DIR / "_skill_gauge_originals" / "SkilGauge"
    if not liq_backup.exists():
        shutil.copytree(liq_dir, liq_backup)
    liq = np.asarray(Image.open(liq_backup / "r0_f0.png").convert("RGBA")).astype(np.float32)
    top, bottom = 3 * s, 24 * s
    n = SKILL_GAUGE_LIQUID_EDGE_ROWS * s
    r = np.arange(liq.shape[0], dtype=np.float32) + 0.5
    t = np.clip(np.minimum(r - top, bottom - r) / n, 0, 1)
    a0 = SKILL_GAUGE_LIQUID_EDGE_ALPHA
    liq[..., 3] *= (a0 + (1 - a0) * np.sin(t * np.pi / 2))[:, None]
    Image.fromarray(liq.clip(0, 255).astype(np.uint8), "RGBA").save(liq_dir / "r0_f0.png")
    print("SkilGauge: liquid edges faded")


def cmd_hd_schem_tone(only: str | None = None) -> None:
    from scipy import ndimage

    base = np.asarray(Image.open(hd_out_dir("art/interface/Schematic_Base.ART") / "r0_f0.png")
                      .convert("RGBA")).astype(np.float32)
    x0, y0 = SCHEM_HOLE[0] * HD_SCALE, SCHEM_HOLE[1] * HD_SCALE
    bl, br, bt, bb = (SCHEM_TONE_BAND[k] for k in ("left", "right", "top", "bottom"))
    g = SCHEM_TONE_GRID
    done = 0
    for name in _schematic_drawing_arts():
        if only is not None and only.lower() not in name.lower():
            continue
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        if not out_dir.exists():
            continue
        backup = config.WORK_DIR / "_schem_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        for src in sorted(backup.glob("*.png")):
            a = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
            h, w = a.shape[:2]
            rgb = a[..., :3]
            gh, gw = h // g + 1, w // g + 1
            rows = np.linspace(0, h - 1, gh).astype(int)
            cols = np.linspace(0, w - 1, gw).astype(int)

            def profile(strip: np.ndarray, axis: int, mask: np.ndarray | None = None) -> np.ndarray:
                # mean across the band (over `mask` pixels only), smoothed
                # along the edge - a normalized blur, so masked-out stretches
                # take their neighbours' tone
                if mask is None:
                    mask = np.ones(strip.shape[:2], np.float32)
                num = (strip * mask[..., None]).sum(axis=axis)
                den = mask.sum(axis=axis)[:, None]
                sig = SCHEM_TONE_SIGMA
                if den.sum() < 1:
                    # no paper at all along this edge: plain mean
                    num = strip.sum(axis=axis)
                    den = np.full_like(den, strip.shape[axis])
                for _ in range(4):
                    n = ndimage.gaussian_filter1d(num, sig, axis=0, mode="nearest")
                    d = ndimage.gaussian_filter1d(den, sig, axis=0, mode="nearest")
                    if d.min() > 1e-3:
                        break
                    sig *= 3  # a long masked-out stretch: widen until covered
                return n / np.maximum(d, 1e-6)

            # paper: not much darker than the drawing's typical edge paper
            lum = rgb.mean(axis=-1)
            ring = np.concatenate([lum[:, :bl].ravel(), lum[:, w - br:].ravel(), lum[:bt].ravel(), lum[h - bb:].ravel()])
            paper = (lum >= np.median(ring) - 30).astype(np.float32)

            # target: the base paper around the hole, filled across it
            target = _harmonic_fill(profile(base[y0:y0 + h, x0 - bl:x0, :3], 1),
                                    profile(base[y0:y0 + h, x0 + w:x0 + w + br, :3], 1),
                                    profile(base[y0 - bt:y0, x0:x0 + w, :3], 0),
                                    profile(base[y0 + h:y0 + h + bb, x0:x0 + w, :3], 0), gh, gw, rows, cols)
            # the drawing's own edges, filled the same way: its tone if it
            # had no bulge of its own
            flat = _harmonic_fill(profile(rgb[:, :bl], 1, paper[:, :bl]), profile(rgb[:, w - br:], 1, paper[:, w - br:]),
                                  profile(rgb[:bt], 0, paper[:bt]), profile(rgb[h - bb:], 0, paper[h - bb:]),
                                  gh, gw, rows, cols)

            # its actual paper tone: normalized blur over all-paper cells
            # (not the drawn object or the grid lines)
            small = np.stack([np.asarray(Image.fromarray(rgb[..., c], "F").resize((gw, gh), Image.BOX))
                              for c in range(3)], axis=-1)
            wsmall = np.asarray(Image.fromarray(paper, "F").resize((gw, gh), Image.BOX))
            wsmall = np.where(wsmall > 0.9, wsmall, 0)
            sigma = 120 / g
            num = ndimage.gaussian_filter(small * wsmall[..., None], (sigma, sigma, 0), mode="nearest")
            den = ndimage.gaussian_filter(wsmall, sigma, mode="nearest")[..., None]
            own = num / np.maximum(den, 1e-6)

            # edges: exactly onto the base; inside (ramping in over
            # SCHEM_TONE_RAMP HD px): the drawing's bulge taken out too
            yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
            edge = np.minimum(np.minimum(xx, gw - 1 - xx), np.minimum(yy, gh - 1 - yy)) * g
            ramp = np.clip(edge / SCHEM_TONE_RAMP, 0, 1)[..., None]
            delta = (target - flat) + ramp * (flat - own)
            corr = np.stack([np.asarray(Image.fromarray(delta[..., c], "F").resize((w, h), Image.BILINEAR))
                             for c in range(3)], axis=-1)
            a[..., :3] = rgb + corr

            # Pass 12 feedback (#49, "still tiny bit off the color"): the
            # smoothed profiles left every drawing's outermost paper ~2/3/7
            # RGB yellower than the base next to it (most at the right). The
            # per-side leftover, measured on the paper pixels (the lighter
            # half - not the grid lines) of the SCHEM_TONE_EDGE outermost px
            # against the base's same-width band, is taken out by one more
            # harmonic field.
            def paper_median(px: np.ndarray) -> np.ndarray:
                px = px.reshape(-1, 3)
                lum = px.mean(axis=1)
                return np.median(px[lum > np.percentile(lum, 50)], axis=0)

            e = SCHEM_TONE_EDGE
            fixed = a[..., :3]
            side = {
                "left": paper_median(base[y0:y0 + h, x0 - e:x0, :3]) - paper_median(fixed[:, :e]),
                "right": paper_median(base[y0:y0 + h, x0 + w:x0 + w + e, :3]) - paper_median(fixed[:, w - e:]),
                "top": paper_median(base[y0 - e:y0, x0:x0 + w, :3]) - paper_median(fixed[:e]),
                "bottom": paper_median(base[y0 + h:y0 + h + e, x0:x0 + w, :3]) - paper_median(fixed[h - e:]),
            }
            rest = _harmonic_fill(np.tile(side["left"], (h, 1)), np.tile(side["right"], (h, 1)),
                                  np.tile(side["top"], (w, 1)), np.tile(side["bottom"], (w, 1)), gh, gw, rows, cols)
            a[..., :3] += np.stack([np.asarray(Image.fromarray(rest[..., c], "F").resize((w, h), Image.BILINEAR))
                                    for c in range(3)], axis=-1)
            a[..., 3] = 255
            Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
        done += 1
    print(f"schematic drawings tone-matched: {done}")


# Near-black panel boxes turned pure black (#202: the HUD money/ammo box
# is (0,8,8) teal-black in IntBotom and (2,2,2) behind the ammo icons, next
# to the counter's (0,0,0) fill - read as a grey patch). BLACK_BOXES: 1x
# seed points; the connected region of exactly the seed's vanilla colour is
# blackened in the sidecar (dark HD pixels only, 1 HD px feather).
# BLACK_BG: arts whose dark border-connected background goes to 0.
BLACK_BOXES = {"IntBotom": [(100, 80)]}
BLACK_BG = ["Ammo_Icon_Arrows", "Ammo_Icon_Bullets", "Ammo_Icon_Charges",
            "Ammo_Icon_Fuel", "Ammo_Icon_Gold", "Ammo_Icon_Mana"]


def cmd_hd_black_fill(only: str | None = None) -> None:
    from PIL import ImageFilter
    from scipy import ndimage

    def feather(m, w, h):
        img = Image.fromarray((m * 255).astype(np.uint8), "L")
        if img.size != (w, h):
            img = img.resize((w, h), Image.NEAREST)
        return np.asarray(img.filter(ImageFilter.GaussianBlur(0.7)), dtype=np.float32) / 255.0

    for name, seeds in BLACK_BOXES.items():
        if only is not None and only.lower() not in name.lower():
            continue
        rel = f"art/interface/{name}.ART"
        wd = cmd_unpack(rel, quiet=True)
        out_dir = hd_out_dir(rel)
        backup = config.WORK_DIR / "_black_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        c = np.asarray(Image.open(sorted(wd.glob("*.bmp"))[0]).convert("RGB")).astype(np.int32)
        region = np.zeros(c.shape[:2], bool)
        for x, y in seeds:
            same = (c == c[y, x]).all(axis=2)
            lab, _ = ndimage.label(same)
            region |= lab == lab[y, x]
        src = backup / "r0_f0.png"
        a = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
        m = feather(region, a.shape[1], a.shape[0])
        m *= a[..., :3].max(axis=2) <= 40
        a[..., :3] *= (1.0 - m)[..., None]
        Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
        print(f"{name}: {int(region.sum())} 1x px -> black")

    for name in BLACK_BG:
        if only is not None and only.lower() not in name.lower():
            continue
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        backup = config.WORK_DIR / "_black_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        for src in sorted(backup.glob("*.png")):
            a = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
            dark = a[..., :3].max(axis=2) <= 12
            lab, _ = ndimage.label(dark)
            edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
            bg = np.isin(lab, list(edge))
            m = feather(bg, a.shape[1], a.shape[0]) * (a[..., :3].max(axis=2) <= 40)
            a[..., :3] *= (1.0 - m)[..., None]
            Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
        print(f"{name}: background -> black")


# Icons painted into a panel, re-upscaled with another model and pasted
# into the panel's sidecar (#204: Inventor's ammo icons with remacri). name
# -> (model, 1x context crop x, y, w, h, [icon rects x, y, w, h in 1x panel
# coordinates]). Only the icons are taken (feathered rect); the patch is
# tone-matched to the current sidecar on a ring of wood around each icon.
# The current sidecar (after hd-frame-smooth) is backed up once to
# work/_icon_originals/ and always used as the base.
ICON_PATCHES = {
    "Inventor": ("remacri-4x", (310, 44, 108, 199), [
        (338, 68, 26, 18),   # gold
        (335, 90, 30, 20),   # arrows
        (337, 114, 28, 20),  # bullets
        (343, 138, 13, 20),  # fuel
        (341, 162, 19, 20),  # mana (kettle)
    ]),
}


def cmd_hd_icon_patch(only: str | None = None) -> None:
    from scipy import ndimage
    stage = config.WORK_DIR / "_icon_patch"
    stage.mkdir(parents=True, exist_ok=True)
    for name, (model, (cx, cy, cw, ch), rects) in ICON_PATCHES.items():
        if only is not None and only.lower() not in name.lower():
            continue
        rel = f"art/interface/{name}.ART"
        wd = cmd_unpack(rel, quiet=True)
        out_dir = hd_out_dir(rel)
        backup = config.WORK_DIR / "_icon_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        van = np.asarray(Image.open(sorted(wd.glob("*.bmp"))[0]).convert("RGB"), dtype=np.uint8)
        src = van[cy:cy + ch, cx:cx + cw]
        src_png = stage / f"{name}.png"
        hd_png = stage / f"{name}_{model}.png"
        Image.fromarray(src, "RGB").save(src_png)
        run_esrgan(src_png, hd_png, model)
        up = load_and_validate(hd_png, (cw * HD_SCALE, ch * HD_SCALE), "hd-icon-patch")
        corr = structural_corr(src, np.asarray(up.convert("RGB").resize((cw, ch), Image.BOX)), np.ones(src.shape[:2], bool))
        if corr < BATCH_OUTPUT_MIN_CORR:
            raise RuntimeError(f"{model} output doesn't match {name} crop (corr {corr:.2f})")
        up = np.asarray(up.convert("RGB"), dtype=np.float32)
        base_img = Image.open(backup / "r0_f0.png").convert("RGBA")
        base = np.asarray(base_img, dtype=np.float32).copy()
        S = HD_SCALE
        for x, y, w, h in rects:
            pad = 4
            X0, Y0 = (x - pad) * S, (y - pad) * S
            W, H = (w + 2 * pad) * S, (h + 2 * pad) * S
            cur = base[Y0:Y0 + H, X0:X0 + W, :3]
            new = up[(y - pad - cy) * S:(y - pad - cy) * S + H, (x - pad - cx) * S:(x - pad - cx) * S + W].copy()
            inner = np.zeros((H, W), bool)
            inner[pad * S:(pad + h) * S, pad * S:(pad + w) * S] = True
            ring = ~inner
            for c in range(3):
                mc, sc = cur[..., c][ring].mean(), cur[..., c][ring].std() + 1e-3
                mn, sn = new[..., c][ring].mean(), new[..., c][ring].std() + 1e-3
                new[..., c] = (new[..., c] - mn) * min(sc / sn, 1.0) + mc
            m = ndimage.gaussian_filter(inner.astype(np.float32), 1.5 * S)
            m = np.clip((m - 0.25) / 0.5, 0, 1)
            base[Y0:Y0 + H, X0:X0 + W, :3] = cur * (1 - m[..., None]) + new * m[..., None]
        Image.fromarray(np.clip(base + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / "r0_f0.png")
        print(f"{name}: {len(rects)} icon(s) from {model}")


# Inventory paperdoll slot silhouettes (inven_ui.c item_ui_item_silhouette_nums,
# blitted at inven_ui_inventory_paperdoll_inv_slot_rects over PDoll): opaque
# rects carrying their own copy of the slot grid, a few levels off PDoll's.
# Upscaled separately, the grid under an empty slot changed colour in HD
# (#189 #190). The sidecar keeps only the silhouette (vanilla pixels that
# differ from PDoll's there); the panel's own HD grid shows through (the
# window background is redrawn before them on every refresh).
CVR_SLOTS = {
    "CVR_Helmet": (151, 107), "CVR_Ring1": (247, 107), "CVR_Ring2": (279, 107),
    "CVR_Medalion": (247, 139), "CVR_Weapon": (23, 170), "CVR_Shield": (247, 171),
    "CVR_Armor": (119, 171), "CVR_Gauntlet": (55, 107), "CVR_Boot": (150, 331),
}


def cmd_hd_cvr_mask(only: str | None = None) -> None:
    from PIL import ImageFilter
    from scipy import ndimage
    bg_wd = cmd_unpack("art/interface/PDoll.ART", quiet=True)
    bg = np.asarray(Image.open(sorted(bg_wd.glob("*.bmp"))[0]).convert("RGB")).astype(np.int32)
    for name, (x, y) in CVR_SLOTS.items():
        if only is not None and only.lower() not in name.lower():
            continue
        rel = f"art/interface/{name}.ART"
        wd = cmd_unpack(rel, quiet=True)
        out_dir = hd_out_dir(rel)
        backup = config.WORK_DIR / "_cvr_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        for bmp in sorted(wd.glob("*.bmp")):
            frame = int(bmp.stem.rsplit("_", 1)[1])
            c = np.asarray(Image.open(bmp).convert("RGB")).astype(np.int32)
            h, w = c.shape[:2]
            under = bg[y:y + h, x:x + w]
            if under.shape != c.shape:
                print(f"{name}: slot rect off the panel, skipped")
                continue
            sil = np.abs(under - c).sum(axis=2) > 24
            sil = ndimage.binary_dilation(sil, iterations=1)
            m = Image.fromarray((sil * 255).astype(np.uint8), "L").resize((w * HD_SCALE, h * HD_SCALE), Image.BILINEAR)
            m = np.asarray(m.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255.0
            src = backup / f"r0_f{frame}.png"
            a = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
            if a.shape[:2] != m.shape:
                print(f"{name}: sidecar {a.shape[:2]} != {m.shape}, skipped")
                continue
            a[..., 3] *= m
            Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
            print(f"{name} f{frame}: silhouette {sil.mean():.0%} of the slot")


# Gold panel frames that came out wobbly / stair-stepped (#191): a 5 px
# median (rounds the contours) + slight blur, only near the gold, feathered.
# name -> vanilla-px rects left alone (icons drawn into the panel).
FRAME_SMOOTH = {"PDoll": [], "Inventor": [(325, 55, 400, 230)]}


def cmd_hd_frame_smooth(only: str | None = None, size: int = 5) -> None:
    from scipy import ndimage
    for name in FRAME_SMOOTH:
        if only is not None and only.lower() not in name.lower():
            continue
        out_dir = hd_out_dir(f"art/interface/{name}.ART")
        backup = config.WORK_DIR / "_frame_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        for src in sorted(backup.glob("*.png")):
            a = np.asarray(Image.open(src).convert("RGBA")).astype(np.float32)
            rgb = a[..., :3]
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            gold = ndimage.binary_opening((r - b > 60) & (r > 100) & (g > 50), iterations=1)
            zone = ndimage.binary_dilation(gold, iterations=6)
            for x0, y0, x1, y1 in FRAME_SMOOTH[name]:
                zone[y0 * HD_SCALE:y1 * HD_SCALE, x0 * HD_SCALE:x1 * HD_SCALE] = False
            w = ndimage.gaussian_filter(zone.astype(np.float32), 2.0)[..., None]
            med = np.stack([ndimage.median_filter(rgb[..., c], size=size) for c in range(3)], -1)
            med = ndimage.gaussian_filter(med, (0.7, 0.7, 0))
            a[..., :3] = w * med + (1 - w) * rgb
            Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
            print(f"{name}/{src.name}: frame smoothed ({gold.mean():.1%} gold)")


# Lens rings over a panel whose HD sidecar is a hole / black under the lens
# (#221-#223): the ring's own corner wood (outside the gold, r > ~49.5 1x
# px) never matched the panel - not even at 1x - so the lens showed as a
# square. Rebuild the corners from the panel's HD wood, mirrored across the
# square's nearest edge (continuous at the edge; the two mirrors blend
# along the diagonal). name -> (panel art, lens x, y in the panel). Charedit
# (Char_PCC) restores its panel in the engine instead.
# The panels' hole edge also fades 1-2 HD px outside the square (a thin dark
# line around the lens): that band is made opaque with the wood just beyond
# it, in every listed panel. (The gold beside the square's sides is the
# panel's - the ring art's circle is wider than its square.) name ->
# ([panels, first = wood source], x, y).
LENS_CORNERS = {
    "PCWinCvr": (["LogBooks_Side"], 25, 24),
    "Lns_Map": (["MapMain"], 25, 24),
    "Lns_Papr": (["PDoll"], 11, 9),
    "Lns_Schm": (["Schematic_Base"], 50, 26),
}
# Ring arts that don't exist in vanilla: a copy of another ring (art and its
# hole-smoothed sidecar) for one panel, where the vanilla ring serves several
# panels with different wood. Lns_Schm = the schematic screen's Lns_Bart
# (the book screen keeps Lns_Bart), Lns_Map = the town map's PCWinCvr (the
# logbook keeps PCWinCvr): interface art 4001 / 4002 in name.c.
LENS_ALIAS = {"Lns_Schm": "Lns_Bart", "Lns_Map": "PCWinCvr"}
LENS_CORNER_R = 49.5  # 1x px: the gold ring's outer edge
LENS_EDGE_BAND = 3  # HD px outside the square rebuilt in the panels


def _lens_panel_edge(panel: str, lx: int, ly: int, n: int) -> None:
    """See LENS_EDGE_BAND. Rebuilt from a backup every run."""
    out_dir = hd_out_dir(f"art/interface/{panel}.ART")
    backup = config.WORK_DIR / "_lens_corner_originals" / panel
    if not backup.exists():
        shutil.copytree(out_dir, backup)
    a = np.asarray(Image.open(backup / "r0_f0.png").convert("RGBA")).copy()
    ph, pw = a.shape[:2]
    X0, Y0, X1, Y1 = lx * HD_SCALE, ly * HD_SCALE, lx * HD_SCALE + n, ly * HD_SCALE + n
    g = LENS_EDGE_BAND
    for k in range(1, g + 1):
        if X0 - k >= 0 and X0 - g - 1 >= 0:
            a[Y0:Y1, X0 - k] = a[Y0:Y1, X0 - g - 1]
        if X1 - 1 + k < pw and X1 + g < pw:
            a[Y0:Y1, X1 - 1 + k] = a[Y0:Y1, X1 + g]
        if Y0 - k >= 0 and Y0 - g - 1 >= 0:
            a[Y0 - k, X0 - g:X1 + g] = a[Y0 - g - 1, X0 - g:X1 + g]
        if Y1 - 1 + k < ph and Y1 + g < ph:
            a[Y1 - 1 + k, X0 - g:X1 + g] = a[Y1 + g, X0 - g:X1 + g]
    Image.fromarray(a, "RGBA").save(out_dir / "r0_f0.png")


def cmd_hd_lens_corners(only: str | None = None) -> None:
    S = HD_SCALE
    for name, (panels, lx, ly) in LENS_CORNERS.items():
        if only is not None and only.lower() not in name.lower():
            continue
        rel = f"art/interface/{name}.ART"
        out_dir = hd_out_dir(rel)
        backup = config.WORK_DIR / "_lens_corner_originals" / name
        if name in LENS_ALIAS:
            # the engine's copy of the source art (data/ is a file repository);
            # the ring source is the alias target's current sidecar
            src_rel = f"art/interface/{LENS_ALIAS[name]}.ART"
            dest = config.HD_OVERLAY_DIR.parent / "data" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(find_source_art(src_rel), dest)
            backup = config.WORK_DIR / "_lens_corner_originals" / LENS_ALIAS[name]
            if not backup.exists():  # not given corners itself
                backup = hd_out_dir(src_rel)
            out_dir.mkdir(parents=True, exist_ok=True)
        elif not backup.exists():
            shutil.copytree(out_dir, backup)
        n = Image.open(backup / "r0_f0.png").width
        for p in panels:
            _lens_panel_edge(p, lx, ly, n)
        panel = panels[0]
        pan = np.asarray(Image.open(hd_out_dir(f"art/interface/{panel}.ART") / "r0_f0.png").convert("RGB"), dtype=np.float32)
        ph, pw = pan.shape[:2]
        for src in sorted(backup.glob("r*_f*.png")):
            ring = np.asarray(Image.open(src).convert("RGBA"), dtype=np.float32).copy()
            n = ring.shape[0]
            X0, Y0 = lx * S, ly * S
            yy, xx = np.mgrid[:n, :n].astype(np.float32)
            # distances to the four edges (HD px, pixel centres)
            d = {"l": xx + 0.5, "r": n - xx - 0.5, "t": yy + 0.5, "b": n - yy - 0.5}
            # mirror sample coordinates in the panel for each edge
            samples = {
                "l": (Y0 + yy, X0 - 1 - xx),
                "r": (Y0 + yy, X0 + n + (n - 1 - xx)),
                "t": (Y0 - 1 - yy, X0 + xx),
                "b": (Y0 + n + (n - 1 - yy), X0 + xx),
            }
            acc = np.zeros((n, n, 3), np.float32)
            wsum = np.zeros((n, n), np.float32)
            for k, (sy, sx) in samples.items():
                ok = (sy >= 0) & (sy < ph) & (sx >= 0) & (sx < pw)
                w = np.exp(-d[k] / (2.0 * S)) * ok
                col = pan[np.clip(sy, 0, ph - 1).astype(int), np.clip(sx, 0, pw - 1).astype(int)]
                acc += col * w[..., None]
                wsum += w
            fill = acc / np.maximum(wsum, 1e-6)[..., None]
            r = np.hypot(xx + 0.5 - n / 2, yy + 0.5 - n / 2)
            m = np.clip((r - LENS_CORNER_R * S) / 3.0 + 0.5, 0, 1) * (wsum > 0)
            ring[..., :3] = ring[..., :3] * (1 - m[..., None]) + fill * m[..., None]
            ring[..., 3] = np.maximum(ring[..., 3], m * 255)
            Image.fromarray(np.clip(ring + 0.5, 0, 255).astype(np.uint8), "RGBA").save(out_dir / src.name)
            print(f"{name}/{src.name}: corners from {panel} ({int((m > 0.5).sum())} HD px)")


def cmd_hd_lens_rings(only: str | None = None, out_root: Path | None = None) -> None:
    """Smooth the lens ring sidecars' hole edge (see LENS_RINGS). Originals go
    to work/_lens_originals/ once; always rebuilt from those."""
    for name, pct in LENS_RINGS.items():
        if only is not None and only.lower() not in name.lower():
            continue
        rel = f"art/interface/{name}.ART"
        out_dir = hd_out_dir(rel)
        backup = config.WORK_DIR / "_lens_originals" / name
        if not backup.exists():
            shutil.copytree(out_dir, backup)
        dest = out_dir if out_root is None else out_root / name
        dest.mkdir(parents=True, exist_ok=True)
        for src in sorted(backup.glob("r*_f*.png")):
            rgba = np.asarray(Image.open(src).convert("RGBA")).copy()
            alpha, r_in = lens_ring_alpha(rgba, pct)
            rgba[..., 3] = alpha
            Image.fromarray(rgba, "RGBA").save(dest / src.name)
            print(f"  {name}/{src.name}: hole radius {r_in:.1f} px")


def cmd_hd_fonts(only: str | None = None, ttf_override: tuple | None = None,
                 out_root: Path | None = None) -> None:
    """Glyph sidecars for the vanilla bitmap fonts, rendered from TTFs
    (FONT_TTF) instead of upscaled: white with the coverage as alpha (the
    engine tints them with the font colour, like the vanilla ALPHA_SRC +
    COLOR_CONST glyph blit). Each keeps its vanilla frame's cell exactly
    (4x size, same baseline, ink centred where the vanilla ink was), so text
    layout - advances, wrapping, centring - is unchanged. Size: the TTF's
    cap height matches the vanilla 'H'; a per-font x scale matches the
    vanilla letter widths; then each glyph is fitted to its own vanilla ink
    box (_fit_glyph), so digits and odd letters sit where vanilla's did.
    Glyphs the TTF lacks fall back to a smooth upscale of the vanilla
    coverage. FONT_THIN erodes a font's strokes (single-weight TTFs that are
    too heavy). Trials: ttf_override = (ttf, weight[, thin]) replaces the
    matched fonts' FONT_TTF entry, out_root puts the sidecars under
    out_root/<art path> instead of the game's hd folder."""
    from PIL import ImageDraw, ImageFont

    s = HD_SCALE
    ss = 4  # supersampling of the TTF render
    for rel, (ttf, weight) in FONT_TTF.items():
        if only is not None and only.lower() not in rel.lower():
            continue
        thin = FONT_THIN.get(rel, 0.0)
        cap = FONT_CAP.get(rel, 1.0)
        scale = FONT_SCALE.get(rel, 1.0)
        free = rel in FONT_FREE_FIT
        uniform = rel in FONT_UNIFORM
        if ttf_override is not None:
            ttf, weight, thin, cap, free = (tuple(ttf_override) + (0.0, 1.0, False)[len(ttf_override) - 2:])[:5]
        src_rel = FONT_ALIAS.get(rel, rel)
        if src_rel != rel:
            # the engine's copy of the source art (data/ is a file repository)
            dest = config.HD_OVERLAY_DIR.parent / "data" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(find_source_art(src_rel), dest)
        wd = cmd_unpack(src_rel, quiet=True)
        basename = Path(src_rel).name.rsplit(".", 1)[0]
        frames = _font_frames(wd, basename)
        by_char = {}
        for f in frames:
            ch = _glyph_char(f["frame"])
            if ch is not None:
                by_char[ch] = f
        caps = rel in FONT_CAPS_ONLY

        def load(size: int):
            font = ImageFont.truetype(str(ensure_font(ttf)), size)
            if weight is not None:
                font.set_variation_by_axes([weight])
            return font

        def render(font, ch: str):
            """(coverage, baseline row, pen column) with the glyph's ink
            box tight."""
            box = font.getbbox(ch, anchor="ls")
            img = Image.new("L", (max(1, box[2] - box[0] + 8), max(1, box[3] - box[1] + 8)), 0)
            ImageDraw.Draw(img).text((4 - box[0], 4 - box[1]), ch, font=font, fill=255, anchor="ls")
            return np.asarray(img, dtype=np.float32) / 255.0, 4 - box[1], 4 - box[0]

        font = None
        size = 0
        ratios = []
        advances = _font_advances(wd / (basename + ".ini")) if free else {}
        if ttf is not None:
            ref_box = _ink_box(by_char["H"]["cov"])
            cap_v = (ref_box[3] - ref_box[1]) * s * ss * cap
            baseline = ref_box[3]  # vanilla row just under the 'H'
            size = int(cap_v)
            for _ in range(3):  # cap height -> point size
                a, _, _ = render(load(size), "H")
                b = _ink_box(a, 0.5)
                size = max(4, int(round(size * cap_v / max(1, b[3] - b[1]))))
            font = load(size)
            missing, _, _ = render(font, "\U0010FFFD")

        # Per-font x scale from the letters' ink widths.
        for ch in "abcdeghknopqrsuvxyzABCDEGHKNOPRSUVXYZ" if font is not None else "":
            f = by_char.get(ch)
            vb = _ink_box(f["cov"]) if f is not None else None
            if vb is None:
                continue
            if free:  # match advances, so joined script letters meet
                adv = advances.get(f["frame"], 0)
                length = font.getlength(ch)
                if adv > 0 and length > 0:
                    ratios.append(adv * s * ss / length)
                continue
            a, _, _ = render(font, ch.upper() if caps else ch)
            tb = _ink_box(a, 0.5)
            if tb is not None and tb[2] > tb[0]:
                ratios.append((vb[2] - vb[0]) * s * ss / (tb[2] - tb[0]))
        x_scale = float(np.clip(np.median(ratios), 0.6, 1.35)) if ratios else 1.0
        x_scale = FONT_XSCALE.get(rel, x_scale)

        out_dir = hd_out_dir(rel) if out_root is None else out_root / Path(rel).with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        fallback = 0
        picto = 0
        for f in frames:
            W, H = f["w"] * s, f["h"] * s
            cell = np.zeros((H * ss, W * ss), np.float32)
            ch = _glyph_char(f["frame"])
            vb = _ink_box(f["cov"])
            drawn = False
            scaled = False
            if vb is not None and _is_picto(rel, ch, f["w"]):
                tmp = config.WORK_DIR / "_font_picto"
                tmp.mkdir(parents=True, exist_ok=True)
                Image.fromarray((f["cov"] * 255).astype(np.uint8), "L").convert("RGB").save(tmp / "in.png")
                run_esrgan(tmp / "in.png", tmp / "out.png", config.REALESRGAN_MODEL)
                up = Image.open(tmp / "out.png").convert("L").resize((W * ss, H * ss), Image.LANCZOS)
                cell = np.asarray(up, dtype=np.float32) / 255.0
                cell = np.where(cell < 0.04, 0.0, cell)
                drawn = True
                picto += 1
            elif font is not None and ch is not None and vb is not None and ch.strip():
                a, base, pen = render(font, ch.upper() if caps else ch)
                if a.shape != missing.shape or not np.array_equal(a, missing):
                    gx = x_scale
                    if free:  # this glyph's pen advance -> its vanilla advance
                        length = font.getlength(ch.upper() if caps else ch)
                        if advances.get(f["frame"], 0) > 0 and length > 0:
                            gx = float(np.clip(advances[f["frame"]] * s * ss / length, x_scale * 0.75, x_scale * 1.33))
                    # Uniform + FONT_SCALE: fit into a cell padded above and
                    # below, scale, and only then squash what still
                    # overflows - squashing first flattened descenders
                    # ('y' read as 'v', #225).
                    pad = H * ss if uniform and scale != 1.0 else 0
                    bl = baseline * s * ss
                    g = _fit_glyph(a, base, vb, ch, gx, bl + pad, W * ss, H * ss + 2 * pad, s * ss,
                                   pen if free else None, uniform, pad)
                    if g is not None:
                        g, x0, y0 = g
                        canvas = np.zeros((H * ss + 2 * pad, W * ss), np.float32)
                        ys0, ys1 = max(0, y0), min(canvas.shape[0], y0 + g.shape[0])
                        if ys1 > ys0:
                            canvas[ys0:ys1, x0:x0 + g.shape[1]] = g[ys0 - y0:ys1 - y0]
                            if pad:
                                canvas = _scale_glyph(canvas, scale, bl + pad)
                                cell = _squash_rows(canvas, pad, H * ss, bl + pad)
                                scaled = True
                            else:
                                cell = canvas
                            drawn = True
            if not drawn and vb is not None:
                up = Image.fromarray((f["cov"] * 255).astype(np.uint8), "L").resize((W * ss, H * ss), Image.LANCZOS)
                cell = np.asarray(up, dtype=np.float32) / 255.0
                fallback += 1
            if thin != 0 and drawn and not _is_picto(rel, ch, f["w"]):
                cell = _thin(cell, thin * ss, 1.5 * ss)
            if scale != 1.0 and drawn and not scaled and not _is_picto(rel, ch, f["w"]):
                cell = _scale_glyph(cell, scale, baseline * s * ss)
            alpha = Image.fromarray((np.clip(cell, 0, 1) * 255).astype(np.uint8), "L").resize((W, H), Image.BOX)
            rgba = Image.new("RGBA", (W, H), (255, 255, 255, 0))
            rgba.putalpha(alpha)
            rgba.save(out_dir / f"r0_f{f['frame']}.png")
        print(f"{rel}: {len(frames)} glyphs from {ttf} at {size / ss:.1f}px x{x_scale:.2f}, {picto} pictures, {fallback} vanilla fallbacks")


# Font arts outside MAIN_FONT_ARTS also drawn as a face (cmd_hd_font_faces):
# their FONT_TTF face at its own advances, sized down to the vanilla line,
# never narrowed. Per-cell fitting clamps each letter to its vanilla cell
# (_fit_glyph's gw = min(cw, ...)), which still squeezed wide letters.
# Round 8 pass 11: CharStatsFont (#65, "just Outfit 400" - its alias source
# is blackletter morph15font), CasablancaAntique30Font (#71, the schematic
# name header in Special Elite), SchemDescFont (#31, its body text).
FACE_EXTRA_ARTS = [
    "art/interface/CharStatsFont.ART",
    "art/interface/CasablancaAntique30Font.ART",
    # pass 11 (#31): the body text too after all - per-cell fitting left gaps
    # after wide letters ("w onders", "fro m") next to the face header
    "art/interface/SchemDescFont.ART",
    # pass 12 (#46): the Save/Load list's bold names - per-cell fitting into
    # Flare12Font's narrow cells squeezed Outfit 500's capitals together
    "art/interface/SaveLoadListFont.ART",
]

# Faces whose '|' is drawn as the face's 'I' (cap height, stem weight), the
# text-edit cursor (mainmenu_ui.c sub_544100): Outfit's own bar is thinner,
# taller than the capitals and hangs below the baseline. Pass 12 feedback
# (#47, "fix the cursor to match this font").
FACE_CURSOR_BAR = {
    "art/interface/SaveLoadListFont.ART",
}

# Extra size factor on a face's matched size (cmd_hd_font_faces), baseline
# kept. Round 8 pass 11 #18: CharStatsFont's Level/Race/... block read a
# little large next to the rest of the character sheet.
FACE_SIZE_SCALE = {
    "art/interface/CharStatsFont.ART": 0.9,
}

# Text the face size is matched on (FONT_FACE): typical UI / dialogue text.
FACE_SAMPLE = ("Having been given the strange ring by Preston Radcliffe, you are currently "
               "attempting to find its owner. Shall we trade? Could you heal me? The Discipline "
               "of the Smithy has been revolutionized by technology!")


def cmd_hd_font_faces(only: str | None = None) -> None:
    """MAIN_FONT (and FACE_EXTRA_ARTS' FONT_TTF faces) drawn as itself at
    HD (round 8 pass 7, #229-#236): no
    per-cell fitting - its own glyph shapes, advances and kerning. The
    engine (font.c, tig_font_hd_face) lays each line out with these HD
    advances from the vanilla line's start (or centre), so only the 1x
    layout (wrapping, centring) keeps the vanilla metrics. Size: the one
    whose FACE_SAMPLE width matches the vanilla font's, so lines wrap about
    where the HD text ends; never taller than the vanilla capitals.
    Output: hd/<art>/face.txt ("g frame advance ox oy": HD px, image
    top-left from the pen / the cell top; "k frame frame adjust": kerning)
    and hd/<art>/face/f<frame>.png (white, coverage as alpha)."""
    import uharfbuzz as hb
    from PIL import ImageDraw, ImageFont

    s = HD_SCALE
    hb_font = None
    upem = 1

    def shape(text: str) -> list:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf, {"liga": False, "clig": False})
        return buf.glyph_positions

    def width_em(text: str) -> float:
        return sum(p.x_advance for p in shape(text)) / upem

    for rel in MAIN_FONT_ARTS + FACE_EXTRA_ARTS:
        if only is not None and only.lower() not in rel.lower():
            continue
        ttf, weight = FONT_TTF[rel]
        path = ensure_font(ttf)
        hb_face = hb.Face(hb.Blob.from_file_path(str(path)))
        hb_font = hb.Font(hb_face)
        if weight is not None:
            hb_font.set_variations({"wght": weight})
        upem = hb_face.upem
        cap_ext = hb_font.get_glyph_extents(hb_font.get_nominal_glyph(ord("H")))
        cap_em = -cap_ext.height / upem

        # an alias (FONT_ALIAS) has its source art's cells and advances
        src_rel = FONT_ALIAS.get(rel, rel)
        wd = cmd_unpack(src_rel, quiet=True)
        basename = Path(src_rel).name.rsplit(".", 1)[0]
        frames = _font_frames(wd, basename)
        advances = _font_advances(wd / (basename + ".ini"))
        by_frame = {f["frame"]: f for f in frames}
        ref = _ink_box(by_frame[ord("H") - 31]["cov"])
        vw = sum(advances.get(ord(c) - 31, 0) for c in FACE_SAMPLE)
        size = min(vw / width_em(FACE_SAMPLE), (ref[3] - ref[1]) / cap_em)  # vanilla px
        size *= FACE_SIZE_SCALE.get(rel, 1.0)
        px = size * s  # HD px per em
        baseline = ref[3] * s  # HD row of the baseline in the cell

        font = ImageFont.truetype(str(path), px)
        if weight is not None:
            font.set_variation_by_axes([weight])

        out_dir = hd_out_dir(rel)
        face_dir = out_dir / "face"
        if face_dir.exists():
            shutil.rmtree(face_dir)
        face_dir.mkdir(parents=True)
        lines = [f"# {ttf} wght {weight} {px:.2f} HD px/em"]
        chars = {}
        fallback = 0
        for f in frames:
            ch = _glyph_char(f["frame"])
            if ch is None or ch in "\t\n":
                continue
            gid = hb_font.get_nominal_glyph(ord(ch))
            if gid is None or gid == 0:
                # not in the face: the vanilla glyph, smoothly upscaled
                cov = f["cov"]
                if _ink_box(cov) is not None:
                    up = Image.fromarray((cov * 255).astype(np.uint8), "L").resize((f["w"] * s, f["h"] * s), Image.LANCZOS)
                    rgba = Image.new("RGBA", up.size, (255, 255, 255, 0))
                    rgba.putalpha(up)
                    rgba.save(face_dir / f"f{f['frame']}.png")
                    lines.append(f"g {f['frame']} {advances.get(f['frame'], 0) * s} 0 0")
                else:
                    lines.append(f"g {f['frame']} {advances.get(f['frame'], 0) * s} 0 0 -")
                fallback += 1
                continue
            chars[ch] = f["frame"]
            adv = hb_font.get_glyph_h_advance(gid) * px / upem
            if not ch.strip():
                lines.append(f"g {f['frame']} {adv:.3f} 0 0 -")
                continue
            pad = int(px)
            img = Image.new("L", (int(px * 3) + 2 * pad, int(px * 2.5) + 2 * pad), 0)
            pen_x, base_y = pad, pad + int(px * 1.5)
            drawn = "I" if ch == "|" and rel in FACE_CURSOR_BAR else ch
            ImageDraw.Draw(img).text((pen_x, base_y), drawn, font=font, fill=255, anchor="ls")
            box = img.getbbox()
            if box is None:
                lines.append(f"g {f['frame']} {adv:.3f} 0 0 -")
                continue
            crop = img.crop(box)
            rgba = Image.new("RGBA", crop.size, (255, 255, 255, 0))
            rgba.putalpha(crop)
            rgba.save(face_dir / f"f{f['frame']}.png")
            lines.append(f"g {f['frame']} {adv:.3f} {box[0] - pen_x} {box[1] - base_y + baseline}")

        # kerning: pair advance minus the two glyphs' own
        pairs = 0
        items = [(c, fr) for c, fr in chars.items() if 32 < ord(c) < 127 or c.isalpha()]
        single = {c: width_em(c) for c, _ in items}
        for a, fa in items:
            for b, fb in items:
                adj = (width_em(a + b) - single[a] - single[b]) * px
                if abs(adj) >= 0.25:
                    lines.append(f"k {fa} {fb} {adj:.2f}")
                    pairs += 1
        (out_dir / "face.txt").write_text("\n".join(lines) + "\n")
        print(f"{rel}: {ttf} at {size:.1f}px ({px:.1f} HD), {len(chars)} glyphs, {pairs} kerning pairs, {fallback} vanilla fallbacks")


def cmd_hd_compose(only: str | None = None, model: str | None = None, dry_run: bool = False) -> None:
    """Rebuild every captured interface frame's sidecar from an in-context
    upscale (see the block comment above). The first run keeps each frame's
    previous sidecar in work/_compose_originals/; frames without a usable
    capture are left alone."""
    placements = read_capture_manifest()
    esrgan_model = model or config.REALESRGAN_MODEL
    matte_pieces = {piece.replace("\\", "/").lower() for piece, _, _ in BACKGROUND_MATTE_PIECES}
    matte_pieces |= {p.lower() for p in SCROLL_THUMB_PIECES}  # built as one stack, hd-scroll-thumb
    stage = config.WORK_DIR / "_compose"
    if stage.exists():
        shutil.rmtree(stage)
    stage_in = stage / "in"
    stage_in.mkdir(parents=True)

    frames_by_art: dict[str, dict[tuple[int, int], Path]] = {}
    jobs = []
    skipped = 0
    for (rel, rot, frame), cands in sorted(placements.items()):
        if only is not None and only.lower() not in rel.lower():
            continue
        if rel.lower() in matte_pieces:
            continue
        if rel not in frames_by_art:
            wd = cmd_unpack(rel, quiet=True)
            basename = Path(rel).name.rsplit(".", 1)[0]
            num_frames, animated = read_ini_frame_count(wd / (basename + ".ini"))
            frames_by_art[rel] = {(r, f): p for r, f, p in hd_frame_bmps(wd, basename, num_frames, animated)}
        bmp = frames_by_art[rel].get((rot, frame))
        if bmp is None:
            skipped += 1
            continue

        w, h = read_bmp_dims(bmp)
        h = abs(h)
        indices = np.asarray(read_bmp_indices(bmp), dtype=np.uint8).reshape(h, w)
        palette = np.asarray(read_bmp_palette(bmp), dtype=np.uint8).reshape(256, 3)
        rgb = palette[indices]
        opaque = indices != 0

        # The placement with the most context around the frame.
        best = None
        for c in cands:
            sx, sy, sw, sh = c["src"]
            dx, dy, dw, dh = c["dst"]
            cx, cy, cw, ch = c["cap"]
            if (c["flags"] & ~(BLT_FLIP_X | BLT_FLIP_Y)) != 0 or (sx, sy, sw, sh) != (0, 0, w, h) or (dw, dh) != (w, h):
                continue
            if dx < cx or dy < cy or dx + w > cx + cw or dy + h > cy + ch or not c["bmp"].is_file():
                continue
            ox, oy = dx - cx, dy - cy
            sides = (ox, oy, cw - ox - w, ch - oy - h)
            # No context on any side (a full-window background): nothing to
            # compose against - its own upscale already is the whole screen.
            if max(sides) == 0:
                continue
            score = (min(sides), sum(sides))
            if best is None or score > best[0]:
                best = (score, c, ox, oy)
        if best is None:
            skipped += 1
            continue

        _, c, ox, oy = best
        cx, cy, cw, ch = c["cap"]
        under = np.asarray(Image.open(c["bmp"]).convert("RGB"))
        if under.shape[:2] != (ch, cw):
            skipped += 1
            continue
        # Transparent window: its colour key means "nothing drawn here" -
        # replace with the nearest real colour so it never bleeds in.
        if c["key"] is not None:
            hole = np.all(under == np.array(c["key"], dtype=np.uint8), axis=2)
            if hole.all():
                under = np.zeros_like(under)
            elif hole.any():
                under = np.clip(inpaint_colorkey(under, hole) + 0.5, 0, 255).astype(np.uint8)

        fr, fo = rgb, opaque
        if c["flags"] & BLT_FLIP_X:
            fr, fo = fr[:, ::-1], fo[:, ::-1]
        if c["flags"] & BLT_FLIP_Y:
            fr, fo = fr[::-1], fo[::-1]

        comp = under.copy()
        comp[oy:oy + h, ox:ox + w][fo] = fr[fo]
        mask = np.zeros(under.shape[:2], dtype=np.uint8)
        mask[oy:oy + h, ox:ox + w][fo] = 255

        group = "double" if max(w, h) <= COMPOSE_DOUBLE_MAX else "single"
        idx = len(jobs)
        (stage_in / group).mkdir(exist_ok=True)
        Image.fromarray(comp, "RGB").save(stage_in / group / f"{idx:05d}_c.png")
        Image.fromarray(mask, "L").convert("RGB").save(stage_in / group / f"{idx:05d}_m.png")
        jobs.append(dict(rel=rel, rot=rot, frame=frame, w=w, h=h, ox=ox, oy=oy, flags=c["flags"],
                         idx=idx, group=group, comp_size=comp.shape[:2]))

    if not jobs or dry_run:
        arts = len({j["rel"] for j in jobs})
        doubles = sum(1 for j in jobs if j["group"] == "double")
        print(f"hd-compose{' (dry run)' if dry_run else ''}: {len(jobs)} frame(s) of {arts} art(s) usable "
              f"({doubles} small enough for the 16x pass), {skipped} captured frame(s) without a usable placement")
        return

    print(f"hd-compose: upscaling {len(jobs)} composite(s) + masks with {esrgan_model}...")
    for group in ("single", "double"):
        if (stage_in / group).is_dir():
            run_esrgan_batch(stage_in / group, stage / f"{group}_x4", esrgan_model)
    if (stage_in / "double").is_dir():
        print("hd-compose: second pass (16x) for small frames...")
        run_esrgan_batch(stage / "double_x4", stage / "double_x16", esrgan_model)

    backup_root = config.WORK_DIR / "_compose_originals"
    s = HD_SCALE
    lo, hi = COMPOSE_MASK_EDGE
    for j in jobs:
        ch, cw = j["comp_size"]
        size4 = (cw * s, ch * s)

        def load(kind: str) -> Image.Image:
            name = f"{j['idx']:05d}_{kind}.png"
            if j["group"] == "double":
                im = load_and_validate(stage / "double_x16" / name, (cw * s * s, ch * s * s), "hd-compose x16")
                return im.resize(size4, Image.LANCZOS)
            return load_and_validate(stage / "single_x4" / name, size4, "hd-compose")

        hd = np.asarray(load("c").convert("RGB"))
        m4 = np.asarray(load("m").convert("L"), dtype=np.float32) / 255.0
        y0, y1 = j["oy"] * s, (j["oy"] + j["h"]) * s
        x0, x1 = j["ox"] * s, (j["ox"] + j["w"]) * s
        t = np.clip((m4[y0:y1, x0:x1] - lo) / (hi - lo), 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)
        rgba = np.dstack([hd[y0:y1, x0:x1], np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)])
        # Back to the frame's own orientation (the capture was drawn flipped).
        if j["flags"] & BLT_FLIP_X:
            rgba = rgba[:, ::-1]
        if j["flags"] & BLT_FLIP_Y:
            rgba = rgba[::-1]

        out_dir = hd_out_dir(j["rel"])
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"r{j['rot']}_f{j['frame']}.png"
        backup = backup_root / hd_out_dir(j["rel"]).relative_to(config.HD_OVERLAY_DIR) / dest.name
        if dest.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dest, backup)
        Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(dest)
        # Its alpha is already smooth (the upscaled mask); a stale
        # hd-smooth-alpha backup would otherwise bring the old frame back
        # on that command's next run.
        alpha_backup = alpha_backup_path(dest)
        if alpha_backup.exists():
            alpha_backup.unlink()
    print(f"hd-compose: wrote {len(jobs)} sidecar(s), skipped {skipped}; originals in {backup_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_overlay = sub.add_parser("hd-overlay", help="Package a hd-produced frame-0 PNG as the loose BMP the main-menu HD spike loads directly")
    p_overlay.add_argument("rel_path", help="Path of the .ART relative to a dat root, e.g. art/interface/MainMenuBack.ART")

    p_slides = sub.add_parser("hd-slides", help="Upscale the story-slide BMPs (death/chapter/credits) to hd/<name>_hd.bmp")
    p_slides.add_argument("--force", action="store_true", help="Regenerate even if the hd/ BMP already exists")

    p_splash = sub.add_parser("hd-splash", help="Upscale the map-load splash BMPs to hd/<name>_hd.bmp")
    p_splash.add_argument("--force", action="store_true", help="Regenerate even if the hd/ BMP already exists")

    p_portraits = sub.add_parser("hd-portraits", help="Upscale character portrait BMPs to hd/portrait/<name>.bmp")
    p_portraits.add_argument("--force", action="store_true", help="Regenerate even if the hd/ BMP already exists")

    p_movies = sub.add_parser("hd-movies", help="Upscale the intro/logo Bink videos frame-by-frame, re-encode to data/movies/<name>.bik")
    p_movies.add_argument("--force", action="store_true", help="Regenerate even if the output already exists")

    p_townmap = sub.add_parser("hd-townmap", help="Composite each town's automap tiles, upscale once, slice back to hd/townmap/<name>/")
    p_townmap.add_argument("name", nargs="?", default=None, help="One town's folder name (e.g. Ashbury); omit to process every town")
    p_townmap.add_argument("--model", default=None, help="Force one ncnn model (default: config.REALESRGAN_MODEL)")
    p_townmap.add_argument("--force", action="store_true", help="Regenerate even if this town's hd/ output already exists")

    p_worldmap = sub.add_parser("hd-worldmap", help="Composite the overworld's SmallMapChunks grid + Map_Zoomed, upscale, slice back to hd/WorldMap/")
    p_worldmap.add_argument("--model", default=None, help="Force one ncnn model (default: config.REALESRGAN_MODEL)")
    p_worldmap.add_argument("--force", action="store_true", help="Regenerate even if hd/WorldMap/ output already exists")

    p_tile_selfwrap = sub.add_parser("hd-tile-selfwrap", help="Reprocess ground tiles with a 3x3 self-wrap composite (fixes independent-upscale seam risk); ALWAYS overwrites hd/art/tile/")
    p_tile_selfwrap.add_argument("--model", default=None, help="Force one ncnn model (default: config.REALESRGAN_MODEL)")

    p_smooth = sub.add_parser("hd-smooth-alpha", help="De-staircase the alpha edges of existing hd/art/<category>/ sidecars (rings, round buttons); originals kept in work/_alpha_originals/")
    p_smooth.add_argument("category", nargs="?", default="interface")
    p_smooth.add_argument("--workers", type=int, default=8)

    p_compose = sub.add_parser("hd-compose", help="Rebuild captured interface sidecars by compose -> upscale -> decompose over their real underlays (game's hd_capture/, see arcanum-ce tig_window_hd_capture)")
    p_compose.add_argument("--only", default=None, help="Only arts whose path contains this text")
    p_compose.add_argument("--model", default=None, help="Force one ncnn model (default: config.REALESRGAN_MODEL)")
    p_compose.add_argument("--dry-run", action="store_true", help="Only report which captured frames are usable")

    p_buttons = sub.add_parser("hd-buttons", help="Re-upscale interface buttons frame by frame with the exact checker de-dither (hover glows); originals kept in work/_button_originals/")
    p_buttons.add_argument("names", nargs="+", help="interface art names (Skills_Button) or rel paths")
    p_buttons.add_argument("--model", default=None)

    p_fonts = sub.add_parser("hd-fonts", help="Glyph sidecars for the vanilla bitmap fonts, rendered from fonts/vanilla/ TTFs in each glyph's own cell")
    p_fonts.add_argument("--only", default=None, help="Only fonts whose path contains this text")

    p_stext = sub.add_parser("hd-splash-text", help="Redo the splashes' Loading Arcanum lettering (SPLASH_TEXT)")
    p_stext.add_argument("--only", default=None)
    p_faces = sub.add_parser("hd-font-faces", help="MAIN_FONT at HD with its own advances and kerning (hd/<art>/face.txt + face/)")
    p_faces.add_argument("--only", default=None)

    p_outer = sub.add_parser("hd-outer-smooth",help="Smooth serrated outer silhouettes (OUTER_SMOOTH)")
    p_outer.add_argument("--only", default=None)
    p_frame = sub.add_parser("hd-frame-smooth", help="Round off wobbly gold panel frames (FRAME_SMOOTH)")
    p_frame.add_argument("--only", default=None)
    p_cvr = sub.add_parser("hd-cvr-mask", help="Paperdoll slot silhouettes: keep only the silhouette (CVR_SLOTS)")
    p_cvr.add_argument("--only", default=None)
    p_disc = sub.add_parser("hd-disc-mask", help="Cut round buttons' sidecars to their disc (DISC_MASKS)")
    p_disc.add_argument("--only", default=None)
    p_lensc = sub.add_parser("hd-lens-context", help="Inventory/barter/loot lens rings upscaled in context with their panel (LENS_CONTEXT)")
    p_lensc.add_argument("--only", default=None)
    p_htft = sub.add_parser("hd-htft-knob", help="HP/fatigue -/+ sidecars from the character sheet's own knobs (HTFT_KNOBS)")
    p_htft.add_argument("--only", default=None)
    p_cyc = sub.add_parser("hd-cycle-arrows", help="Character creation arrow buttons: clean disc + vector arrows (CYCLE_ARROWS)")
    p_cyc.add_argument("--only", default=None)
    sub.add_parser("hd-schem-base-edge", help="Schematic_Base: opaque ring around the drawing's hole (SCHEM_BASE_HOLE)")
    sub.add_parser("hd-nav-pill-rim", help="Worldmap bottom plate pills: even rim around the groove (NAV_PILLS)")
    sub.add_parser("hd-skill-gauge", help="Skills_Window gauges: full glass tube, crisp 1..5 (SKILL_GAUGE_*)")
    sub.add_parser("hd-saveload-chain", help="Save/Load list's scroll track: the Loot panel's chain painted in (SAVELOAD_CHAIN_*)")
    sub.add_parser("hd-scroll-chains", help="Loot/Barter panels: chain repainted between the scroll arrows, centred on them (SCROLL_CHAIN_*)")
    p_schem = sub.add_parser("hd-schem-tone", help="Match schematic drawings' edge tone to the base paper (SCHEM_TONE_*)")
    p_schem.add_argument("--only", default=None)
    p_icon = sub.add_parser("hd-icon-patch", help="Re-upscale icons painted into panels with another model (ICON_PATCHES)")
    p_icon.add_argument("--only", default=None)
    p_black = sub.add_parser("hd-black-fill", help="Near-black panel boxes / icon backgrounds -> pure black (BLACK_BOXES, BLACK_BG)")
    p_black.add_argument("--only", default=None)
    p_lcorn = sub.add_parser("hd-lens-corners", help="Lens ring corners from the panel's HD wood (LENS_CORNERS); run after hd-lens-rings")
    p_lcorn.add_argument("--only", default=None)
    p_lens = sub.add_parser("hd-lens-rings", help="Smooth the PC lens ring sidecars' hole edge (LENS_RINGS)")
    p_lens.add_argument("--only", default=None)

    p_scan = sub.add_parser("hd-scan", help="Find (and with --fix redo) hd/art sidecars written from corrupt ncnn batch output")
    p_scan.add_argument("categories", nargs="+")
    p_scan.add_argument("--workers", type=int, default=8)
    p_scan.add_argument("--fix", action="store_true")
    p_scan.add_argument("--model", default=None)

    p_requeue = sub.add_parser("hd-requeue", help="Regenerate the frames the one-model policy upscales differently (old small-frame CUGAN route, checkerboard dither)")
    p_requeue.add_argument("categories", nargs="+")
    p_requeue.add_argument("--workers", type=int, default=8)
    p_requeue.add_argument("--dry-run", action="store_true")
    p_requeue.add_argument("--model", default=None)

    p_pal = sub.add_parser("hd-palettes", help="Sidecars (r<rot>_f<frame>_p<N>.png) for arts' extra palettes 1-3")
    p_pal.add_argument("categories", nargs="+")
    p_pal.add_argument("--model", default=None)
    p_pal.add_argument("--force", action="store_true")

    sub.add_parser("hd-scroll-thumb", help="Upscale the scrollbar thumb (ScrllSlideT/M1/B) as one stack and cut it back apart (no per-row seams)")

    sub.add_parser("hd-background-matte", help="Replace the background baked into PC-lens-style pieces with the HD background crop (see BACKGROUND_MATTE_PIECES)")

    p_hd = sub.add_parser("hd", help="Emit 4x RGBA PNG sidecars under hd/art/ for one .ART (rel_path) or a whole art/ category")
    p_hd.add_argument("target", help="art/<cat>/Name.ART rel_path, or a category folder name (interface, item, ...)")
    p_hd.add_argument("--model", default=None, help="Force one ncnn model for every frame (default: size-based pick)")
    p_hd.add_argument("--limit", type=int, default=None)
    p_hd.add_argument("--force", action="store_true", help="Regenerate even if PNGs already exist")
    p_hd.add_argument("--list", type=Path, default=None, help="Text file of rel_paths (one per line, '#' comments allowed) to process instead of walking art/<target>/ - target is still used as the batch-log name")
    p_hd.add_argument("--workers", type=int, default=1, help="Run N .ART files concurrently (2-4 recommended; each spends most of its time blocked in realesrgan-ncnn-vulkan.exe, so threads overlap real work). Ignored for a single explicit .ART target.")

    args = parser.parse_args()

    if args.command == "hd":
        if args.target.lower().endswith(".art"):
            cmd_hd(args.target, model=args.model, force=True)  # explicit single file: always regenerate
        else:
            cmd_hd_batch(args.target, model=args.model, limit=args.limit, force=args.force, file_list=args.list, workers=args.workers)
        return

    if args.command == "hd-overlay":
        cmd_hd_overlay(args.rel_path)
        return

    if args.command == "hd-slides":
        cmd_hd_slides(force=args.force)
        return

    if args.command == "hd-splash":
        cmd_hd_splash(force=args.force)
        return

    if args.command == "hd-portraits":
        cmd_hd_portraits(force=args.force)
        return

    if args.command == "hd-movies":
        cmd_hd_movies(force=args.force)
        return

    if args.command == "hd-townmap":
        cmd_hd_townmap(name=args.name, force=args.force, model=args.model)
        return

    if args.command == "hd-worldmap":
        cmd_hd_worldmap(force=args.force, model=args.model)
        return

    if args.command == "hd-smooth-alpha":
        cmd_hd_smooth_alpha(args.category, workers=args.workers)
        return

    if args.command == "hd-compose":
        cmd_hd_compose(only=args.only, model=args.model, dry_run=args.dry_run)
        return

    if args.command == "hd-buttons":
        cmd_hd_buttons(args.names, model=args.model)
        return

    if args.command == "hd-fonts":
        cmd_hd_fonts(args.only)
        return
    if args.command == "hd-splash-text":
        cmd_hd_splash_text(args.only)
        return
    if args.command == "hd-font-faces":
        cmd_hd_font_faces(args.only)
        return
    if args.command == "hd-lens-corners":
        cmd_hd_lens_corners(args.only)
        return
    if args.command == "hd-lens-rings":
        cmd_hd_lens_rings(args.only)
        return
    if args.command == "hd-outer-smooth":
        cmd_hd_outer_smooth(args.only)
        return
    if args.command == "hd-frame-smooth":
        cmd_hd_frame_smooth(args.only)
        return
    if args.command == "hd-cvr-mask":
        cmd_hd_cvr_mask(args.only)
        return
    if args.command == "hd-disc-mask":
        cmd_hd_disc_mask(args.only)
        return
    if args.command == "hd-lens-context":
        cmd_hd_lens_context(args.only)
        return
    if args.command == "hd-htft-knob":
        cmd_hd_htft_knob(args.only)
        return
    if args.command == "hd-cycle-arrows":
        cmd_hd_cycle_arrows(args.only)
        return
    if args.command == "hd-schem-base-edge":
        cmd_hd_schem_base_edge()
        return
    if args.command == "hd-nav-pill-rim":
        cmd_hd_nav_pill_rim()
        return
    if args.command == "hd-skill-gauge":
        cmd_hd_skill_gauge()
        return
    if args.command == "hd-saveload-chain":
        cmd_hd_saveload_chain()
    if args.command == "hd-scroll-chains":
        cmd_hd_scroll_chains()
        return
    if args.command == "hd-schem-tone":
        cmd_hd_schem_tone(args.only)
        return
    if args.command == "hd-icon-patch":
        cmd_hd_icon_patch(args.only)
        return
    if args.command == "hd-black-fill":
        cmd_hd_black_fill(args.only)
        return
        
    if args.command == "hd-scan":
        cmd_hd_scan(args.categories, workers=args.workers, fix=args.fix, model=args.model)
        return

    if args.command == "hd-requeue":
        cmd_hd_requeue(args.categories, workers=args.workers, dry_run=args.dry_run, model=args.model)
        return

    if args.command == "hd-palettes":
        cmd_hd_palettes(args.categories, model=args.model, force=args.force)
        return

    if args.command == "hd-scroll-thumb":
        cmd_hd_scroll_thumb()
        return

    if args.command == "hd-background-matte":
        cmd_hd_background_matte()
        return

    if args.command == "hd-tile-selfwrap":
        cmd_hd_tile_selfwrap(model=args.model)
        return


if __name__ == "__main__":
    main()
