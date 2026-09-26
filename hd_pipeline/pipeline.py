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
            Image.fromarray(src, "RGB").save(src_png)
            run_esrgan(src_png, hd_png, esrgan_model)
            hd = load_and_validate(hd_png, (w * HD_SCALE, h * HD_SCALE), "hd-buttons")
            flat = src.reshape(-1, 3)
            if not np.all(flat == flat[0]) and is_blank_output(hd):
                raise RuntimeError(f"realesrgan produced blank output for {bmp}")
            mask_rgb = Image.fromarray(np.where(key, 0, 255).astype(np.uint8), "L").convert("RGB")
            rgba = hd.convert("RGB")
            rgba.putalpha(hqx.hq4x(mask_rgb).convert("L"))
            dest = out_dir / f"r{rot}_f{frame}.png"
            write_sidecar(rel, dest, rgba)
            print(f"  {bmp.name} -> {dest.relative_to(config.HD_OVERLAY_DIR)}")


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
