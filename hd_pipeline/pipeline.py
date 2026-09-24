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
def run_esrgan_batch(src_dir: Path, dest_dir: Path, model: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    args = [str(config.REALESRGAN_EXE), "-i", str(src_dir), "-o", str(dest_dir), "-s", str(HD_SCALE), "-n", model]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"realesrgan batch failed on {src_dir}:\n{result.stdout}\n{result.stderr}")


def run_realcugan_batch(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    args = [
        str(config.REALCUGAN_EXE), "-i", str(src_dir), "-o", str(dest_dir),
        "-s", str(HD_SCALE), "-n", "-1", "-m", str(config.REALCUGAN_MODEL_DIR),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"realcugan batch failed on {src_dir}:\n{result.stdout}\n{result.stderr}")


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
            Image.open(out_png).convert("RGB").save(dest, "BMP")
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
# Frames below this area always route to Real-CUGAN instead of ESRGAN (see
# cmd_hd) - too little spatial context for the photo/8k ESRGAN model on tiny
# icons.
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
    # round trips). is_small/force_cugan routing unchanged from before, just
    # deferred until both groups' single batch calls run.
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
        is_small = w * h < HD_SMALL_FRAME_PX * HD_SMALL_FRAME_PX
        route = "cugan" if (force_cugan or is_small) else "esrgan"
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
        rgba.save(dest, "PNG", optimize=True)
        if not quiet:
            model_label = "realcugan" if (route == "cugan" or pixel_soup_fallback) else esrgan_model
            print(f"  {bmp.name} -> {dest.relative_to(config.HD_OVERLAY_DIR)} ({rgba.size[0]}x{rgba.size[1]}, {model_label})")
        if progress_cb is not None:
            progress_cb(i, len(frames))

    return out_dir


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
    """
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


if __name__ == "__main__":
    main()
