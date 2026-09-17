"""
Fast-iteration HD art pipeline for arcanum-ce.

Replaces the old workflow (unpack dat -> upscale -> repack ART -> repack whole
dat with dbmaker -> rebuild arcanum-ce.exe -> relaunch) with:

    unpack -> upscale -> quantize -> repack -> deploy

`deploy` drops the rebuilt .ART into Arcanum's loose `data/` overlay
directory, which arcanum-ce already mounts with higher priority than the
.dat archives (see gamelib_load_data() / tig_file_repository_add_native()).
No dat repacking, no exe rebuild, no restart of anything except the game
itself is needed to see a change.

Usage:
    python pipeline.py run art/interface/MainMenuBack.ART
    python pipeline.py revert art/interface/MainMenuBack.ART

Each step can also be run individually: unpack, upscale, quantize, repack,
deploy. Run with -h for details.
"""

import argparse
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image

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


def cmd_unpack(rel_path: str) -> Path:
    src = find_source_art(rel_path)
    wd = work_dir_for(rel_path)
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    basename = wd / Path(rel_path).with_suffix("").name
    run_art_converter(src, basename)
    print(f"Unpacked {src} -> {wd}")
    return wd


def frame_bmps(wd: Path) -> list[Path]:
    return sorted(p for p in wd.glob("*.bmp") if "_hd" not in p.stem)


def cmd_upscale(rel_path: str, model: str | None = None) -> None:
    wd = work_dir_for(rel_path)
    bmps = frame_bmps(wd)
    if not bmps:
        raise RuntimeError(f"No frame BMPs found in {wd}; run 'unpack' first")
    model = model or config.REALESRGAN_MODEL
    for bmp in bmps:
        out_png = bmp.with_name(bmp.stem + "_hd.png")
        args = [str(config.REALESRGAN_EXE), "-i", str(bmp), "-o", str(out_png)]
        if model:
            args += ["-n", model]
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"realesrgan failed on {bmp}:\n{result.stdout}\n{result.stderr}"
            )
        print(f"Upscaled {bmp.name} -> {out_png.name}")


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


FRAME_BLOCK_RE = re.compile(
    r"(frame [^\r\n:]+:\r*\n"
    r"center_x: )(-?\d+)(\r*\n"
    r"center_y: )(-?\d+)(\r*\n"
    r"offset_x: )(-?\d+)(\r*\n"
    r"offset_y: )(-?\d+)"
)


def rescale_ini_frame_offsets(ini_path: Path, scales: list[tuple[float, float]]) -> None:
    """Multiply each frame's center_x/offset_x by its scale_x and
    center_y/offset_y by its scale_y, in the order frame blocks appear
    (which matches the order frames were written, i.e. frame_num order).
    """
    text = ini_path.read_text(newline="")
    it = iter(scales)

    def repl(m: re.Match) -> str:
        sx, sy = next(it)
        cx, cy, ox, oy = (int(m.group(i)) for i in (2, 4, 6, 8))
        return (
            f"{m.group(1)}{round(cx * sx)}"
            f"{m.group(3)}{round(cy * sy)}"
            f"{m.group(5)}{round(ox * sx)}"
            f"{m.group(7)}{round(oy * sy)}"
        )

    new_text, count = FRAME_BLOCK_RE.subn(repl, text)
    if count != len(scales):
        raise RuntimeError(f"Expected {len(scales)} frame blocks in {ini_path}, found {count}")
    ini_path.write_text(new_text, newline="")


def cmd_quantize(rel_path: str, keep_size: bool = False) -> None:
    """keep_size=True supersamples the AI-upscaled frame back down to the
    original BMP's exact pixel dimensions (LANCZOS) before quantizing, so
    the art gets denoised/sharpened detail but stays byte-identical in
    width/height to stock. For asset categories where engine code derives
    gameplay-relevant numbers from art pixel dimensions (e.g. item
    inventory-icon grid-cell sizing, item_inv_icon_size() in item.c) this is
    the only safe way to upscale without changing that derived math.
    """
    wd = work_dir_for(rel_path)
    basename = wd / Path(rel_path).with_suffix("").name
    ini_path = basename.with_suffix(".ini")

    scales: list[tuple[float, float]] = []
    for bmp in frame_bmps(wd):
        hd_png = bmp.with_name(bmp.stem + "_hd.png")
        if not hd_png.exists():
            raise RuntimeError(f"Missing {hd_png}; run 'upscale' first")

        orig_w, orig_h = read_bmp_dims(bmp)

        # LoadBMPS() only reads raw index bytes and ignores the BMP's own
        # embedded palette, but SaveBMPS() *did* write the real ART palette
        # into this original extracted BMP's color table when unpacking.
        # Reuse it so the .ini (which stores the authoritative palette text)
        # never needs to be touched.
        palette = read_bmp_palette(bmp)

        pal_img = Image.new("P", (1, 1))
        pal_img.putpalette(palette)

        upscaled = Image.open(hd_png).convert("RGB")
        if keep_size:
            upscaled = upscaled.resize((orig_w, orig_h), Image.LANCZOS)
        quantized = upscaled.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
        new_w, new_h = quantized.size
        quantized.save(bmp, "BMP")
        scales.append((new_w / orig_w, new_h / orig_h))
        print(f"Quantized {hd_png.name} -> {bmp.name} ({new_w}x{new_h}, 256-color)")

    # The frame's center_x/center_y/offset_x/offset_y in the .ini are pixel
    # coordinates against the *original* frame size. Since the renderer
    # positions/anchors the sprite using these, leaving them unscaled after
    # a 4x upscale points the anchor at 1/4 into the new image instead of
    # the equivalent spot, pushing the real content off-frame (observed as
    # a solid black main menu background). Scale them proportionally.
    rescale_ini_frame_offsets(ini_path, scales)
    print(f"Rescaled frame offsets in {ini_path.name} to match upscaled dimensions")


def cmd_hd_overlay(rel_path: str) -> Path:
    """Emit the full-resolution RGB BMP consumed directly by the engine's
    native-resolution GPU spike (tig_video_set_hd_overlay(), loaded via plain
    SDL_LoadBMP - not the palette-indexed ART/VFS path the rest of this
    pipeline targets). No palette quantization, no downsampling: this is the
    raw AI-upscaled frame, saved as-is.

    Only meaningful for single-frame, full-screen background art (currently
    just MainMenuBack) - the engine hook only ever reads frame 0, and only on
    the mainmenu's fullscreen (4:3) background path. Run 'upscale' first.
    """
    wd = work_dir_for(rel_path)
    bmps = frame_bmps(wd)
    if not bmps:
        raise RuntimeError(f"No frame BMPs found in {wd}; run 'unpack' first")
    hd_png = bmps[0].with_name(bmps[0].stem + "_hd.png")
    if not hd_png.exists():
        raise RuntimeError(f"Missing {hd_png}; run 'upscale' first")

    basename = Path(rel_path.replace("\\", "/")).with_suffix("").name
    dest = config.HD_OVERLAY_DIR / f"{basename}_hd.bmp"
    dest.parent.mkdir(parents=True, exist_ok=True)

    Image.open(hd_png).convert("RGB").save(dest, "BMP")
    print(f"Wrote HD overlay {hd_png.name} -> {dest}")
    return dest


def cmd_repack(rel_path: str) -> Path:
    wd = work_dir_for(rel_path)
    basename = wd / Path(rel_path).with_suffix("").name
    ini_path = basename.with_suffix(".ini")
    out_art = wd / (basename.name + "_HD.ART")
    run_art_converter(ini_path, out_art)
    print(f"Repacked {ini_path.name} -> {out_art}")
    return out_art


def cmd_deploy(rel_path: str, art_path: Path | None = None) -> None:
    rel_path_norm = rel_path.replace("\\", "/")
    if art_path is None:
        wd = work_dir_for(rel_path)
        basename = wd / Path(rel_path).with_suffix("").name
        art_path = wd / (basename.name + "_HD.ART")

    dest = config.DATA_OVERLAY_DIR / rel_path_norm
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(art_path, dest)
    print(f"Deployed {art_path} -> {dest}")


def cmd_revert(rel_path: str) -> None:
    dest = config.DATA_OVERLAY_DIR / rel_path.replace("\\", "/")
    if dest.exists():
        dest.unlink()
        print(f"Removed override {dest} (stock dat-packed art will show again)")
    else:
        print(f"No override present at {dest}")


def cmd_run(rel_path: str, model: str | None = None, keep_size: bool = False) -> None:
    cmd_unpack(rel_path)
    cmd_upscale(rel_path, model=model)
    cmd_quantize(rel_path, keep_size=keep_size)
    cmd_repack(rel_path)
    cmd_deploy(rel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("unpack", "upscale", "quantize", "repack", "deploy", "revert", "hd-overlay", "run"):
        p = sub.add_parser(name)
        p.add_argument("rel_path", help="Path of the .ART relative to a dat root, e.g. art/interface/MainMenuBack.ART")
        if name in ("upscale", "run"):
            p.add_argument("--model", default=None, help=f"ncnn model name to use (default: {config.REALESRGAN_MODEL})")
        if name in ("quantize", "run"):
            p.add_argument(
                "--keep-size",
                action="store_true",
                help="Resample the upscaled art back down to the original pixel "
                "dimensions before quantizing (denoise/sharpen only, no size "
                "change). Required for asset categories where engine code "
                "derives gameplay math from art pixel dimensions, e.g. item "
                "inventory icons — see item_inv_icon_size() pitfall in "
                "HD_ART_PIPELINE.md.",
            )

    args = parser.parse_args()

    if args.command in ("upscale", "run"):
        if args.command == "run":
            cmd_run(args.rel_path, model=args.model, keep_size=args.keep_size)
        else:
            cmd_upscale(args.rel_path, model=args.model)
        return

    if args.command == "quantize":
        cmd_quantize(args.rel_path, keep_size=args.keep_size)
        return

    dispatch = {
        "unpack": cmd_unpack,
        "repack": cmd_repack,
        "deploy": cmd_deploy,
        "revert": cmd_revert,
        "hd-overlay": cmd_hd_overlay,
    }
    dispatch[args.command](args.rel_path)


if __name__ == "__main__":
    main()
