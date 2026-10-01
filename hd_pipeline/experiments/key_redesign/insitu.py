"""Button art placed on its panel: vanilla (x4 nearest) | HD, every frame."""
import sys, os
sys.path.insert(0, r"G:\coding\repos\art-converter\hd_pipeline")
import numpy as np, pipeline as P, config
from PIL import Image
def van(name, k=0):
    wd = P.cmd_unpack(f"art/interface/{name}.ART", quiet=True)
    n, an = P.read_ini_frame_count(wd / (name + ".ini"))
    bmp = P.hd_frame_bmps(wd, name, n, an)[k][2]
    w, h = P.read_bmp_dims(bmp); h = abs(h)
    idx = np.asarray(P.read_bmp_indices(bmp), np.uint8).reshape(h, w)
    pal = np.asarray(P.read_bmp_palette(bmp), np.uint8).reshape(256, 3)
    return Image.fromarray(np.dstack([pal[idx], np.where(idx == 0, 0, 255)]).astype(np.uint8), "RGBA")
def hd(name, k=0):
    return Image.open(config.HD_OVERLAY_DIR / f"art/interface/{name}/r0_f{k}.png").convert("RGBA")
def nframes(name):
    wd = P.cmd_unpack(f"art/interface/{name}.ART", quiet=True)
    n, an = P.read_ini_frame_count(wd / (name + ".ini")); return len(P.hd_frame_bmps(wd, name, n, an))
def scene(panel, px, py, but, bx, by, pad=6, z=3, base=None):
    """px,py: panel origin; bx,by: button pos (same coords, 1x)."""
    out = []
    for k in range(nframes(but)):
        for src, s in ((van, 4), (hd, 4)):
            pan = src(panel) if src is van else hd(panel)
            b = src(but, k)
            if src is van:
                pan = pan.resize((pan.width * 4, pan.height * 4), Image.NEAREST); b = b.resize((b.width * 4, b.height * 4), Image.NEAREST)
            x, y = (bx - px) * 4, (by - py) * 4
            bw, bh = b.size
            crop = (x - pad * 4, y - pad * 4, x + bw + pad * 4, y + bh + pad * 4)
            canvas = Image.new("RGBA", pan.size, (0, 0, 0, 255)); canvas.alpha_composite(pan)
            under = canvas.crop(crop).convert("RGB")
            canvas.alpha_composite(b, (x, y))
            out.append((under, canvas.crop(crop).convert("RGB")))
    return out
if __name__ == "__main__":
    panel, px, py, but, bx, by, outp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    tiles = scene(panel, px, py, but, bx, by)
    w, h = tiles[0][0].size
    sh = Image.new("RGB", ((w + 6) * len(tiles), (h + 6) * 2), (255, 0, 255))
    for i, (u, c) in enumerate(tiles):
        sh.paste(u, (i * (w + 6), 0)); sh.paste(c, (i * (w + 6), h + 6))
    sh.save(outp); print(outp, sh.size)
