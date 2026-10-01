import sys, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, r"G:\coding\repos\art-converter\hd_pipeline")
import config, pipeline as P
STAGE = config.WORK_DIR / "_critter_remacri"
BG = (58, 62, 48)
def on(im, size):
    im = im.convert("RGBA").resize(size, Image.NEAREST if im.width < size[0] / 2 else Image.LANCZOS)
    b = Image.new("RGBA", size, BG + (255,)); b.alpha_composite(im); return b.convert("RGB")
rels = [r for r in P.find_category_files("critter")]
random.seed(int(sys.argv[2])); pick = random.sample(rels, 40)
rows = []
for rel in pick:
    base = Path(rel).stem
    live = P.hd_out_dir(rel)
    stg = STAGE / live.relative_to(config.HD_OVERLAY_DIR)
    if not (stg / "r1_f0.png").exists() or not (live / "r1_f0.png").exists():
        continue
    wd = P.cmd_unpack(rel, quiet=True)
    nf, anim = P.read_ini_frame_count(wd / (base + ".ini"))
    fr = {(r, f): b for r, f, b in P.hd_frame_bmps(wd, base, nf, anim)}
    b = fr[(1, 0)]
    idx = np.asarray(Image.open(b)); v = np.asarray(Image.open(b).convert("RGBA")).copy(); v[idx == 0, 3] = 0
    h, w = idx.shape; K = 3.0 if max(w, h) < 90 else 2.0
    sz = (round(w * K), round(h * K))
    rows.append((base, [on(Image.fromarray(v), sz), on(Image.open(live / "r1_f0.png"), sz), on(Image.open(stg / "r1_f0.png"), sz)], sz))
    if len(rows) >= 8: break
cw = max(3 * r[2][0] + 24 for r in rows); s = Image.new("RGB", (cw * 2, 0))
lines = [rows[i:i + 2] for i in range(0, len(rows), 2)]
H = sum(max(r[2][1] for r in L) + 18 for L in lines)
s = Image.new("RGB", (cw * 2, H), (24, 24, 24)); d = ImageDraw.Draw(s); y = 0
for L in lines:
    for j, (n, ims, sz) in enumerate(L):
        x = j * cw
        d.text((x + 2, y + 2), f"{n}: vanilla | current | remacri+edge", fill=(255, 255, 0))
        for i, im in enumerate(ims): s.paste(im, (x + i * (sz[0] + 6), y + 16))
    y += max(r[2][1] for r in L) + 18
s.save(sys.argv[1]); print(s.size, [r[0] for r in rows])
