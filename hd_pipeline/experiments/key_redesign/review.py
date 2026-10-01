"""Rows: label | vanilla f0 | HD f0 | vanilla f1 | HD f1 ... in place on the host panel."""
import sys; sys.path.insert(0,'.')
from insitu import scene
from PIL import Image, ImageDraw
spots = []
for a in sys.argv[2:]:
    p = a.split(':'); spots.append((p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4]), p[5]))
rows = []
for panel, px, py, bx, by, but in spots:
    t = scene(panel, px, py, but, bx, by, pad=3)
    rows.append((but, [c for _, c in t]))
cw = max(x.width for _, r in rows for x in r) + 4; ch = max(x.height for _, r in rows for x in r) + 16
s = Image.new("RGB", (cw * max(len(r) for _, r in rows), ch * len(rows)), (24, 24, 24)); d = ImageDraw.Draw(s)
for j, (n, r) in enumerate(rows):
    for i, x in enumerate(r):
        s.paste(x, (i * cw, j * ch + 14))
        d.text((i * cw + 2, j * ch + 1), f"{n} {'vanilla' if i % 2 == 0 else 'HD'} f{i // 2}", fill=(255, 255, 0))
s.save(sys.argv[1]); print(sys.argv[1], s.size)
