import sys; sys.path.insert(0, r"G:\coding\repos\art-converter\hd_pipeline")
import numpy as np, pipeline as P, config
from PIL import Image, ImageDraw
from scipy import ndimage

def hd_gold(a):
    R_, G_, B_ = a[..., 0], a[..., 1], a[..., 2]
    g = (R_ > 70) & (G_ > 0.72 * R_) & (G_ < 1.25 * R_) & (B_ < 0.6 * R_)
    g = ndimage.gaussian_filter(g.astype(float), 1.5) > 0.5
    lab, n = ndimage.label(g)
    return ndimage.binary_fill_holes(lab == 1 + np.argmax(ndimage.sum(g, lab, range(1, n + 1))))

def fill_dom(img, hole, domain, s=1, iters=300):
    out = img.copy()
    known = domain & ~hole
    out[hole] = P._nconv(img, known.astype(np.float32))[hole]
    d = domain.astype(np.float32)
    H, W = hole.shape
    for _ in range(iters):
        acc = np.zeros_like(out); cnt = np.zeros((H, W), np.float32)
        for dy, dx in ((s, 0), (-s, 0), (0, s), (0, -s)):
            sh = np.roll(np.roll(out, dy, 0), dx, 1); sd = np.roll(np.roll(d, dy, 0), dx, 1)
            acc += sh * sd[..., None]; cnt += sd
        upd = acc / np.maximum(cnt, 1)[..., None]
        out[hole] = upd[hole]
    return out

def rect_sd(X, Y, x0, x1, y0, y1, r=1.5):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; hx, hy = (x1 - x0) / 2 - r, (y1 - y0) / 2 - r
    qx, qy = np.abs(X - cx) - hx, np.abs(Y - cy) - hy
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r

def key_sd(X, Y, x0, y0, L, v):
    u = lambda t: x0 + t * L
    w = lambda t: y0 + t * L
    bow = np.hypot(X - u(v["bow_x"]), Y - w(0)) - v["bow_r"] * L
    sh = rect_sd(X, Y, u(v["bow_x"]), u(1.0), w(-v["shaft"]), w(v["shaft"]))
    # slanted tip: cut the top-right corner
    tip = (X - u(1.0)) * 0.6 - (Y - w(-v["shaft"])) * 1.0 + v["tip"] * L
    sh = np.maximum(sh, -(-tip)) if v["tip"] > 0 else sh
    sd = np.minimum(bow, sh)
    for t0, t1 in v["teeth"]:
        sd = np.minimum(sd, rect_sd(X, Y, u(t0), u(t1), w(0), w(v["shaft"] + v["tooth"])))
    hole = np.hypot(X - u(v["hole_x"]), Y - w(0)) - v["hole_r"] * L
    if v.get("notch"):
        nx = u(v["notch"]); tri = np.maximum(np.abs(X - nx) * 1.2 + (Y - w(-v["shaft"])) , -(Y - w(-v["shaft"]) - 0.05 * L))
        hole = np.minimum(hole, np.abs(X - nx) * 1.3 - (w(-v["shaft"]) + 0.055 * L - Y))
    return sd, hole

BASE = dict(bow_x=0.17, bow_r=0.17, shaft=0.07, tip=0.04, tooth=0.11, teeth=[(0.65, 0.75), (0.77, 0.87)], hole_x=0.12, hole_r=0.055)
VARIANTS = {
    "A_checker_outline3": dict(BASE, outline=3.0, checker=True),
    "B_checker_outline2_notch": dict(BASE, outline=2.0, checker=True, notch=0.52),
    "C_smooth_outline3": dict(BASE, outline=3.0, checker=False),
    "D_bigbow_slim": dict(BASE, bow_r=0.19, bow_x=0.19, shaft=0.06, hole_r=0.065, hole_x=0.15, outline=3.0, checker=True),
}

def vanilla_fields(f):
    fr = {ff: (c, k) for _, ff, c, k in P._dg_frames("Skills_Button")}
    c, k = fr[f]
    m = P._dg_keymask(c, k)
    avg, half, _ = P._dg_decompose(c, k)
    core = ndimage.binary_erosion(m, np.ones((2, 2)))
    A = P._nconv(avg, core.astype(np.float32), sigmas=(1, 2, 4, 8))
    Hh = P._nconv(half, m.astype(np.float32), sigmas=(1, 2, 4, 8))
    ys, xs = np.nonzero(m)
    return A, Hh, (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)

def render(name, f, v, panel_shield=None):
    a = np.asarray(Image.open(P.hd_out_dir(f"art/interface/{name}.ART") / f"r0_f{f}.png").convert("RGBA"), np.float32)
    rgb = a[..., :3].copy()
    H, W = rgb.shape[:2]
    gold = hd_gold(a)
    ys, xs = np.nonzero(gold)
    tx0, tx1, ty0, ty1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    old = ndimage.binary_dilation(gold, iterations=7)
    near = ndimage.binary_dilation(gold, iterations=14)
    lum = ndimage.gaussian_filter(a[..., :3].mean(-1), 1.0)
    old |= near & (lum < np.median(lum[near & ~old]) - 25)   # the old key's outline / shadow
    old = ndimage.binary_dilation(old, iterations=1)
    fx, fy, fg = P.FACE_CIRCLE[name]
    yy, xx = np.mgrid[:H, :W] + 0.5
    face = np.hypot(xx - fx, yy - fy) < fg
    if panel_shield is not None:
        sh = panel_shield
        rgb = fill_dom(rgb, old & sh, sh & face, s=8, iters=250)
        rgb = fill_dom(rgb, old & ~sh, face & ~sh, s=1, iters=300)
    else:
        rgb = fill_dom(rgb, old, face, s=1, iters=300)
    L = (tx1 - tx0); x0 = tx0; y0 = (ty0 + ty1) / 2 - 0.005 * L
    sd, hole = key_sd(xx, yy, x0, y0, L, v)
    A, Hh, (vx0, vx1, vy0, vy1) = vanilla_fields(f)
    # sample vanilla's fields in key coordinates
    kx0, kx1 = x0, x0 + L
    sx = vx0 + (xx - kx0) / L * (vx1 - vx0) - 0.5
    sy = vy0 + (yy - (ty0)) / (ty1 - ty0) * (vy1 - vy0) - 0.5
    col = np.stack([ndimage.map_coordinates(A[..., c], [sy, sx], order=3, mode="nearest") for c in range(3)], -1)
    if v["checker"]:
        hh = np.stack([ndimage.map_coordinates(Hh[..., c], [sy, sx], order=3, mode="nearest") for c in range(3)], -1)
        col = col + hh * P._dg_checker(H, W)[..., None]
    body = np.clip(0.5 - sd, 0, 1) * np.clip(0.5 + hole, 0, 1)
    ow = v["outline"]
    outer = np.clip(0.5 - (sd - ow), 0, 1)
    hring = np.clip(0.5 - np.abs(hole + ow / 2) + ow / 2, 0, 1) * (sd < 0)
    line = np.clip(np.maximum(outer - body, hring), 0, 1)
    dark = np.array([22, 20, 6], np.float32)
    out = rgb * (1 - line[..., None] * 0.9) + dark * line[..., None] * 0.9
    out = out * (1 - body[..., None]) + np.clip(col, 0, 255) * body[..., None]
    o = a.copy(); o[..., :3] = out
    return o

if __name__ == "__main__":
    shield = None
    fr = P._dg_frames("char_Common_Skills")
    sm = P._dg_shield_mask(fr)
    shield = P._dg_up(sm.astype(np.float32)) > 0.5
    shield = ndimage.binary_erosion(shield, iterations=2)
    hosts = {"Skills_Button": ("IntBotom", 693, 15), "char_Common_Skills": ("Char_Maint", 527, 11)}
    rows = []
    for vn, v in [("current", None)] + list(VARIANTS.items()):
        tiles = []
        for name in ["Skills_Button", "char_Common_Skills"]:
            pn, bx, by = hosts[name]
            pan = Image.open(P.hd_out_dir(f"art/interface/{pn}.ART") / "r0_f0.png").convert("RGBA")
            for f in (0, 2, 1):
                if v is None:
                    o = np.asarray(Image.open(P.hd_out_dir(f"art/interface/{name}.ART") / f"r0_f{f}.png").convert("RGBA"), np.float32)
                else:
                    o = render(name, f, v, shield if name == "char_Common_Skills" else None)
                im = Image.fromarray(np.clip(o + .5, 0, 255).astype(np.uint8), "RGBA")
                bg = pan.crop((bx * 4, by * 4, bx * 4 + im.width, by * 4 + im.height)); bg.alpha_composite(im)
                tiles.append(bg.convert("RGB"))
        rows.append((vn, tiles))
    cw = max(t.width for _, r in rows for t in r) + 4; ch = max(t.height for _, r in rows for t in r) + 16
    s = Image.new("RGB", (cw * 6, ch * len(rows)), (24, 24, 24)); d = ImageDraw.Draw(s)
    for j, (vn, r) in enumerate(rows):
        d.text((2, j * ch + 1), f"{vn}   (rest / hover / pressed, HUD then shield)", fill=(255, 255, 0))
        for i, t in enumerate(r): s.paste(t, (i * cw, j * ch + 14))
    s.save(sys.argv[1]); print(s.size)
