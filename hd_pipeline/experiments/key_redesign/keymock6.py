import sys; sys.path.insert(0, '.'); sys.path.insert(0, r"G:\coding\repos\art-converter\hd_pipeline")
import numpy as np, pipeline as P, config, cv2
from PIL import Image, ImageDraw
from scipy import ndimage
import keymock5 as M5
S = P.HD_SCALE

def basis(x, y, deg):
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)

def polyfit(vals, xs, ys, deg, it=4):
    B = basis(xs, ys, deg); inl = np.ones(len(vals), bool)
    for _ in range(it):
        c, *_ = np.linalg.lstsq(B[inl], vals[inl], rcond=None)
        r = np.abs(vals - B @ c).max(-1); inl = r < 2.5 * np.median(r[inl]) + 3
    return c

def frames(name):
    return {f: (c, k) for _, f, c, k in P._dg_frames(name)}

def key1x(fr):
    key = np.zeros_like(next(iter(fr.values()))[1])
    for c, k in fr.values(): key |= P._dg_keymask(c, k)
    return key

def render(name, f, sil_sd, opts, shield1x=None):
    fr = frames(name); c, k = fr[f]
    avg, half, _ = P._dg_decompose(c, k)
    a = np.asarray(Image.open(P.hd_out_dir(f"art/interface/{name}.ART") / f"r0_f{f}.png").convert("RGBA"), np.float32)
    H, W = a.shape[:2]
    fx, fy, fg = P.FACE_CIRCLE[name]
    key = key1x(fr)
    lum = avg.mean(-1)
    near = ndimage.binary_dilation(key, np.ones((3, 3)), iterations=3)
    excl = ndimage.binary_dilation(key, np.ones((3, 3)), iterations=1) | (near & (lum < np.median(lum[~k & ~near]) - 8))
    h1, w1 = k.shape
    yy1, xx1 = np.mgrid[:h1, :w1] + 0.5
    r1 = np.hypot(xx1 - fx / S, yy1 - fy / S)
    norm = lambda x, y: ((x - fx / S) / (fg / S), (y - fy / S) / (fg / S))
    facepx = (r1 < fg / S - 2.0) & ~k & ~excl
    if shield1x is not None:
        facepx &= ~ndimage.binary_dilation(shield1x, iterations=1)
    X, Y = norm(xx1[facepx], yy1[facepx])
    cf = polyfit(avg[facepx], X, Y, opts.get("face_deg", 4))
    yy, xx = np.mgrid[:H, :W] + 0.5
    Xh, Yh = norm(xx / S, yy / S)
    face = basis(Xh, Yh, opts.get("face_deg", 4)) @ cf
    rr = np.hypot(xx - fx, yy - fy)
    inner = np.clip((fg - 12 - rr) / 8, 0, 1)[..., None]
    out = a[..., :3] * (1 - inner) + face * inner
    if shield1x is not None:
        sh = shield1x
        spx = sh & ~excl & ~ndimage.binary_dilation(k, iterations=1)
        spx &= ndimage.binary_erosion(sh, np.ones((3, 3)))
        Xs, Ys = norm(xx1[spx], yy1[spx])
        cs = polyfit(avg[spx], Xs, Ys, 2)
        amp = np.median(half[spx], 0)
        srgb = basis(Xh, Yh, 2) @ cs + amp * P._dg_checker(H, W)[..., None]
        ys, xs = np.nonzero(sh)
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.float32))
        sm = np.zeros((H * 4, W * 4), np.uint8)
        cv2.fillConvexPoly(sm, ((hull[:, 0] + 0.5) * 16).astype(np.int32), 1)
        sm = cv2.resize(sm.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        sm = np.clip((sm - 0.5) * 2.5 + 0.5, 0, 1)[..., None]
        out = out * (1 - sm) + srgb * sm
    # key: vanilla's colour + checker for this frame, lit along its length like vanilla, darker rim
    # the HUD key's own cells for this frame (both buttons draw the same key)
    hc, hk = frames("Skills_Button")[f]
    kp = ndimage.binary_erosion(P._dg_keymask(hc, hk), np.ones((2, 2)))
    py, px = np.mgrid[:h1, :w1]
    even = (px + py) % 2 == 0
    e, od = np.median(hc[kp & even], 0), np.median(hc[kp & ~even], 0)
    base, amp = (e + od) / 2, (e - od) / 2 * opts.get("contrast", 1.0)
    hl = hc.mean(-1)
    w1h = hk.shape[1]
    prof = np.array([np.median(hl[kp[:, x], x]) if kp[:, x].any() else np.nan for x in range(w1h)])
    good = ~np.isnan(prof)
    prof = np.interp(np.arange(w1h), np.nonzero(good)[0], prof[good]) / np.median(hl[kp])
    prof = np.clip(ndimage.gaussian_filter1d(prof, 2.0), 0.9, 1.35)
    kx0, kx1 = np.nonzero(kp.any(0))[0][[0, -1]]
    gold = M5.hd_gold(a)
    ys, xs = np.nonzero(gold)
    tx0, tx1 = xs.min(), xs.max() + 1
    sd, L, x0 = sil_sd(H, W, fx, (ys.min() + ys.max() + 1) / 2, tx1 - tx0)
    along = np.interp(kx0 + (xx - x0) / L * (kx1 + 1 - kx0), np.arange(w1h) + 0.5, prof)
    inside = -sd
    t = np.clip(inside / opts.get("rim_w", 5.0), 0, 1)
    shade = opts.get("rim_dark", 0.6) + (1 - opts.get("rim_dark", 0.6)) * (t * t * (3 - 2 * t))
    krgb = base * (along * shade)[..., None] + amp * shade[..., None] * P._dg_checker(H, W)[..., None]
    body = np.clip(inside + 0.5, 0, 1)[..., None]
    ow = opts.get("outline", 3.0)
    line = np.clip(1 - np.maximum(-inside, 0) / ow, 0, 1) ** 1.2 * (1 - body[..., 0])
    line = ndimage.gaussian_filter(line, 0.6)[..., None]
    out = out * (1 - opts.get("strength", 0.5) * line)
    out = out * (1 - body) + np.clip(krgb, 0, 255) * body
    o = a.copy(); o[..., :3] = out
    return o

SIL = M5.sil()
SDF = ndimage.distance_transform_edt(~SIL) - ndimage.distance_transform_edt(SIL)
def sil_sd(H, W, fx, cy, width, scale=1.0):
    L = width * scale; x0 = fx - L / 2
    s = SIL.shape[1] / L
    yy, xx = np.mgrid[:H, :W] + 0.5
    u = (xx - x0) * s; v = (yy - cy) * s + SIL.shape[0] / 2
    return ndimage.map_coordinates(SDF, [v, u], order=1, mode="constant", cval=SIL.shape[1]) / s, L, x0

VAR = {
    "AD rim 60% / 5px, outline 50%": dict(rim_dark=0.6, rim_w=5.0, strength=0.5),
    "AE rim 45% / 7px, outline 50%": dict(rim_dark=0.45, rim_w=7.0, strength=0.5),
    "AF rim 60% / 5px, outline 35%, softer dots": dict(rim_dark=0.6, rim_w=5.0, strength=0.35, contrast=0.75),
}
if __name__ == "__main__":
    from insitu import van
    s1 = P._dg_shield_mask(P._dg_frames("char_Common_Skills"))
    hosts = {"Skills_Button": ("IntBotom", 693, 15), "char_Common_Skills": ("Char_Maint", 527, 11)}
    rows = []
    tiles = []
    for name in hosts:
        pn, bx, by = hosts[name]
        pv = van(pn); pan = pv.resize((pv.width * 4, pv.height * 4), Image.NEAREST)
        for f in (0, 2, 1):
            im = van(name, f); im = im.resize((im.width * 4, im.height * 4), Image.NEAREST)
            bg = pan.crop((bx * 4, by * 4, bx * 4 + im.width, by * 4 + im.height)); bg.alpha_composite(im); tiles.append(bg.convert("RGB"))
    rows.append(("vanilla", tiles))
    for vn, opt in VAR.items():
        tiles = []
        for name in hosts:
            pn, bx, by = hosts[name]
            pan = Image.open(P.hd_out_dir(f"art/interface/{pn}.ART") / "r0_f0.png").convert("RGBA")
            for f in (0, 2, 1):
                o = render(name, f, sil_sd, opt, s1 if name == "char_Common_Skills" else None)
                im = Image.fromarray(np.clip(o + .5, 0, 255).astype(np.uint8), "RGBA")
                bg = pan.crop((bx * 4, by * 4, bx * 4 + im.width, by * 4 + im.height)); bg.alpha_composite(im); tiles.append(bg.convert("RGB"))
        rows.append((vn, tiles))
    cw = max(t.width for _, r in rows for t in r) + 4; ch = max(t.height for _, r in rows for t in r) + 16
    s = Image.new("RGB", (cw * 6, ch * len(rows)), (24, 24, 24)); d = ImageDraw.Draw(s)
    for j, (vn, r) in enumerate(rows):
        d.text((2, j * ch + 1), f"{vn}   (rest / hover / pressed, HUD then shield)", fill=(255, 255, 0))
        for i, t in enumerate(r): s.paste(t, (i * cw, j * ch + 14))
    s = s.resize((s.width * 3 // 2, s.height * 3 // 2), Image.LANCZOS)
    s.save(sys.argv[1]); print(s.size)
