import sys; sys.path.insert(0, '.')
from keymock import *
import cv2

def simplified_sil():
    m = np.load('key_sil.npy').astype(np.uint8)
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # simplify: smooth the outline (drop the fine notches, keep the bow,
    # collar, shaft, pointed tip and the stepped bit)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 4) > 0.5
    return m

SIL = simplified_sil()
SH, SW = SIL.shape
SDF_REF = ndimage.distance_transform_edt(~SIL) - ndimage.distance_transform_edt(SIL)   # >0 outside

def key_sdf(xx, yy, x0, y0, L, bow_hole):
    s = SW / L                                     # ref px per HD px
    u = (xx - x0) * s
    v = (yy - y0) * s + SH / 2
    sd = ndimage.map_coordinates(SDF_REF, [v, u], order=1, mode="constant", cval=SW) / s
    hole = np.full_like(sd, 1e3)
    if bow_hole:
        bx, by, br = 0.185 * SW, 0.5 * SH, bow_hole * SH
        hole = (np.hypot(u - bx, v - by) - br) / s
    return sd, hole

def render2(name, f, v, shield=None):
    a = np.asarray(Image.open(P.hd_out_dir(f"art/interface/{name}.ART") / f"r0_f{f}.png").convert("RGBA"), np.float32)
    rgb = a[..., :3].copy()
    H, W = rgb.shape[:2]
    gold = hd_gold(a)
    ys, xs = np.nonzero(gold)
    tx0, tx1, ty0, ty1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    old = ndimage.binary_dilation(gold, iterations=7)
    near = ndimage.binary_dilation(gold, iterations=14)
    lum = ndimage.gaussian_filter(a[..., :3].mean(-1), 1.0)
    old |= near & (lum < np.median(lum[near & ~old]) - 25)
    fx, fy, fg = P.FACE_CIRCLE[name]
    yy, xx = np.mgrid[:H, :W] + 0.5
    face = np.hypot(xx - fx, yy - fy) < fg
    L = v["scale"] * (tx1 - tx0)
    x0 = fx - L / 2 + v.get("dx", 0)
    y0 = (ty0 + ty1) / 2
    sd, hole = key_sdf(xx, yy, x0, y0, L, v["bow_hole"])
    newk = sd < 4
    if shield is not None:
        sh = shield
        # dark scratches on the shield where the old key's teeth were
        shl = ndimage.gaussian_filter(np.where(sh, lum, np.nan), 0)
        ring = sh & ~ndimage.binary_dilation(old, iterations=3)
        ref = np.median(lum[ring]) if ring.any() else 80
        old |= sh & ndimage.binary_dilation(gold, iterations=40) & (lum < ref - 35)
        old = ndimage.binary_dilation(old, iterations=1)
        rgb = fill_dom(rgb, old & sh, sh & face, s=8, iters=250)
        rgb = fill_dom(rgb, old & ~sh, face & ~sh, s=1, iters=300)
    else:
        old = ndimage.binary_dilation(old, iterations=1)
        rgb = fill_dom(rgb, old, face, s=1, iters=300)
    A, Hh, (vx0, vx1, vy0, vy1) = vanilla_fields(f)
    kh = L * SH / SW
    sx = vx0 + (xx - x0) / L * (vx1 - vx0) - 0.5
    sy = vy0 + (yy - (y0 - kh / 2)) / kh * (vy1 - vy0) - 0.5
    col = np.stack([ndimage.map_coordinates(A[..., c], [sy, sx], order=3, mode="nearest") for c in range(3)], -1)
    hh = np.stack([ndimage.map_coordinates(Hh[..., c], [sy, sx], order=3, mode="nearest") for c in range(3)], -1)
    col = col + hh * P._dg_checker(H, W)[..., None]
    inside = np.minimum(-sd, hole)                     # >0 inside the key body
    body = np.clip(inside + 0.5, 0, 1)
    # soft outline: darkened key colour, strongest at the edge, fading out
    ow = v["outline"]
    dist = np.maximum(-inside, 0)
    line = np.clip(1 - dist / ow, 0, 1) ** 1.2 * (1 - body)
    line = ndimage.gaussian_filter(line, 0.6)
    # outline = the face underneath darkened (its shading/reflections show through)
    out = rgb * (1 - v.get("strength", 0.5) * line[..., None])
    bev = np.clip(1 - np.maximum(inside, 0) / 2.0, 0, 1) * body
    kc = np.clip(col, 0, 255) * (1 - v.get("bevel", 0.0) * bev[..., None])
    out = out * (1 - body[..., None]) + kc * body[..., None]
    o = a.copy(); o[..., :3] = out
    return o

VARIANTS2 = {
    "E_ref_solid_bow": dict(scale=1.0, bow_hole=0, outline=4.0),
    "F_ref_bow_hole": dict(scale=1.0, bow_hole=0.13, outline=4.0),
    "G_ref_bow_hole_thin_outline": dict(scale=1.0, bow_hole=0.13, outline=2.5),
    "H_ref_smaller_bow_hole": dict(scale=0.92, bow_hole=0.16, outline=3.5),
}
if __name__ == "__main__":
    fr = P._dg_frames("char_Common_Skills")
    shield = ndimage.binary_erosion(P._dg_up(P._dg_shield_mask(fr).astype(np.float32)) > 0.5, iterations=2)
    hosts = {"Skills_Button": ("IntBotom", 693, 15), "char_Common_Skills": ("Char_Maint", 527, 11)}
    rows = []
    for vn, v in [("current", None)] + list(VARIANTS2.items()):
        tiles = []
        for name in ["Skills_Button", "char_Common_Skills"]:
            pn, bx, by = hosts[name]
            pan = Image.open(P.hd_out_dir(f"art/interface/{pn}.ART") / "r0_f0.png").convert("RGBA")
            for f in (0, 2, 1):
                o = np.asarray(Image.open(P.hd_out_dir(f"art/interface/{name}.ART") / f"r0_f{f}.png").convert("RGBA"), np.float32) if v is None else render2(name, f, v, shield if name == "char_Common_Skills" else None)
                im = Image.fromarray(np.clip(o + .5, 0, 255).astype(np.uint8), "RGBA")
                bg = pan.crop((bx * 4, by * 4, bx * 4 + im.width, by * 4 + im.height)); bg.alpha_composite(im)
                tiles.append(bg.convert("RGB"))
        rows.append((vn, tiles))
    cw = max(t.width for _, r in rows for t in r) + 4; ch = max(t.height for _, r in rows for t in r) + 16
    s = Image.new("RGB", (cw * 6, ch * len(rows)), (24, 24, 24)); d = ImageDraw.Draw(s)
    for j, (vn, r) in enumerate(rows):
        d.text((2, j * ch + 1), f"{vn}   (rest / hover / pressed, HUD then shield)", fill=(255, 255, 0))
        for i, t in enumerate(r): s.paste(t, (i * cw, j * ch + 14))
    s = s.resize((s.width * 3 // 2, s.height * 3 // 2), Image.LANCZOS)
    s.save(sys.argv[1]); print(s.size)
