import sys; sys.path.insert(0, '.')
import keymock2 as K
import keymock3 as K3
from keymock import *
import cv2

def sil(neck=True, notch=True, smooth=10):
    m = np.load('key2_sil.npy').astype(np.uint8)
    m[:, 440:1000] = 0
    m[120:226, 440:1000] = 1
    m[120:354, 815:990] = 1
    if neck:
        cv2.circle(m, (385, 58), 62, 0, -1)
        cv2.circle(m, (385, 288), 64, 0, -1)
    if notch:
        cv2.fillPoly(m, [np.array([[872, 360], [903, 292], [934, 360]], np.int32)], 0)
    m = np.pad(m.astype(np.float32), 20)
    m = cv2.GaussianBlur(m, (0, 0), smooth) > 0.5
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def vanilla_face(name, f, shield1x=None):
    """Vanilla's frame f without its key, smooth at HD: the face's dome
    shading/colour, and (shield variant) the shield with its checker."""
    fr = {ff: (c, k) for _, ff, c, k in P._dg_frames(name)}
    c, k = fr[f]
    avg, half, _ = P._dg_decompose(c, k)
    # the key in every frame (the resting one is too dim to find on its own)
    key = np.zeros_like(k)
    for cc, kk in fr.values():
        key |= P._dg_keymask(cc, kk)
    lum = avg.mean(-1)
    hole = ndimage.binary_dilation(key, np.ones((3, 3)), iterations=1)
    near = ndimage.binary_dilation(key, np.ones((3, 3)), iterations=3)
    hole |= near & (lum < np.median(lum[~k & ~near]) - 8)
    hole = ndimage.binary_dilation(hole, np.ones((2, 2))) & ~k
    base = P.inpaint_colorkey(avg, k)
    if shield1x is None:
        filled = P._dg_fill(base, hole, 600)
        return ndimage.gaussian_filter(P._dg_up(filled), (2.5, 2.5, 0)), None
    sh = shield1x
    # the key's shadow on the shield: darker than the shield around it
    near4 = ndimage.binary_dilation(key, np.ones((3, 3)), iterations=4) & sh
    sl = lum[sh & ~near4]
    hole |= near4 & (lum < np.median(sl) - 12)
    hole = ndimage.binary_dilation(hole, np.ones((2, 2))) & ~k
    core = ndimage.binary_erosion(sh, np.ones((3, 3)))
    face_dom = ~sh & ~k
    fa = P._dg_fill_in(base, hole & face_dom, face_dom, 600)
    sa = P._dg_fill_in(base, sh & ~(core & ~hole), sh, 600)
    sha = P._dg_fill_in(half, sh & ~(core & ~hole), sh, 600)
    fa = P._dg_fill_in(fa, sh, ~k, 600)          # face colours under the shield: no red halo
    face = ndimage.gaussian_filter(P._dg_up(fa), (2.5, 2.5, 0))
    # shield at HD: smooth average + checker, crisp edge from its convex shape
    up_sa = ndimage.gaussian_filter(P._dg_up(sa), (1.0, 1.0, 0))
    up_h = P._dg_up(sha) * P._dg_checker(up_sa.shape[0], up_sa.shape[1])[..., None]
    shield_rgb = up_sa + up_h
    return face, shield_rgb

_render = K.render2
def render3(name, f, v, shield=None, shield1x=None):
    a = np.asarray(Image.open(P.hd_out_dir(f"art/interface/{name}.ART") / f"r0_f{f}.png").convert("RGBA"), np.float32)
    o = _render(name, f, v, shield)
    fx, fy, fg = P.FACE_CIRCLE[name]
    H, W = a.shape[:2]
    yy, xx = np.mgrid[:H, :W] + 0.5
    rr = np.hypot(xx - fx, yy - fy)
    face, shield_rgb = vanilla_face(name, f, shield1x)
    keep = np.abs(o[..., :3] - _bg_cache[(name, f)]).max(-1) > 1.5
    keep = ndimage.gaussian_filter(keep.astype(np.float32), 0.7)
    inner = np.clip((fg - 14 - rr) / 8, 0, 1)            # face interior; the rim's band stays HD
    newbg = _bg_cache[(name, f)] * (1 - inner[..., None]) + face * inner[..., None]
    if shield_rgb is not None:
        # the shield's crisp convex outline at HD
        import cv2
        ys, xs = np.nonzero(shield1x)
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.float32))
        sm = np.zeros((H * 4, W * 4), np.uint8)
        cv2.fillConvexPoly(sm, ((hull[:, 0] + [0.5, 0.5]) * 16).astype(np.int32), 1)
        sm = cv2.resize(sm.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)[..., None]
        sm = np.clip((sm - 0.5) * 3.0 + 0.5, 0, 1)
        newbg = newbg * (1 - sm) + shield_rgb * sm
    out = o.copy()
    out[..., :3] = o[..., :3] * keep[..., None] + newbg * (1 - keep[..., None])
    # outline darkening re-applied over the new face
    return out

_bg_cache = {}
_fill_dom = K.fill_dom
def fill_dom_cached(img, hole, domain, s=1, iters=300):
    return _fill_dom(img, hole, domain, s, iters)

# capture the background (old key erased) render2 builds: wrap render2 to record it
def render2_bg(name, f, v, shield=None):
    v0 = dict(v, outline=0.01, strength=0.0)
    saved = K.key_sdf
    K.key_sdf = lambda xx, yy, x0, y0, L, bh: (np.full(xx.shape, 1e3), np.full(xx.shape, 1e3))
    bg = _render(name, f, v0, shield)
    K.key_sdf = saved
    return bg[..., :3]

V5 = {
    "Y_cuts_vanilla_face (outline 50%, 3px)": dict(scale=1.0, bow_hole=0, outline=3.0, strength=0.5),
    "Z_cuts_vanilla_face (outline 35%, 3px)": dict(scale=1.0, bow_hole=0, outline=3.0, strength=0.35),
}
if __name__ == "__main__":
    K3.install(sil())
    fr = P._dg_frames("char_Common_Skills")
    s1 = P._dg_shield_mask(fr)
    shield = ndimage.binary_erosion(P._dg_up(s1.astype(np.float32)) > 0.5, iterations=2)
    hosts = {"Skills_Button": ("IntBotom", 693, 15), "char_Common_Skills": ("Char_Maint", 527, 11)}
    from insitu import van
    rows = []
    # vanilla row for reference
    tiles = []
    for name in ["Skills_Button", "char_Common_Skills"]:
        pn, bx, by = hosts[name]
        pan = van(pn).resize((van(pn).width * 4, van(pn).height * 4), Image.NEAREST)
        for f in (0, 2, 1):
            im = van(name, f); im = im.resize((im.width * 4, im.height * 4), Image.NEAREST)
            bg = pan.crop((bx * 4, by * 4, bx * 4 + im.width, by * 4 + im.height)); bg.alpha_composite(im)
            tiles.append(bg.convert("RGB"))
    rows.append(("vanilla", tiles))
    for vn, v in V5.items():
        tiles = []
        for name in ["Skills_Button", "char_Common_Skills"]:
            pn, bx, by = hosts[name]
            pan = Image.open(P.hd_out_dir(f"art/interface/{pn}.ART") / "r0_f0.png").convert("RGBA")
            sh = shield if name == "char_Common_Skills" else None
            for f in (0, 2, 1):
                _bg_cache[(name, f)] = render2_bg(name, f, v, sh)
                o = render3(name, f, v, sh, s1 if sh is not None else None)
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
