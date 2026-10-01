import sys; sys.path.insert(0, '.')
import keymock2 as K
from keymock import *
import cv2

def clean_sil(W=1000, notch=True):
    """Simplified reference key, units: W = key length; centre line y=0."""
    H = int(W * 0.44)
    img = np.zeros((H, W), np.uint8)
    cy = H // 2
    s = W / 1000
    P_ = lambda pts: np.array([[int(x * s), int(cy + y * s)] for x, y in pts], np.int32)
    cv2.circle(img, (int(215 * s), cy), int(215 * s), 1, -1)
    # collar flanges
    cv2.fillPoly(img, [P_([(380, -95), (470, -115), (490, -115), (490, 115), (470, 115), (380, 95)])], 1)
    # shaft with pointed tip
    cv2.fillPoly(img, [P_([(480, -62), (930, -62), (1000, 5), (960, 40), (480, 62)])], 1)
    cv2.fillPoly(img, [P_([(480, -62), (930, -62), (1000, 5), (960, 40), (930, 62), (480, 62)])], 1)
    # bit: stepped teeth under the shaft
    if notch == "steps":
        bit = [(540, 55), (540, 140), (620, 140), (620, 105), (680, 105), (680, 140), (760, 140), (760, 105), (820, 105), (820, 140), (880, 140), (920, 55)]
    elif notch == "two":
        bit = [(560, 55), (560, 145), (650, 145), (650, 55)]
        cv2.fillPoly(img, [P_([(720, 55), (720, 145), (810, 145), (850, 55)])], 1)
    else:
        bit = [(540, 55), (540, 125), (900, 125), (930, 55)]
    cv2.fillPoly(img, [P_(bit)], 1)
    m = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 3 * s) > 0.5
    return m

def install(m):
    K.SIL = m
    K.SH, K.SW = m.shape
    K.SDF_REF = ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)

_vf = K.vanilla_fields
def vanilla_fields_clean(f):
    fr = {ff: (c, k) for _, ff, c, k in P._dg_frames("Skills_Button")}
    c, k = fr[f]
    m = P._dg_keymask(c, k)
    avg, half, _ = P._dg_decompose(c, k)
    lum = avg.mean(-1)
    core = ndimage.binary_erosion(m, np.ones((2, 2))) & (lum > np.median(lum[m]) * 0.7)   # no keyhole
    A = P._nconv(avg, core.astype(np.float32), sigmas=(1, 2, 4, 8))
    A = ndimage.gaussian_filter(A, (0.8, 0.8, 0))
    Hh = P._nconv(half, core.astype(np.float32), sigmas=(1, 2, 4, 8))
    ys, xs = np.nonzero(m)
    return A, Hh, (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)
K.vanilla_fields = vanilla_fields_clean

V3 = {
    "M_stepped_bit": dict(scale=1.0, bow_hole=0.12, outline=5.0, notch="steps"),
    "N_two_teeth": dict(scale=1.0, bow_hole=0.12, outline=5.0, notch="two"),
    "O_plain_bit": dict(scale=1.0, bow_hole=0.12, outline=5.0, notch="plain"),
    "P_two_teeth_solid_bow": dict(scale=1.0, bow_hole=0, outline=5.0, notch="two"),
}
if __name__ == "__main__":
    # stronger outline
    src = open('keymock2.py').read()
    fr = P._dg_frames("char_Common_Skills")
    shield = ndimage.binary_erosion(P._dg_up(P._dg_shield_mask(fr).astype(np.float32)) > 0.5, iterations=2)
    hosts = {"Skills_Button": ("IntBotom", 693, 15), "char_Common_Skills": ("Char_Maint", 527, 11)}
    rows = []
    for vn, v in V3.items():
        install(clean_sil(notch=v["notch"]))
        tiles = []
        for name in ["Skills_Button", "char_Common_Skills"]:
            pn, bx, by = hosts[name]
            pan = Image.open(P.hd_out_dir(f"art/interface/{pn}.ART") / "r0_f0.png").convert("RGBA")
            for f in (0, 2, 1):
                o = K.render2(name, f, v, shield if name == "char_Common_Skills" else None)
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
