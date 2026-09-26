"""Re-letter the place names on the upscaled Map_Zoomed.bmp.

The vanilla labels are ~6 px tall. Every upscaler (ESRGAN, CUGAN, x16->x4)
guesses letters at that size ("GLIMMLRING FOUIST"), so the HD map erases
them and draws them again with IM Fell English SC (OFL, fonts/OFL.txt) -
an old-style small-caps face close to the map's lettering. Each line is
rendered big, cropped to its ink and resized onto the vanilla line's ink
box, so position, width and cap height match the original exactly. The
tiny "The World of" / subtitle text in the cartouche is left alone (not
readable even in vanilla); "Arcanum" is big enough that ESRGAN keeps it.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = Path(__file__).parent / "fonts" / "IMFellEnglishSC.ttf"

# Per label: [(text, ink box of that line in vanilla px (x0, y0, x1, y1),
# exclusive)], measured by hand - the labels sit on rivers, coastlines and
# trees, so auto-detecting lines picks up map features.
LABELS = [
    [("Glimmering", (77, 83, 124, 91)), ("Forest", (88, 93, 112, 99))],
    [("Grey Mountains", (192, 97, 263, 105))],
    [("Vendigroth", (275, 104, 320, 111)), ("Wastes", (282, 113, 311, 120))],
    [("Isle of", (306, 150, 329, 157)), ("Despair", (312, 158, 339, 164))],
    [("Morbihan", (210, 180, 249, 187)), ("Plains", (218, 188, 243, 195))],
    [("Stonewall", (106, 216, 150, 223)), ("Range", (106, 223, 130, 230))],
    [("Thanatos", (218, 314, 256, 320))],
    [("Cattan", (127, 318, 151, 325))],
]


def _glyph_mask(text: str, w: int, h: int) -> np.ndarray:
    """Text rendered at 64 px, cropped to its ink, resized to w x h (0..1)."""
    f = ImageFont.truetype(str(FONT), 64)
    bb = f.getbbox(text)
    im = Image.new("L", (bb[2] - bb[0] + 8, bb[3] - bb[1] + 8), 0)
    ImageDraw.Draw(im).text((4 - bb[0], 4 - bb[1]), text, font=f, fill=255)
    im = im.crop(im.getbbox())
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255


def reletter(vanilla: Image.Image, hd: Image.Image) -> Image.Image:
    from pipeline import inpaint_colorkey

    van = np.asarray(vanilla.convert("RGB"))
    out = np.asarray(hd.convert("RGB")).astype(np.float32)
    s = hd.width // vanilla.width
    for label in LABELS:
        x0 = min(b[0] for _, b in label) - 1
        y0 = min(b[1] for _, b in label) - 1
        x1 = max(b[2] for _, b in label) + 1
        y1 = max(b[3] for _, b in label) + 1
        crop = van[y0:y1, x0:x1]
        lum = crop.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        inbox = np.zeros(lum.shape, bool)
        for _, (bx0, by0, bx1, by1) in label:
            inbox[by0 - y0 - 1:by1 - y0 + 1, bx0 - x0 - 1:bx1 - x0 + 1] = True
        dark = inbox & (lum < np.median(lum) - 30)
        # letter cores, not their anti-aliased edges
        ink = crop[dark & (lum <= np.percentile(lum[dark], 25))].mean(0)
        # erase the upscaler's letters: dilated ink mask, inpainted from the parchment
        m = np.kron(dark, np.ones((s, s), bool))
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(9))) > 0
        mask = np.zeros(out.shape[:2], bool)
        mask[y0 * s:y1 * s, x0 * s:x1 * s] = m
        pad = 6 * s
        ys = slice(max(0, y0 * s - pad), y1 * s + pad)
        xs = slice(max(0, x0 * s - pad), x1 * s + pad)
        out[ys, xs] = inpaint_colorkey(out[ys, xs], mask[ys, xs], max_iter=200)
        for text, (bx0, by0, bx1, by1) in label:
            w, h = (bx1 - bx0) * s, (by1 - by0) * s
            a = _glyph_mask(text, w, h)[..., None]
            region = out[by0 * s:by0 * s + h, bx0 * s:bx0 * s + w]
            out[by0 * s:by0 * s + h, bx0 * s:bx0 * s + w] = region * (1 - a) + ink * a
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
