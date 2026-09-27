"""Re-letter the place names on the upscaled Map_Zoomed.bmp.

The vanilla labels are ~6 px tall. Every upscaler (ESRGAN, CUGAN, x16->x4)
guesses letters at that size ("GLIMMLRING FOUIST"), so the HD map erases
them and draws them again with IM Fell English SC (OFL, fonts/OFL.txt) -
an old-style small-caps face close to the map's lettering. Every label uses
one font size (stretching each line into its own measured box made them
visibly different - Thanatos vs Cattan); each line is centred on the
vanilla line with its baseline on the line's bottom edge. The
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
    [("Cattan", (125, 318, 151, 325))],
]

# Placement-only vertical nudge in vanilla px (the box still says what to
# erase): Cattan sat a pixel low against the other labels' spacing.
NUDGE_Y = {"Cattan": -1}


SUPERSAMPLE = 4


def _font_size(cap_px: float) -> int:
    """Font size whose capital height is `cap_px`."""
    f = ImageFont.truetype(str(FONT), 100)
    bb = f.getbbox("H")
    return max(8, round(cap_px * 100 / (bb[3] - bb[1])))


def _glyph_mask(text: str, size: int) -> np.ndarray:
    """Text at font `size` (rendered 4x larger and box-filtered down for
    clean edges), cropped to its ink (alpha 0..1). Small caps have no
    descenders, so the ink bottom is the baseline."""
    f = ImageFont.truetype(str(FONT), size * SUPERSAMPLE)
    bb = f.getbbox(text)
    im = Image.new("L", (bb[2] - bb[0] + 16, bb[3] - bb[1] + 16), 0)
    ImageDraw.Draw(im).text((8 - bb[0], 8 - bb[1]), text, font=f, fill=255)
    im = im.crop(im.getbbox())
    w = max(1, round(im.width / SUPERSAMPLE))
    h = max(1, round(im.height / SUPERSAMPLE))
    return np.asarray(im.resize((w, h), Image.BOX), np.float32) / 255


def reletter(vanilla: Image.Image, hd: Image.Image) -> Image.Image:
    from pipeline import inpaint_colorkey

    van = np.asarray(vanilla.convert("RGB"))
    out = np.asarray(hd.convert("RGB")).astype(np.float32)
    s = hd.width // vanilla.width
    # One size for every label (the vanilla lettering is one size too; the
    # measured boxes only differ by a pixel of anti-aliasing each way):
    # capitals as tall as the median line box, less the half pixel of
    # anti-aliasing the box includes (full height ran Thanatos into its
    # coastline).
    size = _font_size((float(np.median([b[3] - b[1] for label in LABELS for _, b in label])) - 1.0) * s)
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
            mask = _glyph_mask(text, size)
            h, w = mask.shape
            # never wider than the vanilla line (Thanatos ran into its
            # coast): squeeze to the box
            if w > (bx1 - bx0) * s:
                w = (bx1 - bx0) * s
                mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS), np.float32) / 255
            # on the vanilla line: left edge and baseline on its box
            gx = bx0 * s
            gy = by1 * s - h + NUDGE_Y.get(text, 0) * s
            a = mask[..., None]
            region = out[gy:gy + h, gx:gx + w]
            out[gy:gy + h, gx:gx + w] = region * (1 - a) + ink * a
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))
