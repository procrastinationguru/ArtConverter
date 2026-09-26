"""Face restoration for portraits and the tiny splash-screen faces.

GFPGAN / RestoreFormer (FFHQ-trained, 512x512 in and out) loaded through
spandrel so there's no basicsr/facexlib dependency. No face detector
either: portraits are already one centred face per frame, and the splash
faces are listed by hand in config.SPLASH_FACES. torch is imported lazily
so the rest of the pipeline doesn't need it.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config

_models: dict = {}


def _model(name: str):
    if name not in _models:
        from spandrel import ModelLoader
        m = ModelLoader().load_from_file(str(config.FACE_MODEL_DIR / config.FACE_MODELS[name]))
        _models[name] = m.to("cuda").eval()
    return _models[name]


def restore512(name: str, rgb: np.ndarray) -> np.ndarray:
    """One 512x512 RGB uint8 face through the model."""
    import torch
    t = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None].to("cuda")
    with torch.no_grad():
        o = _model(name)(t)
    if isinstance(o, (tuple, list)):
        o = o[0]
    return (o[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def restore_portrait(src: Image.Image, size: int, name: str = config.PORTRAIT_FACE_MODEL,
                     frac: float = 0.7) -> Image.Image:
    """Whole-frame portrait: shrink into the middle `frac` of a reflect-padded
    512 canvas (portraits are cropped tighter than FFHQ; fed full-frame the
    models tear the face at the frame edges), restore, crop back."""
    n = int(512 * frac) // 2 * 2
    p = (512 - n) // 2
    a = np.asarray(src.convert("RGB").resize((n, n), Image.LANCZOS))
    a = np.pad(a, ((p, p), (p, p), (0, 0)), mode="reflect")
    r = restore512(name, a)[p:p + n, p:p + n]
    return Image.fromarray(r).resize((size, size), Image.LANCZOS)


def patch_splash_faces(stem: str, vanilla: Image.Image, hd: Image.Image) -> Image.Image:
    """Replace the listed faces in an upscaled splash with restored ones,
    feathered in through an ellipse."""
    faces = config.SPLASH_FACES.get(stem)
    if not faces:
        return hd
    s = hd.width // vanilla.width
    out = hd.copy()
    for (cx, cy), half, name in faces:
        box = (cx - half, cy - half, cx + half, cy + half)
        inp = vanilla.convert("RGB").crop(box).resize((512, 512), Image.BICUBIC)
        # soften the palette stair-steps so the model sees a blurry face,
        # not pixel blocks
        inp = inp.filter(ImageFilter.GaussianBlur(1.5))
        r = Image.fromarray(restore512(name, np.asarray(inp))).resize((2 * half * s, 2 * half * s), Image.LANCZOS)
        c = half * s
        mask = Image.new("L", r.size, 0)
        rx, up, down = 0.62 * half * s, 0.87 * half * s, half * s  # chin/beard hangs lower than the brow
        ImageDraw.Draw(mask).ellipse((c - rx, c - up, c + rx, c + down), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(0.16 * half * s))
        out.paste(r, (box[0] * s, box[1] * s), mask)
    return out
