# Key button redesign (unfinished, 2026-10-01)

Scratch scripts from the session that tried to replace the key on the two
key buttons (`Skills_Button` on the HUD, `char_Common_Skills` with the red
shield on the character screen). Nothing here is wired into `pipeline.py`
yet; the live art is still what `hd-dither-buttons` makes (vanilla-shaped
key, `KEY_OUTLINE` darkening). Run the scripts from this folder; they write
comparison sheets (`python keymock6.py out.png`).

## What the user asked for

- Silhouette: the skeleton key in `ref_skeleton_key.png`, laid sideways
  like vanilla's (bow left, bit hanging down at the right end), details
  simplified (no collar eye, no neck rings, no bit holes).
- `ref_user_cuts.png`: concave cuts top and bottom at the neck between the
  bow and the collar, and a V notch in the bit's bottom edge.
- Painted in vanilla's yellow with the checker dots, lit like vanilla in
  each frame (rest dim, hover bright with the light band along the shaft).
- `ref_vanilla_key_rim_shading.png`: vanilla's key is darker toward its
  edges and bright in the middle; replicate that gradient.
- Outline: soft, not hard black pixels, and it must not wipe out the
  face's reflection and shading (darken what is underneath, don't paint).
- Shield variant: keep vanilla's exact shield shape, crisp, with its
  checker; no leftover dark lines from the old key.
- Face behind the key: vanilla's colour and dome shading (lighter lower
  left, darker toward the rim), the rim's HD highlight kept.

## What went wrong along the way (don't repeat)

- Copying vanilla's 1x colours (bicubic-upscaled) onto the HD key or face
  looks blurry and "pasted from a smaller button" (user's words).
- The vanilla key's checker difference is clamped by `_dg_decompose`
  (|half| ~64 when the real bright/dark cells are 243/70): measure the
  even/odd cells directly (`keymock6.py`).
- `_dg_keymask` misses the dim resting key of `char_Common_Skills`; take
  the union of all frames' masks.
- An outline that paints dark colour (or an inner bevel) kills the face's
  reflection near the key.
- The segmented reference outline is wobbly; straighten the shaft.

## Latest state (`keymock6.py`, sheet AD/AE/AF)

Face and shield are low-order polynomial fits to vanilla's 1x colours,
evaluated at HD (smooth, no 1x blur); shield edge from its convex hull at
HD; checker crisp via `_dg_checker`; key cells = the HUD key's even/odd
medians per frame x vanilla's light profile along the key x an edge
darkening (`rim_dark`, `rim_w`). Still wrong: the bow ring reads darker than
the shaft (thin, so the edge darkening covers it); the user was not happy
with the overall result. Next session: get the user's pick on a smaller
set of options before iterating, and then port the winner into
`cmd_hd_dither_buttons` (both key buttons, all frames).
