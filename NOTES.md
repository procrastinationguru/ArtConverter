./realesrgan-ncnn-vulkan.exe -i ./mspuwxaa_00.bmp -o ./mspuwxaa_00_4k.png

# hd_pipeline/pipeline.py subcommands (see arcanum-ce/docs/HD_ART_PIPELINE.md for the "why")
python pipeline.py hd-slides            # death/chapter/credits story slides -> hd/slides/
python pipeline.py hd-splash            # map-load splash screens -> hd/splash/
python pipeline.py hd-portraits         # character portraits -> hd/portrait/
python pipeline.py hd-movies            # SierraLogo/TroikaLogo Bink videos -> hd/movies/
python pipeline.py hd-overlay <rel_path>  # single main-menu-style background -> hd/menu/

# Fonts / lens / splash text (round 8; see arcanum-ce/docs/ROUND8_PLAN.md Pass 7, ARCHITECTURE.md "HD text (fonts)")
python pipeline.py hd-fonts [--only X]        # per-glyph sidecars for all bitmap fonts (MAIN_FONT = Outfit)
python pipeline.py hd-font-faces [--only X]   # MAIN_FONT faces: hd/art/<font>/face.txt + face/f<N>.png (needs uharfbuzz); run after hd-fonts
python pipeline.py hd-lens-rings              # smooth PC lens ring hole edges
python pipeline.py hd-lens-corners            # ring corners from panel wood + LENS_ALIAS copies (Lns_Map, Lns_Schm); run after hd-lens-rings
python pipeline.py hd-splash-text [--only X]  # rebuild "Loading Arcanum..." from the vanilla letter shapes (needs opencv); backup in work/_splash_text_originals
python pipeline.py hd-lens-context [--only X] # inventory/barter/loot lens ring upscaled in context with its panel (round 8 pass 11); run after re-upscaling those arts
python pipeline.py hd-lens-unify              # the inventory's (smoothest) gold ring on every PC lens + its panels; run after all other lens/panel steps
python pipeline.py hd-schem-tone [--only X]   # match schematic drawings' paper tone to Schematic_Base (pass 11); run after re-upscaling drawings
python pipeline.py hd-schem-base-edge          # Schematic_Base: opaque ring round the drawing's hole (pass 12 #40); run after re-upscaling Schematic_Base
python pipeline.py hd-htft-knob [--only X]    # HP/fatigue -/+ sidecars (Char_HTFTPlus/Minus) from Char_Maint's knobs + Char_Plus/Minus signs (pass 12 #41); run after re-upscaling those
python pipeline.py hd-saveload-chain           # Loot panel's chain painted into the SaveLoadBackground / Scheme_Rot scroll tracks (pass 13 #47/#52)
python pipeline.py hd-nav-pill-rim             # worldmap bottom plate pills (MapMain + Nav_Cvr): groove re-mapped to one depth all round, seam softened, right pin redrawn, top edge lit left of the globe (#48, 2026-10-01); rewrites MapMain - rerun hd-lens-unify after
# upscale safety net (cmd_hd, hd-scan --fix, hd-palettes): output rougher than 20 is redone with remacri (soup_fallback); Real-CUGAN only if still > 32 (real static) - it used to be CUGAN straight away, which washed out busy textured icons
python pipeline.py hd-item-remacri [--only X] [--force] # items (icons, ground, paperdoll): remacri inside, default model's outline (6 HD px), the sharper one per item kept; current frames backed up in work/_item_default/
python pipeline.py hd-creature-remacri [critter monster unique_npc] [--only X] # creatures: remacri + default-model outline for all (no sharper-wins rule); default frames kept in work/_<cat>_default/
python pipeline.py hd-remacri-revert <category> <name...> # put arts (e.g. critter dfm/dfmbnsad) back on the default model from that backup
python pipeline.py hd-dither-buttons [--only X] # checkerboard-shaded buttons (DITHER_BUTTONS): 50/50 average upscaled (ultrasharp-4x), hand-drawn HD checker as soft dots; round ones (DITHER_FACE) get the whole face rebuilt, the icon centred at HD precision on the face circle (FACE_CIRCLE = the host panel's ring for panel-cut buttons; one shift from the lit up/hover icon body, same for down = vanilla press shift kept); resting key = hover key x KEY_UP_DIM; the shield key button = HUD key on a rebuilt convex shield (DITHER_KEY_DONOR); HP/fatigue +/- = their vanilla 21x21 art on the knob face; base in work/_dither_originals/ -> then hd-round-buttons -> hd-button-face
python pipeline.py hd-button-face [--only X] # panel-cut buttons (BUTTON_FACE_ONLY: skill +/-, slider arrows, clear/cancel/done, map-note, ...): only the face opaque so the host panel's own ring shows in every state; run after hd-round-buttons
python pipeline.py hd-round-buttons [--only X] # round buttons: rim centred + smoothed along the circle, AA circle outline (ROUND_BUTTONS, 68 arts) + painted copies in 21 panels (ROUND_PANEL_SPOTS, 86) + map buttons (ROUND_ART_SPOTS); ROUND_MERGE (+/- and slider arrows) also get their doubled copper ring merged into one; run LAST - it backs up whatever the panels are at first run
python pipeline.py hd-hud-touchup              # IntBotom: hotbar red knob glow back (vanilla red), panel edge jog right of it straightened; run after hd-arc-smooth
python pipeline.py hd-mapmain-patch            # worldmap coordinate boxes: vanilla paste seam filled with the sidecar's own wood from below (#173); after hd-nav-pill-rim is fine, it patches that backup too
python pipeline.py hd-side-bars                # widescreen side bars (hd/art/interface/_SideBars): left/right art per width - IntTop-style metal plates over MPChatBackground's carved knot, the char creation box's gold frame on the game edge, wood re-toned to mid (SIDE_BAR_WOOD_*); see arcanum-ce docs/VIDEO_OPTIONS_SIDEBARS.md
python pipeline.py hd-arc-smooth               # stair-stepped arcs smoothed along the circle (ARC_SMOOTH: IntBotom hotbar ends); after hd-black-fill
python pipeline.py hd-skill-gauge              # Skills_Window: vanilla glass kept, shaded row by row to vanilla on screen (volume), rail carried to the bracket, 1..5 redrawn crisp; SkilGauge liquid top/bottom edges faded (pass 13 #59/#65/#94/#97)
python pipeline.py hd-cycle-arrows [--only X] # char creation arrow buttons: clean disc, vector arrows, socket-centred (pass 13 #62)
python pipeline.py hd-scroll-chains            # Loot/Barter/Barter_Follower: vanilla chain painted out, repainted between the scroll arrows, centred on them (pass 13 #89)
python pipeline.py hd-disc-mask [--only X]    # cut round buttons (DISC_MASKS) to their disc
