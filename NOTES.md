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
python pipeline.py hd-schem-tone [--only X]   # match schematic drawings' paper tone to Schematic_Base (pass 11); run after re-upscaling drawings
python pipeline.py hd-schem-base-edge          # Schematic_Base: opaque ring round the drawing's hole (pass 12 #40); run after re-upscaling Schematic_Base
python pipeline.py hd-htft-knob [--only X]    # HP/fatigue -/+ sidecars (Char_HTFTPlus/Minus) from Char_Maint's knobs + Char_Plus/Minus signs (pass 12 #41); run after re-upscaling those
python pipeline.py hd-saveload-chain           # Loot panel's chain painted into the SaveLoadBackground / Scheme_Rot scroll tracks (pass 13 #47/#52)
python pipeline.py hd-nav-pill-rim             # worldmap bottom plate pills (MapMain + Nav_Cvr): even rim round the groove (pass 13 #48)
python pipeline.py hd-skill-gauge              # Skills_Window: glass tube filled, rail carried to the bracket, 1..5 redrawn crisp (pass 13 #59/#65)
python pipeline.py hd-cycle-arrows [--only X] # char creation arrow buttons: clean disc, vector arrows, socket-centred (pass 13 #62)
python pipeline.py hd-scroll-chains            # Loot/Barter/Barter_Follower: vanilla chain painted out, repainted between the scroll arrows, centred on them (pass 13 #89)
python pipeline.py hd-disc-mask [--only X]    # cut round buttons (DISC_MASKS) to their disc
