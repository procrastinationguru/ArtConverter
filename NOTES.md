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
python pipeline.py hd-splash-text [--only X]  # re-typeset "Loading Arcanum..." on splashes (needs opencv); backup in work/_splash_text_originals
