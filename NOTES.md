./realesrgan-ncnn-vulkan.exe -i ./mspuwxaa_00.bmp -o ./mspuwxaa_00_4k.png

# hd_pipeline/pipeline.py subcommands (see arcanum-ce/docs/HD_ART_PIPELINE.md for the "why")
python pipeline.py hd-slides            # death/chapter/credits story slides -> hd/slides/
python pipeline.py hd-splash            # map-load splash screens -> hd/splash/
python pipeline.py hd-portraits         # character portraits -> hd/portrait/
python pipeline.py hd-movies            # SierraLogo/TroikaLogo Bink videos -> hd/movies/
python pipeline.py hd-overlay <rel_path>  # single main-menu-style background -> hd/menu/
