from pathlib import Path

ART_CONVERTER_EXE = Path(r"G:\coding\repos\art-converter\art-converter.exe")
REALESRGAN_EXE = Path(r"G:\coding\repos\art-converter\real-esrgan\realesrgan-ncnn-vulkan.exe")
REALESRGAN_MODEL = "4xNomos8kSC"  # best general-purpose pick from 3-model comparison; monster/creature
                                   # sprites looked better with "realesrgan-x4plus" (pass --model to override)

ARCANUM_ROOT = Path(r"D:\Galaxy\Games\Arcanum")
DATA_OVERLAY_DIR = ARCANUM_ROOT / "data"

# Loose RGB BMPs consumed directly by tig_video_set_hd_overlay() (SDL_LoadBMP,
# not the game's palette-indexed ART/VFS path) - the native-res menu-background
# spike. Path is relative to the game's cwd at runtime (ARCANUM_ROOT), matching
# the literal "hd/MainMenuBack_hd.bmp" the engine currently hardcodes.
HD_OVERLAY_DIR = ARCANUM_ROOT / "hd"

# Loose, already-unpacked dat trees to search for source .ART files, in order.
EXTRACTED_DAT_ROOTS = [
    ARCANUM_ROOT / "arcanum1",
    ARCANUM_ROOT / "arcanum2",
    ARCANUM_ROOT / "arcanum3",
    ARCANUM_ROOT / "Arcanum4",
    ARCANUM_ROOT / "tig",  # tig.dat, unpacked 2026-09-20 - 17 .ART files (cursor/button/font chrome)
    ARCANUM_ROOT / "modules" / "Vormantown",  # Vormantown.dat, unpacked 2026-09-20 - 0 .ART (townmap/slide BMPs only)
]

# Loose pre-rendered world/town-map BMPs (COLOR_LERP screen-tile path,
# video.c:1007-1091 - a completely separate rendering pipeline from the
# .ART/art_blit() system above, bypassing tig's art format entirely). Not
# consumed by cmd_run()/EXTRACTED_DAT_ROOTS's .ART walk - these are plain
# RGB/paletted BMPs read directly by whatever loads world-map/town-map
# screens, so upscaling them is a straight image-replace, no unpack/quantize/
# repack/ART-container round-trip needed. Two module trees exist:
WORLDMAP_TOWNMAP_ROOTS = [
    ARCANUM_ROOT / "modules" / "Arcanum" / "WorldMap",
    ARCANUM_ROOT / "modules" / "Arcanum" / "townmap",
    ARCANUM_ROOT / "modules" / "Vormantown" / "townmap",
]

WORK_DIR = Path(__file__).parent / "work"
