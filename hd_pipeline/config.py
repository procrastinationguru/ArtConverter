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
]

WORK_DIR = Path(__file__).parent / "work"
