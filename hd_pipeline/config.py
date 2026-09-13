from pathlib import Path

ART_CONVERTER_EXE = Path(r"G:\coding\repos\art-converter\art-converter.exe")
REALESRGAN_EXE = Path(r"G:\coding\repos\art-converter\real-esrgan\realesrgan-ncnn-vulkan.exe")
REALESRGAN_MODEL = "4xNomos8kSC"  # best general-purpose pick from 3-model comparison; monster/creature
                                   # sprites looked better with "realesrgan-x4plus" (pass --model to override)

ARCANUM_ROOT = Path(r"D:\Galaxy\Games\Arcanum")
DATA_OVERLAY_DIR = ARCANUM_ROOT / "data"

# Loose, already-unpacked dat trees to search for source .ART files, in order.
EXTRACTED_DAT_ROOTS = [
    ARCANUM_ROOT / "arcanum1",
    ARCANUM_ROOT / "arcanum2",
    ARCANUM_ROOT / "arcanum3",
    ARCANUM_ROOT / "Arcanum4",
]

WORK_DIR = Path(__file__).parent / "work"
