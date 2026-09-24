from pathlib import Path

ART_CONVERTER_EXE = Path(r"G:\coding\repos\art-converter\art-converter.exe")
REALESRGAN_EXE = Path(r"G:\coding\repos\art-converter\real-esrgan\realesrgan-ncnn-vulkan.exe")
REALESRGAN_MODEL = "4xNomos8kSC"  # best general-purpose pick from 3-model comparison; monster/creature
                                   # sprites looked better with "realesrgan-x4plus" (pass --model to override)

# Free ncnn-vulkan alternative (nihui, same ecosystem as realesrgan above),
# trained on flat-shaded/anime-style 2D art rather than photos. NOT a general
# ESRGAN replacement - a head-to-head comparison across ~14 diverse samples
# found ESRGAN has better detail almost everywhere. Used only as a per-asset
# override for confirmed cases where it clearly wins (see FORCE_CUGAN_ASSETS).
REALCUGAN_EXE = Path(r"G:\coding\repos\art-converter\real-cugan\realcugan-ncnn-vulkan.exe")
REALCUGAN_MODEL_DIR = Path(r"G:\coding\repos\art-converter\real-cugan\models-se")

# rel_paths (as used elsewhere in this pipeline, e.g. "art/eye_candy/blood1_F.ART")
# confirmed by manual inspection to look better through Real-CUGAN than through
# the normal ESRGAN path. No automatic detector exists for this - a roughness/
# size-based classifier couldn't separate blood1_F's mediocre-but-not-noisy
# ESRGAN fallback output from ScrllUP/Combat_But's good one (same size class,
# same fallback path, same roughness range). Same "rare, manually-fixable"
# precedent as the reverted median-filter case in pipeline.py - add to this
# set as more are found rather than trying to auto-detect them.
FORCE_CUGAN_ASSETS = {
    "art/eye_candy/blood1_F.ART",
    "art/interface/Tab_Keys.ART",
    "art/interface/Tab_EgoInjure.ART",
    "art/interface/M_DnBut.ART",
    "art/interface/M_UpBut.ART",
}

ARCANUM_ROOT = Path(r"D:\Galaxy\Games\Arcanum")

# Loose RGB BMPs consumed directly by tig_video_set_hd_overlay() (SDL_LoadBMP,
# not the game's palette-indexed ART/VFS path) - the native-res menu-background
# spike. Path is relative to the game's cwd at runtime (ARCANUM_ROOT), matching
# the literal "hd/MainMenuBack_hd.bmp" the engine currently hardcodes.
HD_OVERLAY_DIR = ARCANUM_ROOT / "hd"

# Loose, already-unpacked dat trees to search for source .ART files, in order.
# Moved under _DAT_UNPACKED/ 2026-09-22 - the old top-level arcanum1-4/tig/
# modules loose folders were removed once _DAT_UNPACKED held verified copies.
_DAT_UNPACKED = ARCANUM_ROOT / "_DAT_UNPACKED"
EXTRACTED_DAT_ROOTS = [
    _DAT_UNPACKED / "arcanum1",
    _DAT_UNPACKED / "arcanum2",
    _DAT_UNPACKED / "arcanum3",
    _DAT_UNPACKED / "Arcanum4",
    _DAT_UNPACKED / "tig",  # tig.dat - 17 .ART files (cursor/button/font chrome)
    _DAT_UNPACKED / "modules" / "Vormantown",  # Vormantown.dat - 0 .ART (townmap/slide BMPs only)
]

# Loose pre-rendered world/town-map BMPs (COLOR_LERP screen-tile path,
# video.c:1007-1091 - a completely separate rendering pipeline from the
# .ART/art_blit() system above, bypassing tig's art format entirely). Not
# consumed by the hd/ .ART walk - these are plain
# RGB/paletted BMPs read directly by whatever loads world-map/town-map
# screens, so upscaling them is a straight image-replace, no unpack/quantize/
# repack/ART-container round-trip needed. Two module trees exist:
WORLDMAP_TOWNMAP_ROOTS = [
    _DAT_UNPACKED / "modules" / "Arcanum" / "Arcanum" / "WorldMap",
    _DAT_UNPACKED / "modules" / "Arcanum" / "Arcanum" / "townmap",
    _DAT_UNPACKED / "modules" / "Vormantown" / "Vormantown" / "townmap",
]

# Townmap only (not WorldMap - different naming/tiling scheme, not yet
# investigated). Each town is a folder of "<Name><NNNNNN>.bmp" tiles (index =
# col + num_hor_tiles*row, a plain row-major grid per TownMapInfo - confirmed
# by reading arcanum-ce's src/game/townmap.c - NOT the diagonal/staggered
# layout the iso world tiles use) plus one "<Name>.tmi" header (48-byte
# TownMapInfo struct: width/height/num_hor_tiles/num_vert_tiles/scale).
# cmd_hd_townmap() assembles each town's populated tiles into one composite
# (cropped to their bounding box, not the full nominal grid - most towns only
# use a fraction of it), upscales that ONCE, then slices back per-tile -
# avoids the independent-per-tile seam problem entirely, since the model sees
# real neighbor content at every former tile boundary (confirmed via a
# seam-check crop straddling a former boundary, hd_pipeline/comparison/
# townmap_seamcheck_*.png - zero visible seam on all 3 non-anime models
# tried). 4xNomos8kSC picked over realesrgan-x4plus/realcugan on authenticity
# (coherent gear/architecture reconstruction, no invented texture pattern -
# realcugan in particular imposed a basket-weave pattern not in the source).
TOWNMAP_ROOTS = [
    _DAT_UNPACKED / "modules" / "Arcanum" / "Arcanum" / "townmap",
    _DAT_UNPACKED / "modules" / "Vormantown" / "Vormantown" / "townmap",
]
TOWNMAP_OUTPUT_DIR = ARCANUM_ROOT / "hd" / "townmap"

# Story-slide BMPs shown by slide_ui.c between chapters / on death / credits -
# plain loose 8-bit-palette BMPs (slide.mes: "slide\<name>.bmp"), not .ART
# sprite sheets. slide_ui.c already tries the same flat "hd/<basename>_hd.bmp"
# convention as the main-menu HD spike (tig_video_set_hd_overlay()) before
# falling back to the vanilla BMP - see slide_ui.c:224-239. Primary/base-game
# module only: Vormantown (community mod) reuses a couple of the same
# basenames (e.g. BatesCastle.bmp) with different art, and the engine's flat
# hd/ overlay path can't disambiguate by module, so cmd_hd_slides() skips any
# Vormantown file whose basename collides with the base game's.
SLIDE_DIR = _DAT_UNPACKED / "modules" / "Arcanum" / "Arcanum" / "Slide"
SLIDE_DIR_VORMANTOWN = _DAT_UNPACKED / "modules" / "Vormantown" / "Vormantown" / "slide"

# Map-load splash BMPs (gamelib_splash(), "art\\splash\\*.bmp" via the real
# VFS, arcanum2.dat) - same loose full-screen BMP shape as the slides above
# (800x400, no .ART unpack needed) and now the same "hd/<basename>_hd.bmp"
# tig_video_set_hd_overlay() convention (wired into gamelib_splash() to
# match slide_ui.c/mainmenu_ui.c).
SPLASH_DIR = _DAT_UNPACKED / "arcanum2" / "art" / "splash"

# Character portraits (portrait.c: "portrait\\<name>.bmp" 64x64 + optional
# "portrait\\<name>_b.bmp" 128x128 "big" dialogue variant, via the real VFS)
# - loose 8-bit BMPs, no .ART unpack needed. User picked REALESRGAN_MODEL
# (4xNomos8kSC) over x4plus/x4plus-anime after a 3-model comparison on
# ELF1_b/HAM1_b/DWM1_b (hd_pipeline/comparison/) - same model as the
# full-screen painted slides/splash/menu, just for a different reason
# (faithful to source palette/shading, less cartoonish than the anime model).
# Drawn inline alongside other composited UI (dialogue, character sheet,
# party bar), not a full-screen overlay swap, so portrait_draw_func() loads
# "hd/<vanilla path>" as a plain higher-res BMP and lets the existing scaled
# blit downsize it - same flat "hd/portrait/<name>[_b].bmp" layout as the
# vanilla VFS path, just prefixed.
PORTRAIT_DIR = _DAT_UNPACKED / "arcanum3" / "portrait"

# Intro/logo Bink videos (SierraLogo.bik, TroikaLogo.bik - the only .bik files
# that exist anywhere in this install, confirmed via undat -l across all 5
# .dat archives). FFMPEG_EXE decodes/re-encodes; REALESRGAN_EXE upscales the
# extracted frames. Compared 4xNomos8kSC/x4plus/realesr-animevideov3-x4 on
# sample frames - visually identical on this flat-shaded logo content, went
# with x4plus.
FFMPEG_EXE = Path(r"G:\coding\repos\art-converter\ffmpeg\bin\ffmpeg.exe")
FFPROBE_EXE = Path(r"G:\coding\repos\art-converter\ffmpeg\bin\ffprobe.exe")
MOVIE_DIR = _DAT_UNPACKED / "arcanum3" / "movies"
MOVIE_MODEL = "realesrgan-x4plus"
# bink_compat (arcanum-ce, first_party/bink_compat) now decodes movies via
# FFmpeg content-probing rather than trusting the .bik extension, so the
# output here can be any container FFmpeg can write (VP9/WebM - BSD-licensed
# libvpx, no GPL/patent-pool baggage) while keeping the original ".bik"
# filename the engine's gmovie_play_path() calls still reference literally.
# tig_movie_play() (first_party/tig/src/movie.c) tries "hd/<path>" before the
# vanilla VFS-relative path, same convention as tig_video_set_hd_overlay()
# and the hd/art/ PNG sidecars - keeps every AI-remastered asset under hd/
# and out of data/, which is reserved for real content overrides/mods.
# (Bink loading has never gone through TIG's file-repository/VFS system -
# BinkOpen resolves its path as a plain relative file open, not a database
# lookup - so this is a straightforward path-prefix probe, not a VFS trick.)
MOVIE_OUTPUT_DIR = ARCANUM_ROOT / "hd" / "movies"

WORK_DIR = Path(__file__).parent / "work"
