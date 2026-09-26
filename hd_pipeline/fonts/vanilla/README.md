# Vanilla font arts -> TrueType

The game's text fonts are pre-rendered bitmap arts (`art/interface/*Font.ART`,
unpacked under `work/`). This folder holds the TTF/OTF each one was rendered
from, or the closest freely licensed stand-in. Font files here are gitignored:
the licences are mixed (system fonts may not be redistributed).

| Font art | Typeface | File here | Source / licence |
|---|---|---|---|
| arial10font, ArialB12Font | Arial / Arial Bold | arial.ttf, arialbd.ttf | Windows system copy |
| NewTimes16Font | Times New Roman | times.ttf | Windows system copy |
| Courier10Font | Courier New | cour.ttf | Windows system copy |
| Comic12Font | Comic Sans MS | comic.ttf | Windows system copy |
| Georgia30Font | Georgia | georgia.ttf | Windows system copy |
| morph15font, Morph30Font | Morpheus (Kiwi Media, 1996) | MORPHEUS.TTF | dafont; shareware ($5 to the author), see Morpheus_README.txt |
| Cloister18Font | Cloister Black | CloisterBlack.ttf | dafont; Dieter Steffmann, free incl. commercial |
| Garmond6/8/9Font | Garamond | EBGaramond.ttf (stand-in) | Google Fonts, OFL |
| BookmanOldBold18Font | Bookman Old Style Bold | texgyrebonum-bold.otf (stand-in) | CTAN TeX Gyre Bonum, GUST font licence |
| Swiss921Font | Bitstream Swiss 921 | Anton-Regular.ttf (stand-in) | Google Fonts, OFL |
| CasablancaAntique30Font, casablanca16font | Corel Casablanca Antique (= Caslon Antique) | IMFeENrm28P.ttf (IM Fell English, stand-in) | Google Fonts, OFL |
| Zurich16/20Font | Bitstream Zurich (= Univers), condensed | ArchivoNarrow[wght].ttf at wght 700 (stand-in) | Google Fonts, OFL |
| LatinXCN30Font | Latin Extra Condensed (caps only) | StintUltraCondensed-Regular.ttf, upper case (stand-in) | Google Fonts, OFL |
| ClarendonBLK18Font | Clarendon Black | Coustard-Black.ttf (stand-in) | Google Fonts, OFL |
| Flare12/14Font | Corel Flareserif 821 (= Albertus), bold | AlegreyaSans-ExtraBold.ttf (stand-in) | Google Fonts, OFL |
| Elga12Font | unidentified heavy old-style serif | CrimsonPro[wght].ttf at wght 800 (stand-in) | Google Fonts, OFL |
| Euph30Font | unidentified condensed Victorian serif | Grenze[wght].ttf at wght 700 (stand-in) | Google Fonts, OFL |
| Nick16Font | unidentified, Nicolas Cochin style | LindenHill-Regular.ttf (stand-in) | Google Fonts, OFL |
| Pepper20Font | unidentified pen italic | Fondamento-Italic.ttf (stand-in) | Google Fonts, OFL |
| pork12font | unidentified rough antique serif | IMFePIrm28P.ttf (IM Fell DW Pica, stand-in) | Google Fonts, OFL |

Stand-ins were picked by eye against the vanilla glyphs:
comparison/font_lookalikes/<font art>.png (vanilla at x4 nearest above,
the stand-in below).

Image sheets, not typefaces: BookImagesFont, NewsIconsFont, Icons17/32Font,
MPIconsFont, rollerfont.
