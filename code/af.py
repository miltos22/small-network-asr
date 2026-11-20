"""Module 2 labels: ARPAbet phone -> 22 articulatory features (standard phonology), substates, and per-detector input masks.
Every detector reads ONLY the tracks listed for it (its physical evidence) and is calibrated ONLY on its own feature label."""
import numpy as np

PHONES = "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split()
FEATS = ["voiced", "aspirated", "sonorant", "syllabic", "consonantal", "continuant", "nasal", "strident", "lat_rhotic",
         "labial", "dental", "alveolar", "postalveolar", "velar", "glottal", "high", "low", "front", "back", "rounded",
         "tense", "glide"]
VOW = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
S = lambda s: set(s.split())
MEMBERS = {
    "voiced": VOW | S("B D G DH V Z ZH JH M N NG L R W Y"),
    "aspirated": S("P T K CH HH"),
    "sonorant": VOW | S("M N NG L R W Y"),
    "syllabic": VOW,
    "consonantal": S("P B T D K G CH JH F V TH DH S Z SH ZH M N NG L"),
    "continuant": VOW | S("F V TH DH S Z SH ZH HH L R W Y"),
    "nasal": S("M N NG"),
    "strident": S("S Z SH ZH CH JH"),
    "lat_rhotic": S("L R ER"),
    "labial": S("P B M F V W"),
    "dental": S("TH DH"),
    "alveolar": S("T D N S Z L"),
    "postalveolar": S("SH ZH CH JH R"),
    "velar": S("K G NG W"),
    "glottal": S("HH"),
    "high": S("IY IH UW UH Y W"),
    "low": S("AA AE"),
    "front": S("IY IH EY EH AE Y"),
    "back": S("AA AO UW UH OW W"),
    "rounded": S("UW UH OW AO OY W"),
    "tense": S("IY EY UW OW"),
    "glide": S("AY AW OY EY OW W Y"),
}
# diphthongs: vowel-quality features follow the start target in the first half, the end target in the second half
DIPH = {"AY": ("AA", "IY"), "AW": ("AA", "UH"), "OY": ("AO", "IY"), "EY": ("EH", "IY"), "OW": ("AH", "UH")}
QUAL = {"high", "low", "front", "back", "rounded"}


def vec(ph, half=0):
    v = np.zeros(len(FEATS), np.int8)
    q = DIPH[ph][half] if ph in DIPH else ph
    for i, f in enumerate(FEATS):
        v[i] = (q if f in QUAL else ph) in MEMBERS[f]
    return v


# per-detector evidence (tracks; each enters as 4 Legendre coefficients over +-80 ms)
FORM = ["B1", "B2", "B3", "W1", "W2", "W3"]
MASK = {
    "voiced": ["Elow", "R", "F0", "Erel", "C1", "H"],
    "aspirated": ["Ehigh", "R", "flux", "Erel", "H", "g_vert_hi"],
    "sonorant": ["R", "Erel", "Elow", "Emid", "C1", "C2", "H", "B1"],
    "syllabic": ["Erel", "R", "B1", "W1", "C1", "Emid", "H"],
    "consonantal": ["Erel", "B1", "flux", "C1", "H", "R", "Emid"],
    "continuant": ["Erel", "flux", "Ehigh", "R", "H", "g_vert_lo", "g_vert_hi"],
    "nasal": ["Elow", "Emid", "B1", "W1", "B2", "C1", "C2", "Erel", "R"],
    "strident": ["Ehigh", "BH", "WH", "H", "C1", "Erel"],
    "lat_rhotic": ["B2", "B3", "W3", "C3", "C4", "R", "Emid"],
    "labial": ["B2", "BH", "C1", "C2", "Ehigh", "g_rise_lo", "g_fall_lo"],
    "dental": ["Ehigh", "BH", "WH", "H", "Erel", "C1"],
    "alveolar": ["B2", "B3", "BH", "Ehigh", "C2", "g_rise_hi", "g_fall_hi"],
    "postalveolar": ["BH", "WH", "B3", "B2", "Ehigh", "C3"],
    "velar": ["B2", "B3", "C2", "C3", "Ehigh", "g_rise_hi", "g_fall_hi"],
    "glottal": ["H", "R", "Ehigh", "C1", "Erel", "Elow"],
    "high": FORM + ["C1", "C2", "C3", "R"],
    "low": FORM + ["C1", "C2", "C3", "R"],
    "front": FORM + ["C2", "C3", "C4", "R"],
    "back": FORM + ["C2", "C3", "C4", "R"],
    "rounded": FORM + ["C1", "C3", "C4", "C5", "R"],
    "tense": FORM + ["C2", "C3", "Erel", "R"],
    "glide": FORM + ["C2", "C3", "C4"],
}
ALL_TRACKS = sorted({t for v in MASK.values() for t in v})
STOPS = S("P B T D K G CH JH")
