"""Text side of the Darija NeuTTS pipeline: Darija -> IPA, and sample filtering.

Why this file exists
--------------------
`examples/finetune.py` upstream instantiates a bare `phonemizer.backend.EspeakBackend`.
Inference (`NeuTTS._to_phones`) instead goes through `neutts.phonemizers.BasePhonemizer`,
which loads the espeak-ng shared library *bundled with the neutts wheel*. Training with
system espeak and inferring with the bundled one silently produces two different phoneme
distributions. Everything here funnels both sides through one class so that cannot happen.

`DarijaPhonemizer` also registers itself into `neutts.phonemizers.CUSTOM_PHONEMIZERS`
under "ar", which is the same extension point `FrenchPhonemizer` uses. At inference:

    import src.darija_g2p           # noqa: F401  (registers the phonemizer)
    from neutts import NeuTTS
    tts = NeuTTS(backbone_repo="<your-finetuned-repo>", language="ar")

`language="ar"` must be passed explicitly: a finetuned repo will not be in
`BACKBONE_LANGUAGE_MAP`, and `_load_phonemizer` raises if it cannot resolve a code.

When a real Darija G2P is built later, it goes in `preprocess()`/`clean()` here and both
training and inference pick it up with no other changes.
"""

from __future__ import annotations

import importlib.util
import re
import unicodedata
import warnings
from pathlib import Path
from typing import List, Union

# --------------------------------------------------------------------------------------
# Base class: prefer neutts' bundled espeak-ng; fall back to a system build for dev
# --------------------------------------------------------------------------------------
#
# `neutts/__init__.py` does `from .neutts import NeuTTS`, which imports librosa, neucodec,
# torch and transformers. `neutts/phonemizers.py` itself needs only `phonemizer` plus
# stdlib. A plain `from neutts.phonemizers import BasePhonemizer` therefore fails whenever
# any of those heavy deps is broken or absent - and takes the bundled espeak-ng down with
# it, even though the .dylib/.so is sitting in the installed package.
#
# None of those heavy deps are needed to turn text into phonemes, so load the module by
# path instead. `find_spec` locates the package WITHOUT executing its __init__.

_NEUTTS_IMPORT_ERROR: str | None = None


def _load_bundled_phonemizers():
    """Return neutts' phonemizers module, or None. Never raises."""
    global _NEUTTS_IMPORT_ERROR
    try:
        spec = importlib.util.find_spec("neutts")
        if spec is None or not spec.submodule_search_locations:
            _NEUTTS_IMPORT_ERROR = "neutts is not installed"
            return None
        path = Path(spec.submodule_search_locations[0]) / "phonemizers.py"
        if not path.is_file():
            _NEUTTS_IMPORT_ERROR = f"{path} not found"
            return None
        sub = importlib.util.spec_from_file_location("_neutts_phonemizers", path)
        module = importlib.util.module_from_spec(sub)
        sub.loader.exec_module(module)
        return module
    except Exception as exc:  # noqa: BLE001
        _NEUTTS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None


_np = _load_bundled_phonemizers()

if _np is not None and getattr(_np, "_using_bundled_espeak", False):
    BasePhonemizer = _np.BasePhonemizer
    CUSTOM_PHONEMIZERS = _np.CUSTOM_PHONEMIZERS
    USING_NEUTTS_BASE = True
else:
    from phonemizer.backend import EspeakBackend

    USING_NEUTTS_BASE = False
    CUSTOM_PHONEMIZERS = {}

    _detail = _NEUTTS_IMPORT_ERROR or "bundled espeak-ng library not found in the package"
    warnings.warn(
        f"Could not use neutts' bundled espeak-ng ({_detail}). Falling back to a system "
        "espeak-ng, which must be installed separately (macOS: brew install espeak-ng). "
        "Phoneme output may differ from the bundled build. Do NOT generate training data "
        "in this mode.",
        stacklevel=2,
    )

    class BasePhonemizer:  # noqa: D101 - mirrors neutts.phonemizers.BasePhonemizer
        def __init__(self, language_code: str = None):
            self.code = language_code
            if not self.code:
                raise ValueError("A language code must be provided")
            # kwargs kept byte-identical to neutts.phonemizers.BasePhonemizer
            self.g2p = EspeakBackend(
                language=self.code,
                preserve_punctuation=True,
                with_stress=True,
                words_mismatch="ignore",
                language_switch="remove-flags",
            )
            self.espeak_version = self.g2p.version()

        def preprocess(self, text: str) -> str:
            return text

        def clean(self, phonemes: str) -> str:
            return phonemes

        def phonemize(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
            single = isinstance(text, str)
            if single:
                text = [text]
            out = [self.clean(p) for p in self.g2p.phonemize([self.preprocess(t) for t in text])]
            return out[0] if single else out


# --------------------------------------------------------------------------------------
# Normalisation tables
# --------------------------------------------------------------------------------------

# espeak-ng preserves Latin punctuation and silently DROPS Arabic punctuation.
# Measured: "شنو كتدير هنا؟" -> "ʃnˈuː ktdiːr hˈunaː"  (the ؟ is gone).
# Without this mapping the model never learns question or clause intonation.
_PUNCT_MAP = {
    "\u061F": "?",   # ؟  ARABIC QUESTION MARK
    "\u060C": ",",   # ،  ARABIC COMMA
    "\u061B": ";",   # ؛  ARABIC SEMICOLON
    "\u06D4": ".",   # ۔  ARABIC FULL STOP
    "\u2026": ".",   # …  HORIZONTAL ELLIPSIS
    "\u2013": "-",   # –  EN DASH
    "\u2014": "-",   # —  EM DASH
}

# Smart quotes, mirroring neutts.neutts._QUOTE_MAP so both paths agree.
_QUOTE_MAP = {"\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"'}

# Tatweel is an intra-word decorative elongation with no phonetic content, so it is
# deleted outright.
_DELETE = dict.fromkeys([0x0640], None)  # ـ  ARABIC TATWEEL

# Zero-width and bidi marks are extremely common in scraped Arabic text. They are mapped
# to a SPACE rather than deleted: they frequently sit at a word boundary, and deleting one
# welds the neighbours into a single token that espeak then reads as one word
# ("هاد<RLM>الشي" -> "هادالشي" -> hˈaːdaːlʃˌiː instead of hˈaːd ʔaʃʃˈajj).
# The trailing _WS collapse removes any doubled spaces this introduces.
_TO_SPACE = dict.fromkeys(
    [
        0x200B, 0x200C, 0x200D, 0x200E, 0x200F,  # ZWSP/ZWNJ/ZWJ/LRM/RLM
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embedding/override
        0x2066, 0x2067, 0x2068, 0x2069,          # bidi isolates
        0x00A0,                                   # NBSP
        0xFEFF,                                   # BOM
    ],
    " ",
)

_TRANSLATION = str.maketrans({**_PUNCT_MAP, **_QUOTE_MAP, **_DELETE, **_TO_SPACE})

_WS = re.compile(r"\s+")

# Arabic letters incl. Arabic Supplement / Extended-A; excludes digits and punctuation.
_ARABIC_LETTER = re.compile(r"[\u0620-\u064A\u0671-\u06D3\u06FA-\u06FF\u0750-\u077F]")

# Any Unicode decimal digit. Python's \d matches Arabic-Indic digits (٠-٩) too.
_ANY_DIGIT = re.compile(r"\d")

_LATIN_RUN = re.compile(r"[A-Za-z]{2,}")


# --------------------------------------------------------------------------------------
# Phonemizer
# --------------------------------------------------------------------------------------


class DarijaPhonemizer(BasePhonemizer):
    """Moroccan Darija (Arabic script) -> IPA via espeak-ng's `ar` voice.

    Baseline only. espeak-ng's `ar` applies Modern Standard Arabic morphophonology and
    reads short vowels from diacritics that undiacritised Darija does not carry, so its
    output is MSA-flavoured and frequently vowel-less:

        نبداو      -> mbdˈaːw     (Darija /nəbdaw/: ن read as m, schwa absent)
        اليوم      -> ʔaljˈawm    (Darija /lyum/: MSA definite article)
        فالسبعة    -> fassbʕt     (five consonants, no vowel at all)

    It is deterministic and, on the minimal pairs tested, collision-free, so it is a
    usable baseline: the model can learn a consistent mapping even from wrong IPA.
    Replacing it is the job of `preprocess`/`clean` in a subclass or in this class.
    """

    def __init__(self, language_code: str = "ar", strip_diacritics: bool = True):
        super().__init__(language_code)
        self.strip_diacritics = strip_diacritics

    def preprocess(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(_TRANSLATION)
        if self.strip_diacritics:
            # Harakat appear in under 1% of MoulSot transcripts (689 combining marks
            # across 79,641 rows, mostly shadda). espeak-ar reads them for vowelisation,
            # so leaving them in means that minority gets phonemised on a different
            # basis from everything else - and text typed at inference will not carry
            # them either. Stripping makes training consistent with both.
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return _WS.sub(" ", text).strip()

    def clean(self, phonemes: str) -> str:
        # Collapse whitespace so training output matches NeuTTS._to_phones exactly,
        # which does phones.split() then " ".join(phones).
        return _WS.sub(" ", phonemes).strip()


# Register under the espeak language code the inference path resolves.
CUSTOM_PHONEMIZERS["ar"] = DarijaPhonemizer


def to_phones(phonemizer: BasePhonemizer, text: str) -> str:
    """Behaviourally identical to `NeuTTS._to_phones`. Use this, not raw phonemize()."""
    phones = phonemizer.phonemize([text])
    return " ".join(phones[0].split())


# --------------------------------------------------------------------------------------
# Sample filter
# --------------------------------------------------------------------------------------
#
# Upstream `data_filter` rejects any text whose last character is not in ".,?!", which
# discards essentially every Darija transcript (Arabic ؟ and ، are different codepoints,
# and segment-final punctuation is often absent). It also runs an uppercase-acronym regex
# that can never match Arabic script. Rewritten here around what actually matters.


def filter_reason(
    text: str,
    n_speech_tokens: int,
    max_speech_tokens: int,
    min_speech_tokens: int = 50,
    reject_latin_runs: bool = False,
) -> str | None:
    """Return None to keep the sample, else a short reason string.

    min_speech_tokens defaults to 50 (1.0 s) to match `min_new_tokens=50` in
    `NeuTTS._infer_torch` - the model is never asked to produce less than that.
    """
    if text is None:
        return "null_text"

    norm = _WS.sub(" ", unicodedata.normalize("NFKC", text).translate(_TRANSLATION)).strip()
    if not norm:
        return "empty_text"

    if not _ARABIC_LETTER.search(norm):
        return "no_arabic_letters"

    # espeak verbalises numerals into MSA ("250" -> mˌiʔatˈaːnwa xamsˈuːn) while the
    # audio contains the Darija form. Guaranteed text/audio mismatch, so drop it.
    if _ANY_DIGIT.search(norm):
        return "contains_digits"

    if reject_latin_runs and _LATIN_RUN.search(norm):
        return "latin_run"

    if n_speech_tokens < min_speech_tokens:
        return "too_short"
    if n_speech_tokens > max_speech_tokens:
        return "too_long"

    return None