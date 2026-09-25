"""Name and address normalization.

Every rule here is a hand-written language rule from CLAUDE.md, applied by matching the
*text*. Nothing branches on the `country` column: France is unseen in training, so a
country-conditioned rule would silently not apply to a third of the test set. Postal
codes, landmarks and street words are therefore detected by pattern only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from unidecode import unidecode

# --- dictionaries (CLAUDE.md) ------------------------------------------------

LEGAL_SUFFIXES = {
    "pvt", "private", "ltd", "limited", "llp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "llc", "plc", "sarl", "sas", "sa", "eurl",
    "snc", "sci", "gmbh",
}

# Applied to address tokens regardless of position.
ADDRESS_ABBREV = {
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard",
    "nr": "near",
    "opp": "opposite",
    "bldg": "building", "bldgs": "building",
    "flr": "floor", "fl": "floor",
    "pl": "place",
    "chem": "chemin",
    "imp": "impasse",
    "fbg": "faubourg",
    "sq": "square",
    "apt": "apartment",
    "no": "number", "nos": "number",
    "ph": "phase",
    "dist": "district",
    "tq": "taluk",
    "kh": "khasra",
    "r": "rue",
}

# Tokens that carry no identity signal in a business name.
NAME_STOPWORDS = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "el"}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_DIGITS = re.compile(r"\d")
# 5-6 digit runs cover US ZIP (5), France code postal (5) and India PIN (6).
_POSTAL = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_NUMBER = re.compile(r"\d+")
_TOKEN = re.compile(r"[a-z0-9]+")
# "s.a.s" / "l.l.c." -> one token, so the legal-suffix dictionary can match it.
_DOTTED_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\b")


def _ascii_fold(s: str) -> str:
    """Transliterate to ASCII. Handles accents and Devanagari alike."""
    return unidecode(s)


def _collapse_acronyms(s: str) -> str:
    """Join dotted acronyms: 'fractales s.a.s' -> 'fractales sas'."""
    return _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)


def _merge_single_letters(tokens: list[str]) -> list[str]:
    """Join runs of 2+ single-letter tokens: ['s','a','s'] -> ['sas'].

    Catches acronyms written with spaces instead of dots. A lone single letter between
    multi-letter tokens is left alone, so "B and C" does not become "bandc".
    """
    out: list[str] = []
    run: list[str] = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        out.append(t)
    if len(run) >= 2:
        out.append("".join(run))
    else:
        out.extend(run)
    return out


@lru_cache(maxsize=1 << 18)
def normalize_name(s: str) -> str:
    """Lowercase ASCII name with legal suffixes and punctuation removed."""
    if not s:
        return ""
    return split_name(s).core


@dataclass(frozen=True)
class NameParts:
    """A business name split into the identity-bearing part and its legal suffixes."""

    core: str
    suffixes: tuple[str, ...]

    @property
    def legal_suffix(self) -> str:
        return " ".join(self.suffixes)

    def __bool__(self) -> bool:
        return bool(self.core)


@lru_cache(maxsize=1 << 18)
def split_name(s: str) -> NameParts:
    """Split into core_name + legal_suffix using CLAUDE.md's dictionary.

    Suffixes are recognised anywhere, not just at the end: this data has "LLC Moncada
    Learning Center" and "OF India Pvt Ltd" alike. If every token is a suffix the name is
    kept whole as the core, since dropping it would erase the record's only signal.
    """
    if not s:
        return NameParts("", ())
    t = _ascii_fold(s).lower().replace("&", " and ")
    t = _collapse_acronyms(t)
    t = _PUNCT.sub(" ", t)
    tokens = _merge_single_letters([x for x in _WS.split(t) if x])
    core = [x for x in tokens if x not in LEGAL_SUFFIXES]
    sufs = tuple(x for x in tokens if x in LEGAL_SUFFIXES)
    if not core:
        return NameParts(" ".join(tokens), sufs)
    return NameParts(" ".join(core), sufs)


def core_name(s: str) -> str:
    """The name with legal suffixes removed -- what we index and compare."""
    return split_name(s).core


def name_tokens(s: str) -> set[str]:
    """Identity-bearing name tokens: no legal suffixes, no stopwords, no 1-char noise."""
    norm = normalize_name(s)
    return {t for t in _TOKEN.findall(norm) if len(t) > 1 and t not in NAME_STOPWORDS}


def name_sort_key(s: str) -> str:
    """Tokens sorted alphabetically — cancels word-order transpositions."""
    return " ".join(sorted(name_tokens(s)))


def _expand_st(tokens: list[str]) -> list[str]:
    """'st' is 'street' at the end of a comma segment, 'saint' before a name.

    "105 ELM ST, MORGANTON" -> street;  "ST LOUIS, MO" -> saint.
    """
    out = []
    for i, t in enumerate(tokens):
        if t != "st":
            out.append(t)
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        prev = tokens[i - 1] if i else None
        if nxt is None or nxt == ",":
            out.append("street")
        elif prev is None or prev == ",":
            out.append("saint")
        else:
            out.append("street")
    return out


@lru_cache(maxsize=1 << 18)
def normalize_address(s: str) -> str:
    """Lowercase ASCII address with abbreviations expanded. Empty input stays empty."""
    if not s:
        return ""
    s = _ascii_fold(s).lower()
    s = s.replace("&", " and ")
    s = _collapse_acronyms(s)
    # Keep commas as segment markers while the 'st' rule needs them.
    s = re.sub(r"[^\w\s,]", " ", s)
    s = re.sub(r",", " , ", s)
    tokens = [t for t in _WS.split(s) if t]
    tokens = _expand_st(tokens)
    tokens = [ADDRESS_ABBREV.get(t, t) for t in tokens]
    tokens = [t for t in tokens if t != ","]
    # "cedex" is a French delivery marker, not part of the identity.
    tokens = [t for t in tokens if t != "cedex"]
    return " ".join(tokens)


def address_tokens(s: str) -> set[str]:
    norm = normalize_address(s)
    return {t for t in _TOKEN.findall(norm) if len(t) > 1}


_NUM_RE = re.compile(r"(?<![0-9])([0-9]{3} [0-9]{3}|[0-9]{5,6})(?![0-9])")
_UNIT_BEFORE = re.compile(
    r"(unit|apt|apartment|suite|ste|flat|plot|bldg|building|box|bp|#|no\.?|h\.?\s?no\.?)"
    r"\s*[:#-]?\s*$", re.I)
# Street-type words, after ADDRESS_ABBREV expansion. A digit run followed by one of these
# is a house number, not a postal code.
STREET_WORDS = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "court", "place", "square",
    "way", "trail", "circle", "terrace", "parkway", "highway", "close", "walk", "row",
    "rue", "chemin", "impasse", "faubourg", "allee", "quai", "route", "voie", "passage",
    "marg", "nagar", "colony", "sector", "phase", "block", "cross", "main", "galli", "gali",
}


def _segment_tail(s: str, end: int) -> str:
    """Text from `end` to the next comma -- the rest of this address component."""
    nxt = s.find(",", end)
    return s[end:] if nxt < 0 else s[end:nxt]


def postal_codes_in(addr: str) -> dict[str, str]:
    """Postal-code candidates as {code: kind}, by pattern and position only.

    Most 5/6-digit runs in this data are house numbers. The discriminator, verified
    against real records in p1c, is what FOLLOWS the number inside its comma segment:

        "11244 Westfall Road"  -> street word follows      -> house number
        "11940 72"             -> only digits follow       -> house + road number
        "33000 BORDEAUX"       -> a plain word follows     -> postal code + city
        "59100, ROUBAIX"       -> segment ends             -> postal code

    plus positional rules for unit numbers ("Unit 10207"), zero-padded house numbers
    ("00109", "003801"), a run opening the whole address ("11024, S/F, Gali Peepal Wali"),
    and "ddd ddd" house+road pairs ("602 723").

    Never conditioned on the country label: one rule set covers ZIP, PIN and code postal.
    """
    s = (addr or "").strip()
    out: dict[str, str] = {}
    for m in _NUM_RE.finditer(s):
        raw = m.group(1)
        code = raw.replace(" ", "")
        before, after = s[: m.start()], s[m.end():]

        # Glued to a preceding letter: part of another token ("BP60105").
        if before and (before[-1].isalpha() or before[-1] == "-"):
            continue
        if _UNIT_BEFORE.search(before):
            continue
        if code[0] == "0":                       # zero-padded house number
            continue
        if m.start() == 0:                       # opens the whole address
            continue

        seg_start = before.rfind(",") + 1
        opens_segment = before[seg_start:].strip() == ""
        if " " in raw and opens_segment:         # "602 723" house + road
            continue

        tail = _segment_tail(s, m.end()).strip(" ,.-")
        if tail:
            words = {ADDRESS_ABBREV.get(w, w) for w in _TOKEN.findall(tail.lower())}
            if words & STREET_WORDS:             # "11244 Westfall Road"
                continue
            if not any(c.isalpha() for c in tail):  # "11940 72"
                continue
        out[code] = "6-digit" if len(code) == 6 else "5-digit"
    return out


def naive_codes_in(addr: str) -> set[str]:
    """Any 5/6-digit or 'ddd ddd' run -- kept only to show how misleading it is."""
    return {m.group(1).replace(" ", "") for m in _NUM_RE.finditer(addr or "")}


def postal_codes(s: str) -> set[str]:
    """Just the codes, for callers that do not care which format matched."""
    return set(postal_codes_in(s))


def naive_codes_in(addr: str) -> set[str]:
    """Any 5/6-digit or 'ddd ddd' run -- kept only to show how misleading it is."""
    return {m.group(1).replace(" ", "") for m in _NUM_RE.finditer(addr or "")}


def street_numbers(s: str) -> set[str]:
    """Short digit runs — house/building/plot numbers, excluding postal-length runs."""
    if not s:
        return set()
    return {n for n in _NUMBER.findall(_ascii_fold(s)) if len(n) <= 4}


def has_digit(s: str) -> bool:
    return bool(s) and bool(_DIGITS.search(s))


def is_non_latin(s: str) -> bool:
    """True when folding to ASCII actually changed the letters (e.g. Devanagari)."""
    if not s:
        return False
    return _ascii_fold(s).lower() != s.lower()


def char_ngrams(s: str, n: int = 3) -> set[str]:
    """Character n-grams of the normalized name — robust to typos."""
    t = normalize_name(s).replace(" ", "")
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}
