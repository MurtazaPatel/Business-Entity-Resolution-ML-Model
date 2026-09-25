"""Name and address normalization.

Every rule here is a hand-written language rule from CLAUDE.md, applied by matching the
*text*. Nothing branches on the `country` column: France is unseen in training, so a
country-conditioned rule would silently not apply to a third of the test set. Postal
codes, landmarks and street words are therefore detected by pattern only.
"""

from __future__ import annotations

import re
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
    s = _ascii_fold(s).lower()
    s = s.replace("&", " and ")
    s = _collapse_acronyms(s)
    s = _PUNCT.sub(" ", s)
    tokens = _merge_single_letters([t for t in _WS.split(s) if t])
    kept = [t for t in tokens if t not in LEGAL_SUFFIXES]
    # A name that is *only* a legal suffix keeps its tokens rather than becoming empty.
    if not kept:
        kept = tokens
    return " ".join(kept)


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


def postal_codes(s: str) -> set[str]:
    """5-6 digit runs, by pattern. Never conditioned on the country label."""
    if not s:
        return set()
    return set(_POSTAL.findall(_ascii_fold(s)))


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
