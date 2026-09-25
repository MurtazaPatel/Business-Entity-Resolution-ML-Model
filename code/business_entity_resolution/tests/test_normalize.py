"""Tests for the normalization rules.

These encode CLAUDE.md's dictionaries. The country-independence tests matter most: a rule
that only fires for US/India text would silently skip a third of the test set (France).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import normalize as nz  # noqa: E402


class TestNames:
    def test_legal_suffixes_are_dropped(self):
        assert nz.normalize_name("Acme Pvt Ltd") == "acme"
        assert nz.normalize_name("Acme Private Limited") == "acme"
        assert nz.normalize_name("Acme Corp") == nz.normalize_name("Acme Corporation")
        assert nz.normalize_name("Zephay Labs Inc") == "zephay labs"

    def test_suffix_only_name_survives(self):
        # Dropping every token would erase the record's only signal.
        assert nz.normalize_name("Ltd") == "ltd"

    def test_ampersand_becomes_and_then_drops_as_stopword(self):
        assert nz.normalize_name("B & C") == nz.normalize_name("B and C")
        assert nz.name_tokens("B & C Inc") == nz.name_tokens("B and C Incorporated")

    def test_punctuation_and_case_are_ignored(self):
        assert nz.normalize_name("B+ Retail Inc") == nz.normalize_name("b retail")
        assert nz.normalize_name("Fractales Amis Groupe S.A.S") == "fractales amis groupe"

    def test_accents_fold_to_ascii(self):
        assert nz.normalize_name("SCI Ptit Àmicale") == nz.normalize_name("Ptit Amicale")
        assert nz.normalize_name("Moncada Léarning Center") == "moncada learning center"

    def test_devanagari_transliterates(self):
        out = nz.normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
        assert out and out.isascii()

    def test_word_order_is_cancelled_by_sort_key(self):
        assert nz.name_sort_key("South International Consultants") == nz.name_sort_key(
            "International South Consultants"
        )

    def test_tokens_exclude_noise(self):
        toks = nz.name_tokens("The Bank of A Ltd")
        assert "the" not in toks and "of" not in toks and "ltd" not in toks
        assert "a" not in toks  # single chars dropped
        assert "bank" in toks


class TestAddresses:
    def test_road_abbreviation(self):
        assert "road" in nz.normalize_address("Mack Rd, Haltom City")

    def test_st_is_street_at_segment_end(self):
        assert "street" in nz.normalize_address("105 ELM ST, MORGANTON, NC")

    def test_st_is_saint_before_a_name(self):
        assert "saint" in nz.normalize_address("ST LOUIS, MO")

    def test_french_abbreviations(self):
        out = nz.normalize_address("63 R. DE DIEPPE, LILLE")
        assert "rue" in out
        assert "boulevard" in nz.normalize_address("18 BD Voltaire, Paris")

    def test_cedex_is_dropped(self):
        assert "cedex" not in nz.normalize_address("12 rue Victor Hugo, 75001 Paris Cedex")

    def test_near_and_opposite_expand(self):
        assert "near" in nz.normalize_address("Nr SBI ATM, Pune")
        assert "opposite" in nz.normalize_address("Opp Railway Station")

    def test_empty_address_stays_empty(self):
        assert nz.normalize_address("") == ""
        assert nz.address_tokens("") == set()


class TestPatternsAreCountryIndependent:
    """Postal extraction must work on all three formats with no country argument."""

    def test_us_zip_france_cp_india_pin(self):
        assert nz.postal_codes("MORGANTON, NC 28655") == {"28655"}       # US ZIP-5
        assert nz.postal_codes("75001 Paris") == {"75001"}                # FR code postal
        assert nz.postal_codes("Gurugram, HR 122016") == {"122016"}       # IN PIN-6

    def test_long_digit_runs_are_not_postal(self):
        assert nz.postal_codes("KH NO. -570134567890") == set()

    def test_street_numbers_exclude_postal_length(self):
        nums = nz.street_numbers("1795 Westchester Drive, High Point, NC 27262")
        assert "1795" in nums and "27262" not in nums

    def test_signature_takes_no_country(self):
        import inspect

        for fn in (nz.normalize_name, nz.normalize_address, nz.postal_codes, nz.street_numbers):
            params = inspect.signature(fn).parameters
            assert "country" not in params, f"{fn.__name__} must not branch on country"


class TestSimilarityHelpers:
    def test_trigrams_overlap_for_typos(self):
        a, b = nz.char_ngrams("Coimbatore"), nz.char_ngrams("Coimatore")
        assert len(a & b) >= 4

    def test_is_non_latin(self):
        assert nz.is_non_latin("मॉडर्न फाइनेंस")
        assert not nz.is_non_latin("Modern Finance")

    def test_has_digit(self):
        assert nz.has_digit("3/115, East Delhi")
        assert not nz.has_digit("Mack Rd")
