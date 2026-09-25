"""Tests for the normalization rules.

These encode CLAUDE.md's dictionaries. The country-independence tests matter most: a rule
that only fires for US/India text would silently skip a third of the test set (France).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


class TestPostalCodes:
    """15 real addresses from the p1c samples, France included.

    Every expectation was verified against the actual record. The discriminator is what
    FOLLOWS the digit run inside its comma segment -- a street word or bare digits mean a
    house number, a plain word or segment end mean a postal code. No rule looks at the
    country label, which is what lets one rule set cover ZIP, PIN and code postal.
    """

    HOUSE_NUMBERS = [
        ("OH, 11244 Westfall Road, Chillicothe", "US house number after reordering"),
        ("011244 Westfall Road, Frankfort, Ohio", "zero-padded house number"),
        ("3801 Aw Grimes Boulevard, Unit 10207, Round Rock, TX", "unit number"),
        ("602 723, Bldg Residence, Crittenden County, KY", "house + road number"),
        ("CHEROKEE, AL, 11940 72", "house number + numeric road"),
        ("00109 BOULEVARD DES BELGES, NANTES", "French zero-padded house number"),
        ("27 R DU BALLET, BP60105, NANTES, Pays de la Loire", "PO box glued to letters"),
        ("11024, S/F, Gali Peepal Wali, Pahar Ganj", "Indian house number opening address"),
    ]
    POSTAL_CODES = [
        ("MORGANTON, NC 28655", {"28655"}, "US ZIP-5"),
        ("Gurugram, Haryana 122016", {"122016"}, "India PIN-6"),
        ("Kothrud, Pune 411 038", {"411038"}, "India PIN as 'ddd ddd'"),
        ("195 R. DE LA MACKELLERIE, 59100, ROUBAIX, Nord", {"59100"}, "FR code, segment ends"),
        ("N°2 RUE DES AUGUSTINS, 33000 BORDEAUX, BORDEAUX, Gironde", {"33000"}, "FR code + city"),
        ("8 R. DE L'ESCAUT, 59000 LILLE, LILLE, Nord", {"59000"}, "FR code + city"),
        ("46 BD VINCENT GACHE, 44000, NANTES, Loire-Atlantique", {"44000"}, "FR code"),
    ]

    @pytest.mark.parametrize("addr,why", HOUSE_NUMBERS)
    def test_house_and_unit_numbers_are_not_postal_codes(self, addr, why):
        assert nz.postal_codes_in(addr) == {}, why

    @pytest.mark.parametrize("addr,want,why", POSTAL_CODES)
    def test_genuine_codes_are_found(self, addr, want, why):
        assert set(nz.postal_codes_in(addr)) == want, why

    def test_french_codes_survive_the_us_house_number_rule(self):
        """'33000 BORDEAUX' and '11244 Westfall Road' have the same shape.

        Only the segment tail separates them, so this is the case that would regress if
        the rule were ever simplified back to a positional one.
        """
        assert set(nz.postal_codes_in("N°2 RUE, 33000 BORDEAUX, Gironde")) == {"33000"}
        assert nz.postal_codes_in("OH, 11244 Westfall Road") == {}

    def test_kind_is_reported(self):
        assert nz.postal_codes_in("Pune 411038") == {"411038": "6-digit"}
        assert nz.postal_codes_in("Austin, TX 78701") == {"78701": "5-digit"}

    def test_postal_codes_returns_just_the_codes(self):
        assert nz.postal_codes("MORGANTON, NC 28655") == {"28655"}

    def test_naive_extractor_is_fooled(self):
        assert nz.naive_codes_in("OH, 11244 Westfall Road") == {"11244"}

    def test_signature_takes_no_country(self):
        import inspect

        for fn in (nz.normalize_name, nz.normalize_address, nz.postal_codes,
                   nz.postal_codes_in, nz.street_numbers, nz.split_name):
            assert "country" not in inspect.signature(fn).parameters, fn.__name__

    def test_street_numbers_exclude_postal_length(self):
        nums = nz.street_numbers("1795 Westchester Drive, High Point, NC 27262")
        assert "1795" in nums and "27262" not in nums


class TestNameSplit:
    """core_name + legal_suffix, using CLAUDE.md's dictionary."""

    @pytest.mark.parametrize("raw,core,sufs", [
        ("Acme Pvt Ltd", "acme", ("pvt", "ltd")),
        ("Zephay Labs Inc", "zephay labs", ("inc",)),
        ("Fractales Amis Groupe S.A.S", "fractales amis groupe", ("sas",)),
        ("Marina Ecole France Sarl", "marina ecole france", ("sarl",)),
        ("LLC Moncada Léarning Center", "moncada learning center", ("llc",)),
        ("International South Consultants Private Ltd",
         "international south consultants", ("private", "ltd")),
        ("Defense Ecole SAS", "defense ecole", ("sas",)),
        ("SCI Ptit Àmicale", "ptit amicale", ("sci",)),
        ("B+ Retail Inc", "b retail", ("inc",)),
        ("wilfordhancock.com", "wilfordhancock com", ()),
    ])
    def test_split(self, raw, core, sufs):
        parts = nz.split_name(raw)
        assert parts.core == core
        assert parts.suffixes == sufs

    def test_suffix_only_name_keeps_its_tokens(self):
        parts = nz.split_name("Ltd")
        assert parts.core == "ltd" and parts.suffixes == ("ltd",)

    def test_core_name_matches_normalize_name(self):
        for raw in ("Acme Pvt Ltd", "राम मार्केटिंग प्राइवेट लिमिटेड", "B & C, Inc", ""):
            assert nz.core_name(raw) == nz.normalize_name(raw)

    def test_suffix_differences_do_not_change_the_core(self):
        assert nz.core_name("Acme Corp") == nz.core_name("Acme Corporation") == "acme"

    def test_digits_are_kept(self):
        assert "7" in nz.core_name("L 7 Prime Hudson")
        assert nz.core_name("#2 Transport") == "2 transport"

    def test_legal_suffix_string(self):
        assert nz.split_name("Acme Pvt Ltd").legal_suffix == "pvt ltd"


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
