"""Tests for the p1c EDA helpers.

The postal cases are real true-pair addresses whose "codes" disagreed under a naive
digit-run extractor; each one is a house, unit or road number, not a postal code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eda  # noqa: E402
import io_utils  # noqa: E402


class TestPostalCodes:
    def test_house_number_followed_by_street_is_not_postal(self):
        assert eda.postal_codes_in("OH, 11244 Westfall Road, Chillicothe") == {}

    def test_zero_padded_house_number_is_not_postal(self):
        assert eda.postal_codes_in("011244 Westfall Road, Frankfort, Ohio") == {}
        assert eda.postal_codes_in("ROUND ROCK, TX, 003801 AW GRIMES BLVD") == {}

    def test_unit_number_is_not_postal(self):
        assert eda.postal_codes_in("3801 Aw Grimes Boulevard, Unit 10207, Round Rock, TX") == {}

    def test_house_plus_road_number_is_not_postal(self):
        assert eda.postal_codes_in("602 723, Bldg Residence, Crittenden County, KY") == {}
        assert eda.postal_codes_in("CHEROKEE, AL, 11940 72") == {}
        assert eda.postal_codes_in("714 200th Place, Lynnwood, WA") == {}

    def test_genuine_codes_are_found_without_a_country_hint(self):
        assert eda.postal_codes_in("Gurugram, Haryana 122016") == {"122016": "6-digit"}
        assert eda.postal_codes_in("Pune, 411001") == {"411001": "6-digit"}
        assert eda.postal_codes_in("Austin, TX 78701") == {"78701": "5-digit"}
        assert eda.postal_codes_in("Kothrud, Pune 411 038") == {"411038": "6-digit"}

    def test_naive_extractor_is_fooled_by_house_numbers(self):
        # the contrast the report shows
        assert eda.naive_codes_in("OH, 11244 Westfall Road") == {"11244"}


class TestArrowHelpers:
    def _gt(self):
        return pa.table({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "matched_entity_ids": ["S2-1,S3-9", "", "S2-5"],
        })

    def test_gt_pairs_arrow_explodes_and_skips_singletons(self):
        pairs = eda.gt_pairs_arrow(self._gt()).to_pylist()
        assert pairs == [{"s1": "S1-1", "other": "S2-1"}, {"s1": "S1-1", "other": "S3-9"},
                         {"s1": "S1-3", "other": "S2-5"}]

    def test_match_counts_treats_empty_as_zero(self):
        assert eda.match_counts(self._gt()).tolist() == [2, 0, 1]

    def test_read_table_matches_read_source(self, tmp_path):
        p = tmp_path / "x.tsv"
        p.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                     "S3-1\tAcme, Inc\t\tIndia\nS3-2\tNA\tNA Road, NA\tUS\n", encoding="utf-8")
        t = io_utils.read_table(p)
        assert t.num_rows == io_utils.count_rows(p) == 2
        assert t.column("business_address").to_pylist() == ["", "NA Road, NA"]
        assert t.column("business_name").to_pylist() == ["Acme, Inc", "NA"]
        assert t.column("business_address").null_count == 0


class TestNoiseDirection:
    def test_counts_only_when_reference_countries_agree(self):
        sig = pd.DataFrame([
            # metric where both references say S2 > S1 by >= 1pp, and France agrees
            {"split": "test", "source": "S1", "country": "US", "m %": 1.0},
            {"split": "test", "source": "S2", "country": "US", "m %": 5.0},
            {"split": "test", "source": "S1", "country": "India", "m %": 2.0},
            {"split": "test", "source": "S2", "country": "India", "m %": 9.0},
            {"split": "test", "source": "S1", "country": "France", "m %": 0.5},
            {"split": "test", "source": "S2", "country": "France", "m %": 3.0},
        ])
        df, share, n = eda.noise_direction(sig, ["US", "India"], ["France"])
        assert n == 1 and share == 1.0
        assert bool(df.iloc[0]["same direction"])

    def test_reference_disagreement_is_skipped(self):
        sig = pd.DataFrame([
            {"split": "test", "source": "S1", "country": "US", "m %": 1.0},
            {"split": "test", "source": "S2", "country": "US", "m %": 5.0},
            {"split": "test", "source": "S1", "country": "India", "m %": 9.0},
            {"split": "test", "source": "S2", "country": "India", "m %": 2.0},
            {"split": "test", "source": "S1", "country": "France", "m %": 0.5},
            {"split": "test", "source": "S2", "country": "France", "m %": 3.0},
        ])
        _, share, n = eda.noise_direction(sig, ["US", "India"], ["France"])
        assert n == 0 and np.isnan(share)


def test_category_rules():
    assert eda._category(95, 95, False).startswith("duplicate")
    assert eda._category(40, 95, False) == "name variant at same address"
    assert eda._category(95, 30, False).endswith("(branch?)")
    assert eda._category(95, 0, True) == "no address, same name"
