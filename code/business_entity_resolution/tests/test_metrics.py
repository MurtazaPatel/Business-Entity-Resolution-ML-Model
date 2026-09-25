"""Tests for macro F0.5.

The metric is the whole competition, and its degenerate cases carry real weight:
singletons are 5.58% of train entities and each scores a full 1.0 or 0.0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import metrics as M  # noqa: E402


class TestProblemStatementExample:
    """PROBLEM.md: pred [S2-47, S2-193, S3-812], truth [S2-47, S3-812] -> 0.714."""

    PRED = ["S2-00047", "S2-00193", "S3-00812"]
    TRUTH = ["S2-00047", "S3-00812"]

    def test_example_scores_0_714(self):
        assert M.score_entity(self.PRED, self.TRUTH) == pytest.approx(0.714, abs=5e-4)

    def test_example_precision_and_recall(self):
        assert M.f_beta(2 / 3, 1.0) == pytest.approx(0.714, abs=5e-4)

    def test_example_matches_the_written_formula(self):
        p, r = 2 / 3, 1.0
        expected = (1.25 * p * r) / (0.25 * p + r)
        assert M.score_entity(self.PRED, self.TRUTH) == pytest.approx(expected)

    def test_macro_average_of_one_entity_is_that_entity(self):
        s = M.macro_f05({"S1-1": self.TRUTH}, {"S1-1": self.PRED})
        assert s.macro_f == pytest.approx(0.714, abs=5e-4)
        assert s.n == 1 and s.n_singletons == 0


class TestSingletonCases:
    def test_empty_pred_on_empty_truth_is_one(self):
        assert M.score_entity([], []) == 1.0

    def test_any_pred_on_empty_truth_is_zero(self):
        assert M.score_entity(["S2-1"], []) == 0.0
        assert M.score_entity(["S2-1", "S3-2"], []) == 0.0

    def test_empty_pred_on_non_empty_truth_is_zero(self):
        assert M.score_entity([], ["S2-1"]) == 0.0

    def test_all_three_cases_inside_a_macro_average(self):
        truth = {"a": [], "b": [], "c": ["S2-1"]}
        pred = {"a": [], "b": ["S2-9"], "c": []}
        s = M.macro_f05(truth, pred)
        assert s.macro_f == pytest.approx(1 / 3)
        assert s.n_singletons == 2
        assert s.singleton_f == pytest.approx(0.5)
        assert s.non_singleton_f == pytest.approx(0.0)
        assert s.singleton_accuracy == pytest.approx(0.5)


class TestPrecisionWeighting:
    def test_beta_half_favours_precision(self):
        """Same counts: losing precision must cost more than losing recall."""
        truth = list("abcd")
        precise = M.score_entity(["a", "b"], truth)            # P=1.00, R=0.50
        recalling = M.score_entity(["a", "b", "c", "d", "e", "f", "g", "h"], truth)  # P=0.50, R=1.00
        assert precise > recalling

    def test_perfect_prediction_is_one(self):
        assert M.score_entity(["S2-1", "S3-2"], ["S3-2", "S2-1"]) == 1.0

    def test_no_overlap_is_zero(self):
        assert M.score_entity(["S2-9"], ["S2-1"]) == 0.0

    def test_duplicate_ids_do_not_inflate_the_score(self):
        assert M.score_entity(["S2-1", "S2-1"], ["S2-1"]) == 1.0

    def test_f_beta_handles_zero_denominator(self):
        assert M.f_beta(0.0, 0.0) == 0.0


class TestMacroBehaviour:
    def test_macro_is_unweighted_over_entities(self):
        """An entity with 11 matches counts exactly as much as a singleton."""
        truth = {"big": [f"S2-{i}" for i in range(11)], "single": []}
        s = M.macro_f05(truth, {"big": [], "single": []})
        assert s.macro_f == pytest.approx(0.5)

    def test_missing_prediction_counts_as_empty(self):
        s = M.macro_f05({"a": ["S2-1"], "b": []}, {})
        assert s.macro_f == pytest.approx(0.5)
        assert s.empty_pred_rate == 1.0

    def test_truth_defines_the_evaluation_set(self):
        s = M.macro_f05({"a": []}, {"a": [], "extra": ["S2-1"]})
        assert s.n == 1 and s.macro_f == 1.0

    def test_predict_empty_everywhere_scores_the_singleton_rate(self):
        """The 'always empty' baseline: 5.58% singletons -> macro F0.5 = 0.0558."""
        truth = {f"s{i}": [] for i in range(558)}
        truth.update({f"m{i}": ["S2-1"] for i in range(9442)})
        s = M.macro_f05(truth, {})
        assert s.macro_f == pytest.approx(0.0558, abs=1e-4)


class TestAdapters:
    def test_id_lists_from_submission_frame(self):
        df = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2"],
                           "matched_entity_ids": ["S2-1,S3-2", ""]})
        assert M.id_lists_from_frame(df) == {"S1-1": ["S2-1", "S3-2"], "S1-2": []}

    def test_score_frames_end_to_end(self):
        truth = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2"],
                              "matched_entity_ids": ["S2-00047,S3-00812", ""]})
        pred = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2"],
                             "matched_entity_ids": ["S2-00047,S2-00193,S3-00812", ""]})
        s = M.score_frames(truth, pred)
        assert s.macro_f == pytest.approx((0.714 + 1.0) / 2, abs=5e-4)

    def test_pairs_to_id_lists(self):
        pairs = pd.DataFrame({"s1": ["A", "A", "B"], "other": ["S2-1", "S3-2", "S2-3"]})
        assert M.pairs_to_id_lists(pairs) == {"A": ["S2-1", "S3-2"], "B": ["S2-3"]}

    def test_score_reports_nan_when_a_group_is_absent(self):
        s = M.macro_f05({"a": ["S2-1"]}, {"a": ["S2-1"]})
        assert math.isnan(s.singleton_f) and s.non_singleton_f == 1.0
