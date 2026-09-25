"""Tests for the CV folds and LOCO splits.

The leakage guarantee is the point: if one S1 entity's pairs span two folds, validation
scores a near-duplicate of something the model trained on and every number after that
is optimistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import splits as S  # noqa: E402


def make_frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    n_matches = np.where(rng.random(n) < 0.0558, 0, rng.integers(1, 12, n))
    return pd.DataFrame({
        "source1_entity_id": [f"S1-{i}" for i in range(n)],
        "country": rng.choice(["US", "India"], n, p=[0.6, 0.4]),
        "n_matches": n_matches.astype(np.int32),
        "is_singleton": n_matches == 0,
    })


class TestFoldAssignment:
    def test_every_entity_gets_exactly_one_fold(self):
        f = S.assign_folds(make_frame())
        assert set(f["fold"]) == set(range(5))
        assert (f["fold"] >= 0).all()
        assert f.groupby("source1_entity_id")["fold"].nunique().max() == 1

    def test_folds_partition_the_entities(self):
        frame = make_frame()
        f = S.assign_folds(frame)
        assert len(f) == len(frame)
        assert sum((f["fold"] == k).sum() for k in range(5)) == len(frame)

    def test_singleton_rate_is_balanced_across_folds(self):
        f = S.assign_folds(make_frame(4000))
        rates = f.groupby("fold")["is_singleton"].mean()
        assert rates.max() - rates.min() < 0.02

    def test_country_mix_is_balanced_across_folds(self):
        f = S.assign_folds(make_frame(4000))
        share = f.groupby("fold")["country"].apply(lambda s: (s == "US").mean())
        assert share.max() - share.min() < 0.02

    def test_assignment_is_deterministic_for_a_seed(self):
        frame = make_frame()
        assert S.assign_folds(frame, seed=42)["fold"].equals(S.assign_folds(frame, seed=42)["fold"])

    def test_different_seeds_give_different_folds(self):
        frame = make_frame()
        assert not S.assign_folds(frame, seed=1)["fold"].equals(S.assign_folds(frame, seed=2)["fold"])

    def test_columns_are_the_documented_ones(self):
        assert list(S.assign_folds(make_frame()).columns) == S.FOLD_COLUMNS


class TestNoLeakage:
    def test_pairs_of_one_entity_never_span_folds(self):
        """The guarantee that matters once the frame is exploded into pairs."""
        frame = make_frame(800)
        f = S.assign_folds(frame)
        pairs = pd.DataFrame([
            {"s1": r.source1_entity_id, "other": f"S2-{r.source1_entity_id}-{j}"}
            for r in frame.itertuples() for j in range(r.n_matches)
        ])
        fold_of = dict(zip(f["source1_entity_id"], f["fold"]))
        pairs["fold"] = pairs["s1"].map(fold_of)
        assert pairs.groupby("s1")["fold"].nunique().max() == 1

    def test_train_and_val_masks_are_disjoint_and_complete(self):
        f = S.assign_folds(make_frame())
        for k in range(5):
            tr, va = S.fold_masks(f, k)
            assert not (tr & va).any()
            assert (tr | va).all()

    def test_filter_pairs_selects_only_masked_entities(self):
        f = S.assign_folds(make_frame(200))
        pairs = pd.DataFrame({"s1": f["source1_entity_id"], "other": "S2-x"})
        _, va = S.fold_masks(f, 0)
        kept = S.filter_pairs(pairs, f, va)
        assert set(kept["s1"]) == set(f.loc[va, "source1_entity_id"])

    def test_check_no_leakage_all_pass(self):
        checks = S.check_no_leakage(S.assign_folds(make_frame(4000)))
        assert checks["pass"].all(), checks[~checks["pass"]].to_dict("records")

    def test_check_no_leakage_all_pass_at_scale(self):
        checks = S.check_no_leakage(S.assign_folds(make_frame(50_000)))
        assert checks["pass"].all(), checks[~checks["pass"]].to_dict("records")

    def test_singleton_spread_tolerance_tightens_with_n(self):
        """The balance check must get stricter as the sample grows, not looser."""
        def tol(n):
            row = S.check_no_leakage(S.assign_folds(make_frame(n)))["check"][3]
            return float(row.split("<= ")[1].rstrip("pp"))
        assert tol(50_000) < tol(4_000)

    def test_check_no_leakage_catches_a_spanning_entity(self):
        f = S.assign_folds(make_frame(500))
        broken = pd.concat([f, f.iloc[[0]].assign(fold=(f.iloc[0]["fold"] + 1) % 5)])
        checks = S.check_no_leakage(broken).set_index("check")
        assert not checks.loc["every S1 in exactly one fold", "pass"]
        assert not checks.loc["no duplicate S1 rows", "pass"]


class TestLoco:
    def test_both_directions_are_produced(self):
        f = S.assign_folds(make_frame())
        assert {s["name"] for s in S.loco_splits(f)} == {"US->India", "India->US"}

    def test_masks_are_disjoint_and_match_the_country(self):
        f = S.assign_folds(make_frame())
        for s in S.loco_splits(f):
            tr, ev = S.loco_masks(f, s)
            assert not (tr & ev).any()
            assert set(f.loc[tr, "country"]) == {s["train_country"]}
            assert set(f.loc[ev, "country"]) == {s["eval_country"]}

    def test_countries_are_discovered_not_hard_coded(self):
        """A third country must appear without touching the code."""
        f = S.assign_folds(make_frame())
        f.loc[f.index[:50], "country"] = "France"
        names = {s["name"] for s in S.loco_splits(f)}
        assert "US->France" in names and "France->US" in names
        assert len(names) == 6

    def test_summaries_render(self):
        f = S.assign_folds(make_frame())
        assert len(S.fold_summary(f)) == 6          # 5 folds + ALL
        assert len(S.loco_summary(f)) == 2
