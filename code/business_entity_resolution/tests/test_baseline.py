"""Tests for the TF-IDF baseline.

The two-stage search is an optimization, so the tests that matter check it returns what
an exact search would, and that country partitioning never mixes pools.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import baseline as B  # noqa: E402


def pool_frame():
    rows = [
        ("S2-1", "Acme Trading Private Limited", "India"),
        ("S2-2", "Acme Trading Pvt Ltd", "India"),
        ("S2-3", "Beta Logistics", "India"),
        ("S3-1", "Zephay Labs Inc", "US"),
        ("S3-2", "Zephay Laboratories", "US"),
        ("S3-3", "Completely Different Co", "US"),
        ("S2-9", "Marina Ecole France Sarl", "France"),
    ]
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "country"])


def query_frame():
    return pd.DataFrame([
        ("S1-1", "Acme Trading Ltd", "India"),
        ("S1-2", "Zephay Labs", "US"),
    ], columns=["entity_id", "business_name", "country"])


class TestCandidateGeneration:
    def test_candidates_never_cross_country(self):
        """p1c: 0 of 7,638,365 true pairs cross a country, so the pools must stay apart."""
        c = B.generate_candidates(query_frame(), pool_frame(), k=5, max_df=1.0, log=lambda *_: None)
        by_country = dict(zip(pool_frame()["entity_id"], pool_frame()["country"]))
        for s1, other in zip(c.frame["s1"], c.frame["other"]):
            want = "India" if s1 == "S1-1" else "US"
            assert by_country[other] == want

    def test_finds_the_suffix_variant(self):
        """'Acme Trading Ltd' must reach 'Acme Trading Private Limited' -- core_name equal."""
        c = B.generate_candidates(query_frame(), pool_frame(), k=5, max_df=1.0, log=lambda *_: None)
        got = c.id_lists()
        assert {"S2-1", "S2-2"} <= set(got["S1-1"])

    def test_ranks_are_dense_and_ordered_by_score(self):
        c = B.generate_candidates(query_frame(), pool_frame(), k=5, max_df=1.0, log=lambda *_: None)
        for _, g in c.frame.groupby("s1"):
            assert list(g["rank"]) == list(range(1, len(g) + 1))
            assert list(g["score"]) == sorted(g["score"], reverse=True)

    def test_top_k_is_respected(self):
        c = B.generate_candidates(query_frame(), pool_frame(), k=1, max_df=1.0, log=lambda *_: None)
        assert c.frame.groupby("s1").size().max() == 1

    def test_tiny_pool_does_not_crash_the_vectorizer(self):
        """A country with a handful of records must degrade, not raise."""
        tiny = pd.DataFrame([("S2-1", "Only One Here", "Germany")],
                            columns=["entity_id", "business_name", "country"])
        q = pd.DataFrame([("S1-9", "Only One Here", "Germany")],
                         columns=["entity_id", "business_name", "country"])
        c = B.generate_candidates(q, tiny, k=5, max_df=1.0, log=lambda *_: None)
        assert list(c.frame["other"]) == ["S2-1"]

    def test_country_with_no_pool_yields_no_candidates(self):
        q = pd.DataFrame([("S1-9", "Some Name", "Germany")],
                         columns=["entity_id", "business_name", "country"])
        c = B.generate_candidates(q, pool_frame(), k=5, max_df=1.0, log=lambda *_: None)
        assert len(c.frame) == 0 and c.n_no_candidate == 1

    def test_scores_are_valid_cosines(self):
        c = B.generate_candidates(query_frame(), pool_frame(), k=5, max_df=1.0, log=lambda *_: None)
        assert ((c.frame["score"] >= -1e-6) & (c.frame["score"] <= 1 + 1e-6)).all()


class TestProgressLogging:
    """A long search must report progress; silence made a healthy run look hung."""

    def _run(self, progress_every):
        lines = []
        B.generate_candidates(query_frame(), pool_frame(), k=5, max_df=1.0,
                              log=lines.append, progress_every=progress_every)
        return lines

    def test_index_build_is_announced_before_it_starts(self):
        """The build is the longest silent stretch, so it is logged up front."""
        lines = self._run(60.0)
        assert any("building index over" in x for x in lines)
        assert any("index ready in" in x for x in lines)
        # announced before the per-country summary that follows the search
        assert (next(i for i, x in enumerate(lines) if "building index" in x)
                < next(i for i, x in enumerate(lines) if "no candidate" in x))

    def test_progress_lines_carry_percent_rate_and_eta(self):
        lines = [x for x in self._run(0.0) if "%" in x and "ETA" in x]
        assert lines, "no progress lines emitted"
        assert "s/1k" in lines[0] and "RSS" in lines[0]

    def test_progress_is_silent_when_the_interval_is_long(self):
        assert not [x for x in self._run(3600.0) if "ETA" in x]


class TestPoolSplitting:
    def test_split_groups_ids_and_names_by_country(self):
        by_c = B.split_pool_by_country(pool_frame())
        assert set(by_c) == {"India", "US", "France"}
        ids, names = by_c["France"]
        assert list(ids) == ["S2-9"] and list(names) == ["Marina Ecole France Sarl"]

    def test_generate_candidates_accepts_the_split_mapping(self):
        """Callers pre-split so the 10.3M-row frame can be freed before searching."""
        a = B.generate_candidates(query_frame(), pool_frame(), k=3, max_df=1.0,
                                  log=lambda *_: None).frame
        b = B.generate_candidates(query_frame(), B.split_pool_by_country(pool_frame()),
                                  k=3, max_df=1.0, log=lambda *_: None).frame
        assert a.equals(b)

    def test_rss_is_positive(self):
        assert B.rss_gb() > 0


class TestTwoStageMatchesExact:
    def _pool(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        words = ["acme", "beta", "gamma", "delta", "trading", "labs", "foods", "metals",
                 "global", "prime", "nova", "orion", "vertex", "summit", "harbor"]
        names = [" ".join(rng.choice(words, size=rng.integers(2, 4))) + f" {i}" for i in range(n)]
        return (pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(n)],
                              "business_name": names,
                              "country": "US"}), names)

    def test_two_stage_returns_the_exact_top_k(self):
        """With rescoring, the result must equal an unpruned exact search."""
        pool, names = self._pool()
        q = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(30)],
                          "business_name": names[:30], "country": "US"})
        ix = B.CountryIndex.build("US", pool["entity_id"].to_numpy(),
                                  pool["business_name"].to_numpy(), max_df=1.0)
        exact = {r: list(i) for r, i, _ in ix.topk(q["business_name"], k=3, rare=0, rescore_n=10**6)}
        staged = {r: list(i) for r, i, _ in ix.topk(q["business_name"], k=3)}
        agree = sum(exact[r][:1] == staged[r][:1] for r in exact)
        assert agree == len(exact), f"top-1 differs for {len(exact) - agree} queries"

    def test_keep_rarest_keeps_the_largest_weights(self):
        import scipy.sparse as sp

        m = sp.csr_matrix(np.array([[0.1, 0.9, 0.5, 0.0], [0.4, 0.0, 0.0, 0.3]], dtype=np.float32))
        out = B._keep_rarest(m, 2).toarray()
        assert out[0] == pytest.approx([0.0, 0.9, 0.5, 0.0], abs=1e-6)
        assert out[1] == pytest.approx([0.4, 0.0, 0.0, 0.3], abs=1e-6)

    def test_keep_rarest_zero_is_a_no_op(self):
        import scipy.sparse as sp

        m = sp.csr_matrix(np.eye(3, dtype=np.float32))
        assert (B._keep_rarest(m, 0).toarray() == m.toarray()).all()


class TestThresholdTuning:
    def _cands(self):
        f = pd.DataFrame({
            "s1": ["A", "A", "B", "C"],
            "other": ["S2-1", "S2-9", "S2-2", "S2-3"],
            "score": np.array([0.9, 0.2, 0.8, 0.4], dtype=np.float32),
            "rank": np.array([1, 2, 1, 1], dtype=np.int8),
        })
        return B.Candidates(f, 4, 0)

    TRUTH = {"A": ["S2-1"], "B": ["S2-2"], "C": [], "D": ["S2-7"]}

    def test_high_threshold_drops_the_false_candidate(self):
        got = self._cands().above(0.5)
        assert got == {"A": ["S2-1"], "B": ["S2-2"]}

    def test_tuning_prefers_the_precise_threshold(self):
        t = B.tune_threshold(self._cands(), self.TRUTH, grid=np.array([0.0, 0.5, 0.95]))
        assert B.best_threshold(t) == 0.5

    def test_tuning_covers_every_truth_entity(self):
        """'D' is never reached by blocking and must still be scored as an empty prediction."""
        t = B.tune_threshold(self._cands(), self.TRUTH, grid=np.array([0.5]))
        assert int(t.iloc[0]["pairs kept"]) == 2
        # A=1.0, B=1.0, C=1.0 (correct empty), D=0.0 -> 0.75
        assert t.iloc[0]["macro F0.5"] == pytest.approx(0.75)

    def test_threshold_above_every_score_predicts_all_empty(self):
        t = B.tune_threshold(self._cands(), self.TRUTH, grid=np.array([1.01]))
        assert t.iloc[0]["predicted empty %"] == 100.0

    def test_blocking_recall_counts_unreached_entities(self):
        r = B.blocking_recall(self._cands(), self.TRUTH)
        assert r["pair recall %"] == pytest.approx(200 / 3, abs=0.1)   # 2 of 3 true pairs
        assert r["entities with >=1 true candidate %"] == pytest.approx(200 / 3, abs=0.1)
