"""p2b baseline: char 3-gram TF-IDF on core_name, top-k by cosine, one global threshold.

Deliberately name-only, so it measures how far the name alone gets us. p1c already says
that ceiling is low -- at 10.3M candidates the median name-nearest-neighbour cosine is 1.0
even for singletons -- and this step puts a number on it for the address features to beat.

Scale note: an exact top-k over every (S1, S2/S3) pair is not affordable. Two things make
it tractable without changing what is computed:
  * candidates are partitioned by country, which p1c showed costs no recall at all
    (0 of 7,638,365 true pairs cross a country) and is a same_country constraint, not a
    hard-coded country list;
  * trigrams above `max_df` are dropped, so a query is only ever compared against records
    sharing a reasonably rare trigram. Everything surviving that is scored exactly.
The surviving comparisons are exactly what candidate_pairs.tsv reports.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

import normalize as nz
from config import SEED

TOP_K = 5
MAX_DF = 0.01
# Optional two-stage search: keep each query's RARE_TRIGRAMS highest-IDF trigrams, take
# the best RESCORE_N by that partial score, then rescore exactly. Measured on the 6.2M-row
# US pool it is a net LOSS -- 26-72s per 1k queries against 15s for the single-stage path,
# for 64.3% entity recall@5 against 65.0% -- because the per-query Python rescore costs
# more than the pruning saves. Off by default; max_df is the knob that actually works.
RARE_TRIGRAMS = 0
RESCORE_N = 50
MIN_DF = 3
NGRAM = (3, 3)
# Peak memory is driven by the sparse product for one chunk, not by the index. At
# max_df=0.01 the US pool averages ~143k non-zeros per query, i.e. ~1.1 MB per query at
# 8 bytes each (float32 value + int32 column), so a chunk of 500 costs ~0.5 GB per worker
# and 4 workers stay near 2 GB. A chunk of 2000 would be 8.5 GB across 4 workers, which is
# enough to fall over on a smaller box.
QUERY_CHUNK = 500
WORKER_CHUNK = 20_000


def core_names(names) -> list[str]:
    return [nz.core_name(x) for x in names]


def rss_gb() -> float:
    """Resident set size. maxrss is bytes on macOS, kilobytes on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**3 if os.uname().sysname == "Darwin" else rss / 1024**2


def split_pool_by_country(pool: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{country: (ids, names)} so the caller can free the DataFrame before searching.

    The pooled frame is ~10.3M rows; holding it alongside the index and the forked
    workers is exactly the peak we do not need.
    """
    country = pool["country"].astype(str).to_numpy()
    ids = pool["entity_id"].astype(str).to_numpy()
    names = pool["business_name"].astype(str).to_numpy()
    return {c: (ids[country == c], names[country == c]) for c in np.unique(country)}


@dataclass
class CountryIndex:
    """TF-IDF index over one country's S2/S3 pool."""

    country: str
    ids: np.ndarray
    vectorizer: TfidfVectorizer
    matrix: sp.csr_matrix            # (n_candidates, vocab) -- fast row slicing
    matrix_t: sp.csr_matrix          # (vocab, n_candidates) -- fast blocking matmul
    n_candidates: int
    vocab: int
    nnz_per_doc: float

    @classmethod
    def build(cls, country: str, ids, names, max_df: float = MAX_DF,
              min_df: int = MIN_DF, ngram=NGRAM) -> "CountryIndex":
        """Build the index, relaxing the frequency filters if they empty the vocabulary.

        min_df/max_df are tuned for pools of millions. A country with only a handful of
        records -- which is exactly what an unseen country could look like -- would
        otherwise prune every trigram and raise, so fall back rather than crash.
        """
        docs = core_names(names)
        attempts = [(min_df, max_df), (1, 1.0)]
        last: Exception | None = None
        for md, xd in attempts:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=md,
                                  max_df=xd, dtype=np.float32, sublinear_tf=True)
            try:
                C = vec.fit_transform(docs).tocsr()
            except ValueError as e:       # "After pruning, no terms remain"
                last = e
                continue
            if C.shape[1] == 0:
                continue
            return cls(country, np.asarray(ids, dtype=object), vec, C, C.T.tocsr(),
                       C.shape[0], C.shape[1], C.nnz / max(C.shape[0], 1))
        raise ValueError(f"{country}: could not build a vocabulary from "
                         f"{len(docs):,} names") from last

    def topk(self, query_names, k: int = TOP_K, chunk: int = QUERY_CHUNK,
             rare: int = RARE_TRIGRAMS, rescore_n: int = RESCORE_N):
        """Yield (row, candidate_positions, exact_cosine) per query, best first.

        Scores are exact cosines over the indexed vocabulary. Rows with no trigram
        overlap yield empty arrays: a real "no candidate" answer, not an error.

        With `rare > 0` the optional two-stage path runs (see RARE_TRIGRAMS); it is off by
        default because measurement showed it slower for no recall gain.
        """
        Q = self.vectorizer.transform(core_names(query_names)).tocsr()
        for start in range(0, Q.shape[0], chunk):
            Qc = Q[start:start + chunk]
            S = ((Qc if rare <= 0 else _keep_rarest(Qc, rare)) @ self.matrix_t).tocsr()
            for r in range(S.shape[0]):
                lo, hi = S.indptr[r], S.indptr[r + 1]
                if lo == hi:
                    yield start + r, np.empty(0, np.int64), np.empty(0, np.float32)
                    continue
                idx, val = S.indices[lo:hi], S.data[lo:hi]
                if rare > 0:
                    # Stage-1 scores are partial; rescore the survivors exactly.
                    if len(idx) > rescore_n:
                        idx = idx[np.argpartition(-val, rescore_n)[:rescore_n]]
                    val = np.asarray((self.matrix[idx] @ Qc[r].T).todense()).ravel()
                if len(idx) > k:
                    part = np.argpartition(-val, k)[:k]
                    idx, val = idx[part], val[part]
                order = np.argsort(-val)
                yield start + r, idx[order], val[order].astype(np.float32)


def _keep_rarest(Q: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """Zero all but each row's n largest tf-idf weights (its rarest trigrams)."""
    if n <= 0:
        return Q
    out = Q.copy()
    indptr, data = out.indptr, out.data
    for r in range(out.shape[0]):
        lo, hi = indptr[r], indptr[r + 1]
        extra = hi - lo - n
        if extra > 0:
            data[lo + np.argpartition(data[lo:hi], extra)[:extra]] = 0
    out.eliminate_zeros()
    return out


@dataclass
class Candidates:
    """Top-k candidates per S1: a long frame plus the blocking stats worth reporting."""

    frame: pd.DataFrame              # s1, other, score, rank
    n_queries: int
    n_no_candidate: int
    stats: dict = field(default_factory=dict)

    def id_lists(self) -> dict[str, list[str]]:
        from metrics import pairs_to_id_lists

        return pairs_to_id_lists(self.frame)

    def above(self, threshold: float) -> dict[str, list[str]]:
        f = self.frame[self.frame["score"] >= threshold]
        from metrics import pairs_to_id_lists

        return pairs_to_id_lists(f)


# The index is handed to forked workers through a module global: on Linux, fork shares it
# copy-on-write, so the multi-GB matrices are never pickled or duplicated.
_SHARED_INDEX: "CountryIndex | None" = None


def _topk_worker(job):
    start, names, k, rare, rescore_n, chunk = job
    ix = _SHARED_INDEX
    out = []
    for r, idx, val in ix.topk(names, k=k, chunk=chunk, rare=rare, rescore_n=rescore_n):
        if len(idx):
            out.append((start + r, ix.ids[idx].tolist(), val.tolist()))
        else:
            out.append((start + r, [], []))
    return out


def _search(index: "CountryIndex", names: np.ndarray, k: int, jobs: int,
            rare: int = RARE_TRIGRAMS, rescore_n: int = RESCORE_N,
            chunk: int = QUERY_CHUNK):
    """Yield (row, candidate_ids, scores). Uses forked workers when jobs > 1."""
    if jobs <= 1:
        for r, idx, val in index.topk(names, k=k, chunk=chunk, rare=rare, rescore_n=rescore_n):
            yield r, index.ids[idx].tolist() if len(idx) else [], val.tolist()
        return

    import multiprocessing as mp

    global _SHARED_INDEX
    _SHARED_INDEX = index
    jobs_list = [(s, names[s:s + WORKER_CHUNK], k, rare, rescore_n, chunk)
                 for s in range(0, len(names), WORKER_CHUNK)]
    try:
        ctx = mp.get_context("fork")
    except ValueError:                      # no fork (Windows/macOS spawn): stay serial
        _SHARED_INDEX = None
        yield from _search(index, names, k, 1, rare, rescore_n, chunk)
        return
    try:
        with ctx.Pool(jobs) as pool:
            for batch in pool.imap_unordered(_topk_worker, jobs_list):
                yield from batch
    finally:
        _SHARED_INDEX = None


def generate_candidates(queries: pd.DataFrame, pool, k: int = TOP_K,
                        max_df: float = MAX_DF, jobs: int = 1, chunk: int = QUERY_CHUNK,
                        log=print, progress_every: float = 60.0) -> Candidates:
    """Top-k candidates for every S1 in `queries`, country by country.

    `queries` needs entity_id/business_name/country. `pool` is the S2+S3 records, either
    as a DataFrame or as the {country: (ids, names)} mapping from split_pool_by_country,
    which lets the caller free the frame first.
    """
    rows_s1, rows_other, rows_score, rows_rank = [], [], [], []
    n_no_cand = 0
    stats = {}

    q_country = queries["country"].astype(str).to_numpy()
    q_ids = queries["entity_id"].astype(str).to_numpy()
    q_names = queries["business_name"].astype(str).to_numpy()
    by_country = pool if isinstance(pool, dict) else split_pool_by_country(pool)

    for country in sorted(set(q_country)):
        qm = q_country == country
        p_ids, p_names = by_country.get(country, (np.empty(0), np.empty(0)))
        n_q, n_p = int(qm.sum()), len(p_ids)
        if n_q == 0:
            continue
        if n_p == 0:
            # No pool for this country: every query legitimately gets nothing.
            n_no_cand += n_q
            stats[country] = {"queries": n_q, "pool": 0, "no candidate": n_q}
            log(f"  {country}: {n_q:,} queries, empty pool -> no candidates")
            continue

        log(f"  {country}: building index over {n_p:,} records "
            f"({n_q:,} queries, RSS {rss_gb():.1f} GB)")
        t0 = time.perf_counter()
        index = CountryIndex.build(country, p_ids, p_names, max_df=max_df)
        built = time.perf_counter() - t0
        log(f"  {country}: index ready in {built:.0f}s "
            f"(vocab {index.vocab:,}, {index.nnz_per_doc:.1f} nnz/doc, RSS {rss_gb():.1f} GB)"
            f" -- searching with {jobs} job(s)")

        t0 = time.perf_counter()
        ids_q = q_ids[qm]
        local_no_cand = 0
        done, last = 0, time.perf_counter()
        for r, cand_ids, val in _search(index, q_names[qm], k, jobs, chunk=chunk):
            done += 1
            if cand_ids:
                rows_s1.extend([ids_q[r]] * len(cand_ids))
                rows_other.extend(cand_ids)
                rows_score.extend(val)
                rows_rank.extend(range(1, len(cand_ids) + 1))
            else:
                local_no_cand += 1
            # Without this a country is silent for minutes and a hang is
            # indistinguishable from progress.
            now = time.perf_counter()
            if now - last >= progress_every:
                el = now - t0
                rate = done / el
                log(f"    {country}: {done:,}/{n_q:,} ({100 * done / n_q:.0f}%) "
                    f"{1000 / rate:.1f}s/1k, ETA {(n_q - done) / rate / 60:.0f} min, "
                    f"RSS {rss_gb():.1f} GB")
                last = now
        searched = time.perf_counter() - t0

        n_no_cand += local_no_cand
        stats[country] = {
            "queries": n_q, "pool": n_p, "vocab": index.vocab,
            "nnz per doc": round(index.nnz_per_doc, 2),
            "no candidate": local_no_cand,
            "no candidate %": round(100 * local_no_cand / n_q, 2),
            "build s": round(built, 1), "search s": round(searched, 1),
        }
        log(f"  {country}: {n_q:,} queries vs {n_p:,} pool | "
            f"{local_no_cand:,} with no candidate | {built:.0f}s build + {searched:.0f}s search"
            f" ({jobs} job{'s' if jobs > 1 else ''}, {1000 * searched / max(n_q, 1):.1f}s/1k)")
        del index

    frame = pd.DataFrame({
        "s1": rows_s1, "other": rows_other,
        "score": np.asarray(rows_score, dtype=np.float32),
        "rank": np.asarray(rows_rank, dtype=np.int8),
    })
    return Candidates(frame, len(queries), n_no_cand, stats)


# --- threshold tuning -----------------------------------------------------------

def tune_threshold(cands: Candidates, truth: dict[str, list[str]],
                   grid: np.ndarray | None = None) -> pd.DataFrame:
    """Macro F0.5 across a grid of cosine thresholds.

    Evaluated over every S1 in `truth`, so entities that blocking never reached are
    counted as empty predictions rather than quietly dropped.
    """
    from metrics import macro_f05

    if grid is None:
        grid = np.round(np.arange(0.0, 1.001, 0.025), 3)
    f = cands.frame
    rows = []
    for t in grid:
        sub = f[f["score"] >= t]
        pred = {k: list(v) for k, v in sub.groupby("s1", sort=False)["other"]}
        s = macro_f05(truth, pred)
        rows.append({"threshold": float(t), "macro F0.5": round(s.macro_f, 5),
                     "singletons": round(s.singleton_f, 5),
                     "non-singletons": round(s.non_singleton_f, 5),
                     "mean precision": round(s.mean_precision, 4),
                     "mean recall": round(s.mean_recall, 4),
                     "predicted empty %": round(100 * s.empty_pred_rate, 2),
                     "pairs kept": int(len(sub))})
    return pd.DataFrame(rows)


def best_threshold(tuning: pd.DataFrame) -> float:
    return float(tuning.loc[tuning["macro F0.5"].idxmax(), "threshold"])


def blocking_recall(cands: Candidates, truth: dict[str, list[str]]) -> dict:
    """Ceiling the candidate set imposes, before any threshold is applied."""
    got = cands.id_lists()
    tp = n_true = 0
    covered = 0
    non_single = 0
    for k, t in truth.items():
        t = set(t)
        if not t:
            continue
        non_single += 1
        c = set(got.get(k, ()))
        hit = len(t & c)
        tp += hit
        n_true += len(t)
        covered += bool(hit)
    return {
        "pair recall %": round(100 * tp / max(n_true, 1), 2),
        "entities with >=1 true candidate %": round(100 * covered / max(non_single, 1), 2),
        "candidate pairs": int(len(cands.frame)),
        "S1 with no candidate": cands.n_no_candidate,
        "S1 with no candidate %": round(100 * cands.n_no_candidate / max(cands.n_queries, 1), 2),
    }
