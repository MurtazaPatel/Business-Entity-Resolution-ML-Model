"""Validation splits: grouped 5-fold CV, plus leave-one-country-out.

Blocking runs once over the whole train pool; these splits then divide the resulting
pairs by S1 id. Every pair of one S1 entity therefore lands in exactly one fold, which is
what stops a near-duplicate of a training pair from appearing in validation.

Folds are stratified on (country, singleton flag) because both shift the score directly:
singletons are 5.58% of entities and each scores a flat 1.0 or 0.0, and the countries have
very different address conventions.

LOCO answers the question CV cannot: the test set contains France, a country absent from
train. Training on US and evaluating on India (and the reverse) estimates how much is lost
when the evaluation country was never seen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import SEED

N_SPLITS = 5
FOLD_COLUMNS = ["source1_entity_id", "country", "n_matches", "is_singleton", "fold"]


def build_fold_frame(gt: pd.DataFrame, s1: pd.DataFrame) -> pd.DataFrame:
    """One row per S1 entity with the fields the stratification needs."""
    from io_utils import parse_id_list

    n_matches = gt["matched_entity_ids"].map(lambda v: len(parse_id_list(v))).astype(np.int32)
    country = (s1.set_index(s1["entity_id"].astype(str))["country"].astype(str)
               .reindex(gt["source1_entity_id"].astype(str)).fillna("").to_numpy())
    return pd.DataFrame({
        "source1_entity_id": gt["source1_entity_id"].astype(str).to_numpy(),
        "country": country,
        "n_matches": n_matches.to_numpy(),
        "is_singleton": (n_matches == 0).to_numpy(),
    })


def assign_folds(frame: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED) -> pd.DataFrame:
    """Add a `fold` column via StratifiedGroupKFold grouped on the S1 id.

    Groups are the S1 ids themselves, so grouping is what guarantees no S1 spans folds
    once the frame is exploded into pairs.
    """
    out = frame.copy()
    strata = out["country"] + "|" + out["is_singleton"].map({True: "single", False: "matched"})
    groups = out["source1_entity_id"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(out), -1, dtype=np.int8)
    for k, (_, val) in enumerate(splitter.split(np.zeros(len(out)), strata.to_numpy(), groups)):
        fold[val] = k
    if (fold < 0).any():
        raise RuntimeError(f"{int((fold < 0).sum()):,} S1 entities were never assigned a fold")
    out["fold"] = fold
    return out[FOLD_COLUMNS]


def make_folds(paths, n_splits: int = N_SPLITS, seed: int = SEED, force: bool = False,
               cache: Path | None = None) -> pd.DataFrame:
    """Build (or reuse) artifacts/folds.parquet."""
    from io_utils import read_source

    cache = Path(cache) if cache is not None else paths.artifacts / "folds.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    gt = read_source(paths.train_gt)
    s1 = read_source(paths.train_s1)
    folds = assign_folds(build_fold_frame(gt, s1), n_splits, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(cache, index=False)
    return folds


# --- leave-one-country-out ------------------------------------------------------

def loco_splits(folds: pd.DataFrame) -> list[dict]:
    """Every ordered (train country -> eval country) pair present in the data.

    Derived from the labels in the data, never hard-coded, so adding a country needs no
    code change.
    """
    countries = sorted(c for c in folds["country"].unique() if c)
    return [
        {"name": f"{a}->{b}", "train_country": a, "eval_country": b,
         "n_train": int((folds["country"] == a).sum()),
         "n_eval": int((folds["country"] == b).sum())}
        for a in countries
        for b in countries
        if a != b
    ]


def loco_masks(folds: pd.DataFrame, split: dict) -> tuple[np.ndarray, np.ndarray]:
    return ((folds["country"] == split["train_country"]).to_numpy(),
            (folds["country"] == split["eval_country"]).to_numpy())


def fold_masks(folds: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    f = folds["fold"].to_numpy()
    return f != k, f == k


def filter_pairs(pairs: pd.DataFrame, folds: pd.DataFrame, mask: np.ndarray,
                 s1_col: str = "s1") -> pd.DataFrame:
    """Keep only the pairs whose S1 entity is selected by `mask`."""
    keep = set(folds.loc[mask, "source1_entity_id"])
    return pairs[pairs[s1_col].astype(str).isin(keep)]


# --- reporting -------------------------------------------------------------------

def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in folds.groupby("fold", sort=True):
        row = {"fold": int(k), "S1 entities": len(g),
               "singletons": int(g["is_singleton"].sum()),
               "singleton %": round(100 * g["is_singleton"].mean(), 3),
               "mean matches": round(float(g["n_matches"].mean()), 3),
               "true pairs": int(g["n_matches"].sum())}
        for c in sorted(x for x in folds["country"].unique() if x):
            row[f"{c} %"] = round(100 * (g["country"] == c).mean(), 2)
        rows.append(row)
    total = {"fold": "ALL", "S1 entities": len(folds),
             "singletons": int(folds["is_singleton"].sum()),
             "singleton %": round(100 * folds["is_singleton"].mean(), 3),
             "mean matches": round(float(folds["n_matches"].mean()), 3),
             "true pairs": int(folds["n_matches"].sum())}
    for c in sorted(x for x in folds["country"].unique() if x):
        total[f"{c} %"] = round(100 * (folds["country"] == c).mean(), 2)
    return pd.DataFrame(rows + [total])


def loco_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in loco_splits(folds):
        tr = folds[folds["country"] == s["train_country"]]
        ev = folds[folds["country"] == s["eval_country"]]
        rows.append({
            "split": s["name"],
            "train S1": len(tr), "train true pairs": int(tr["n_matches"].sum()),
            "train singleton %": round(100 * tr["is_singleton"].mean(), 3),
            "eval S1": len(ev), "eval true pairs": int(ev["n_matches"].sum()),
            "eval singleton %": round(100 * ev["is_singleton"].mean(), 3),
        })
    return pd.DataFrame(rows)


def check_no_leakage(folds: pd.DataFrame, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Assertions worth re-running on every rebuild."""
    per_entity = folds.groupby("source1_entity_id")["fold"].nunique()
    spanning = int((per_entity > 1).sum())
    dup = int(folds["source1_entity_id"].duplicated().sum())
    rates = folds.groupby("fold")["is_singleton"].mean() * 100
    spread = float(rates.max() - rates.min())

    # Even perfect stratification cannot beat integer rounding: each of the
    # (country x singleton) strata splits n_splits ways, so a fold can be off by about
    # one member per stratum. Tolerance tracks that instead of a flat constant, which
    # would fail on small samples and be far too slack on the real 2.2M rows.
    n_strata = folds.groupby(["country", "is_singleton"]).ngroups
    tol = max(0.05, 100.0 * n_splits * n_strata / max(len(folds), 1))

    checks = [
        ("every S1 in exactly one fold", spanning == 0, f"{spanning:,} span >1 fold"),
        ("no duplicate S1 rows", dup == 0, f"{dup:,} duplicates"),
        ("all folds non-empty", folds.groupby("fold").size().min() > 0, ""),
        (f"singleton rate spread <= {tol:.3f}pp", spread <= tol, f"spread {spread:.4f}pp"),
        ("no unlabelled country", (folds["country"] != "").all(),
         f"{int((folds['country'] == '').sum()):,} blank"),
    ]
    return pd.DataFrame([{"check": c, "pass": bool(ok), "detail": d} for c, ok, d in checks])
