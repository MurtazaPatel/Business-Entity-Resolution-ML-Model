"""Macro-averaged F0.5, exactly as PROBLEM.md defines it.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed per Source-1 entity and then averaged over every S1 entity in the evaluation
set. Singletons are included in that average, so how the degenerate cases are scored is
part of the metric, not an implementation detail:

    empty prediction, empty truth      -> 1.0   (correctly identified singleton)
    any prediction,   empty truth      -> 0.0   (false merge on a singleton)
    empty prediction, non-empty truth  -> 0.0   (missed everything)
    otherwise                          -> F0.5 from precision and recall

BETA = 0.5 weights precision twice as heavily as recall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

BETA = 0.5


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """F-measure. Returns 0.0 when precision and recall are both 0."""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(pred: Iterable[str], truth: Iterable[str], beta: float = BETA) -> float:
    """F0.5 for one S1 entity, including the singleton conventions above."""
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    return f_beta(tp / len(pred), tp / len(truth), beta)


@dataclass(frozen=True)
class Score:
    """Macro F0.5 overall and split by whether the entity has any true match."""

    macro_f: float
    n: int
    singleton_f: float
    n_singletons: int
    non_singleton_f: float
    n_non_singletons: int
    mean_precision: float
    mean_recall: float
    singleton_accuracy: float      # share of singletons predicted empty
    empty_pred_rate: float         # share of ALL entities predicted empty

    def as_dict(self) -> dict:
        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("macro F0.5 (all S1)", round(self.macro_f, 5), self.n),
            ("  singletons", round(self.singleton_f, 5), self.n_singletons),
            ("  non-singletons", round(self.non_singleton_f, 5), self.n_non_singletons),
            ("mean precision", round(self.mean_precision, 5), self.n),
            ("mean recall", round(self.mean_recall, 5), self.n),
            ("singletons predicted empty", round(self.singleton_accuracy, 5), self.n_singletons),
            ("all entities predicted empty", round(self.empty_pred_rate, 5), self.n),
        ]
        return pd.DataFrame(rows, columns=["metric", "value", "n"])

    def __str__(self) -> str:
        return (f"macro F0.5 {self.macro_f:.5f} over {self.n:,} S1 "
                f"(singletons {self.singleton_f:.5f} n={self.n_singletons:,}; "
                f"non-singletons {self.non_singleton_f:.5f} n={self.n_non_singletons:,})")


def macro_f05(
    truth: Mapping[str, Iterable[str]],
    pred: Mapping[str, Iterable[str]],
    beta: float = BETA,
) -> Score:
    """Macro F0.5 over every S1 in `truth`. A missing prediction counts as empty.

    `truth` defines the evaluation set: an S1 absent from `pred` scores as if it had been
    predicted empty, which is what a submission missing a row would effectively be.
    """
    scores = np.empty(len(truth), dtype=np.float64)
    precisions = np.zeros(len(truth), dtype=np.float64)
    recalls = np.zeros(len(truth), dtype=np.float64)
    is_single = np.zeros(len(truth), dtype=bool)
    pred_empty = np.zeros(len(truth), dtype=bool)

    for i, (key, t) in enumerate(truth.items()):
        t = set(t)
        p = set(pred.get(key, ()))
        is_single[i] = not t
        pred_empty[i] = not p
        scores[i] = score_entity(p, t, beta)
        if t and p:
            tp = len(p & t)
            precisions[i] = tp / len(p)
            recalls[i] = tp / len(t)
        elif not t and not p:
            precisions[i] = recalls[i] = 1.0

    n = len(truth)
    ns = int(is_single.sum())
    return Score(
        macro_f=float(scores.mean()) if n else 0.0,
        n=n,
        singleton_f=float(scores[is_single].mean()) if ns else float("nan"),
        n_singletons=ns,
        non_singleton_f=float(scores[~is_single].mean()) if n - ns else float("nan"),
        n_non_singletons=n - ns,
        mean_precision=float(precisions.mean()) if n else 0.0,
        mean_recall=float(recalls.mean()) if n else 0.0,
        singleton_accuracy=float(pred_empty[is_single].mean()) if ns else float("nan"),
        empty_pred_rate=float(pred_empty.mean()) if n else 0.0,
    )


# --- adapters -----------------------------------------------------------------

def id_lists_from_frame(df: pd.DataFrame, id_col: str | None = None) -> dict[str, list[str]]:
    """{source1_entity_id: [ids]} from a submission-shaped frame."""
    from io_utils import parse_id_list

    if id_col is None:
        others = [c for c in df.columns if c != "source1_entity_id"]
        if len(others) != 1:
            raise ValueError(f"cannot tell which column holds the ID lists: {list(df.columns)}")
        id_col = others[0]
    return {
        str(k): (list(v) if not isinstance(v, str) else parse_id_list(v))
        for k, v in zip(df["source1_entity_id"].astype(str), df[id_col])
    }


def pairs_to_id_lists(pairs: pd.DataFrame, s1_col: str = "s1",
                      other_col: str = "other") -> dict[str, list[str]]:
    """{s1: [ids]} from a long (s1, other) frame."""
    g = pairs.groupby(s1_col, sort=False)[other_col].apply(list)
    return {str(k): list(v) for k, v in g.items()}


def score_frames(truth_df: pd.DataFrame, pred_df: pd.DataFrame, beta: float = BETA) -> Score:
    """Score two submission-shaped frames against each other."""
    return macro_f05(id_lists_from_frame(truth_df), id_lists_from_frame(pred_df), beta)
