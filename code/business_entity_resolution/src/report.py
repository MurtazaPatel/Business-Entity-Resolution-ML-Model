"""Markdown helpers shared by every step's report."""

from __future__ import annotations

import math
import numbers

import pandas as pd


def _cell(v, max_cell: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, bool):
        v = str(v)
    elif isinstance(v, numbers.Integral):
        v = f"{int(v):,}"
    elif isinstance(v, numbers.Real):
        f = float(v)
        if f.is_integer() and abs(f) < 1e15:
            v = f"{int(f):,}"
        elif abs(f) < 1:
            v = f"{f:.4g}"
        else:
            v = f"{f:,.2f}".rstrip("0").rstrip(".")
    v = str(v).replace("|", "\\|").replace("\n", " ")
    return (v[: max_cell - 1] + "\u2026") if len(v) > max_cell else v


def md_table(df: pd.DataFrame, max_cell: int = 60) -> str:
    """Small markdown table. Hand-rolled so reports never depend on `tabulate`."""
    cols = [_cell(c, max_cell) for c in df.columns]
    rows = [[_cell(v, max_cell) for v in row] for row in df.itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)
