"""Reading the given TSVs and writing the two submission TSVs.

Two rules drive this module:
  * every field is a string and nothing is ever coerced to NaN (empty addresses are real
    values in this data), and
  * the big files are 5M rows each, so frames are backed by pyarrow strings rather than
    Python objects -- same values, roughly a fifth of the RAM.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import SEED

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def read_source(path: str | Path, nrows: int | None = None, backend: str = "pyarrow") -> pd.DataFrame:
    """Read a challenge TSV as all-string data with no NA coercion.

    backend="pyarrow" keeps 5M-row files near 0.6 GB instead of ~2 GB; pass
    backend="object" only when a caller genuinely needs plain Python str cells.
    """
    dtype = "string[pyarrow]" if backend == "pyarrow" else str
    return pd.read_csv(
        path,
        sep="\t",
        dtype=dtype,
        keep_default_na=False,
        na_filter=False,
        nrows=nrows,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )


def read_table(path: str | Path, columns: list[str] | None = None):
    """Arrow-native read for full-data passes: all string, never null, no quoting.

    Holding the data as a pyarrow Table and using pyarrow.compute avoids materializing
    millions of Python str objects, which is what pushed p1b to 9.3 GB.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(
            delimiter="\t", quote_char=False, double_quote=False, escape_char=False
        ),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in header},
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
            include_columns=columns,
        ),
    )


def count_rows(path: str | Path) -> int:
    """Data rows (header excluded), streamed so nothing large is held in memory."""
    with open(path, "rb") as f:
        total = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1024 * 1024), b""))
        f.seek(0, 2)
        size = f.tell()
    if size == 0:
        return 0
    # A file not ending in a newline still has a final row.
    with open(path, "rb") as f:
        f.seek(max(0, size - 1))
        if f.read(1) != b"\n":
            total += 1
    return max(0, total - 1)


def peek(path: str | Path, n: int = 3) -> pd.DataFrame:
    """The first n data rows, for display. Never reads the whole file."""
    return read_source(path, nrows=n)


def parse_id_list(value: str) -> list[str]:
    """Split a comma-joined ID list. Empty string means 'no matches', not a missing value."""
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [part for part in (p.strip() for p in value.split(",")) if part]


def format_id_list(ids: Iterable[str]) -> str:
    """Comma-join with no spaces, de-duplicated, order preserved.

    The validator splits on a bare ',' and does not strip, so a space after a comma would
    turn every ID after the first into an unknown ID.
    """
    seen: dict[str, None] = {}
    for i in ids:
        i = str(i).strip()
        if i:
            seen.setdefault(i, None)
    return ",".join(seen)


def write_submission(df: pd.DataFrame, path: str | Path, col_label: str = "matched") -> Path:
    """Write matching_results.tsv or candidate_pairs.tsv in exactly the spec format.

    df needs a source1_entity_id column plus an ID-list column, which may hold either
    pre-joined strings or lists/sets of IDs. col_label picks the header: "matched" or
    "candidate".
    """
    columns = MATCHING_COLUMNS if col_label == "matched" else CANDIDATE_COLUMNS
    id_col = columns[1]

    if "source1_entity_id" not in df.columns:
        raise ValueError("write_submission requires a 'source1_entity_id' column")

    # Accept whichever ID-list column the caller supplied.
    candidates = [c for c in df.columns if c != "source1_entity_id"]
    if id_col in df.columns:
        src_col = id_col
    elif len(candidates) == 1:
        src_col = candidates[0]
    else:
        raise ValueError(
            f"cannot tell which column holds the ID lists; expected '{id_col}', got {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "source1_entity_id": df["source1_entity_id"].astype(str),
            id_col: [
                v if isinstance(v, str) else format_id_list(v if v is not None else [])
                for v in df[src_col]
            ],
        }
    )
    # Normalize strings too: a caller may pass "S2-1, S2-2" with spaces.
    out[id_col] = [format_id_list(parse_id_list(v)) for v in out[id_col]]

    dupes = out["source1_entity_id"].duplicated()
    if dupes.any():
        raise ValueError(
            f"duplicate source1_entity_id rows: {out.loc[dupes, 'source1_entity_id'].head().tolist()}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        path,
        sep="\t",
        index=False,
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar=None,
        lineterminator="\n",
        encoding="utf-8",
    )
    return path


def gzip_file(path: str | Path, level: int = 6, keep: bool = True) -> Path:
    """Write `path`.gz beside `path`. Level 6 -- level 9 saves <1% here for 2.5x the time."""
    import gzip
    import shutil

    path = Path(path)
    out = path.with_suffix(path.suffix + ".gz")
    with open(path, "rb") as src, gzip.open(out, "wb", compresslevel=level) as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    if not keep:
        path.unlink()
    return out


def gunzip_file(path: str | Path, keep: bool = True) -> Path:
    """Expand `x.tsv.gz` back to `x.tsv`."""
    import gzip
    import shutil

    path = Path(path)
    if path.suffix != ".gz":
        raise ValueError(f"not a .gz file: {path}")
    out = path.with_suffix("")
    with gzip.open(path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    if not keep:
        path.unlink()
    return out


def load_sample(paths, n: int, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Smoke-run slice: n train S1 entities, their true matches, plus random S2/S3 filler.

    Keeps every S1 entity's full truth intact so recall/precision on the slice mean
    something, and pads S2/S3 with unrelated rows so blocking has real negatives to reject.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    s1 = read_source(paths.train_s1, nrows=max(n * 4, n))
    if len(s1) > n:
        idx = rng.choice(len(s1), size=n, replace=False)
        s1 = s1.iloc[sorted(idx)].reset_index(drop=True)
    keep_s1 = set(s1["entity_id"].astype(str))

    gt = read_source(paths.train_gt)
    gt = gt[gt["source1_entity_id"].astype(str).isin(keep_s1)].reset_index(drop=True)

    truth: set[str] = set()
    for v in gt["matched_entity_ids"]:
        truth.update(parse_id_list(v))

    out = {"source1": s1, "ground_truth": gt}
    for key, path, prefix in (
        ("source2", paths.train_s2, "S2-"),
        ("source3", paths.train_s3, "S3-"),
    ):
        wanted = {t for t in truth if t.startswith(prefix)}
        # One pass: keep every true match, plus a random sample of everything else.
        kept, filler_budget = [], max(n, 2000)
        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype="string[pyarrow]",
            keep_default_na=False,
            na_filter=False,
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
            chunksize=500_000,
        ):
            ids = chunk["entity_id"].astype(str)
            hits = chunk[ids.isin(wanted)]
            if len(hits):
                kept.append(hits)
            if filler_budget > 0:
                take = min(filler_budget, max(1, len(chunk) // 200))
                sel = rng.choice(len(chunk), size=min(take, len(chunk)), replace=False)
                kept.append(chunk.iloc[sorted(sel)])
                filler_budget -= take
        frame = pd.concat(kept, ignore_index=True) if kept else read_source(path, nrows=0)
        out[key] = frame.drop_duplicates(subset="entity_id").reset_index(drop=True)

    return out
