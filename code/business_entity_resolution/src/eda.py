"""Step p1: the dataset facts that decide the blocking and matching design.

Answers, in order:
  1. Is each S2/S3 record claimed by at most one S1? (CLAUDE.md asks us to verify this)
  2. How many matches does an S1 entity have, and how many records go unmatched?
  3. Do true matches ever cross country? -- decides whether same_country can gate blocking
  4. How noisy are the fields (empty, non-Latin, digit-free)?
  5. For each candidate blocking key, what share of true pairs does it recover?
     This is the recall ceiling of a blocking scheme, and the whole point of the step.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import normalize as nz
from io_utils import parse_id_list, read_source


def _pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


def gt_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Explode ground truth into one row per (s1, matched) pair."""
    s1 = gt["source1_entity_id"].astype(str).tolist()
    lists = [parse_id_list(v) for v in gt["matched_entity_ids"]]
    left = np.repeat(s1, [len(x) for x in lists])
    right = [i for sub in lists for i in sub]
    return pd.DataFrame({"s1": left, "other": right})


def check_one_s1_per_record(pairs: pd.DataFrame) -> list[str]:
    """The exclusivity invariant. If it holds, a matched record can be claimed once only."""
    n = len(pairs)
    dup = pairs["other"].duplicated(keep=False)
    n_dup_rows = int(dup.sum())
    lines = ["#### 1. Exclusivity: does each S2/S3 record belong to at most one S1?", ""]
    if n_dup_rows == 0:
        lines += [
            f"- **HOLDS.** All {n:,} matched IDs are distinct.",
            "- Exploitable: a record may be assigned to at most one S1, so scoring can be",
            "  a global assignment rather than independent per-pair decisions.",
        ]
    else:
        offenders = pairs.loc[dup, "other"]
        top = offenders.value_counts().head(5)
        lines += [
            f"- **VIOLATED.** {n_dup_rows:,} of {n:,} matched IDs ({_pct(n_dup_rows, n)}) "
            "appear under more than one S1.",
            f"- Worst offenders: {', '.join(f'{k} x{v}' for k, v in top.items())}",
            "- Do NOT assume exclusivity.",
        ]
    lines.append("")
    return lines


def match_count_stats(gt: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    counts = gt["matched_entity_ids"].map(lambda v: len(parse_id_list(v)))
    n = len(gt)
    singles = int((counts == 0).sum())
    s2 = int(pairs["other"].str.startswith("S2-").sum())
    s3 = len(pairs) - s2

    lines = ["#### 2. How many matches per S1 entity?", ""]
    lines += [
        f"- S1 entities: **{n:,}**   matched IDs: **{len(pairs):,}** (S2 {s2:,} / S3 {s3:,})",
        f"- singletons (no true match): **{singles:,}** = **{_pct(singles, n)}**",
        f"- matches per non-singleton: mean **{counts[counts > 0].mean():.2f}**, "
        f"median {int(counts[counts > 0].median())}, max **{int(counts.max())}**",
        "",
        "| matches | S1 entities | share |",
        "| --- | --- | --- |",
    ]
    hist = counts.value_counts().sort_index()
    for k, v in hist.items():
        lines.append(f"| {k} | {v:,} | {_pct(v, n)} |")
    lines += [
        "",
        f"> Singletons are only {_pct(singles, n)} of entities, so a blanket "
        "'predict empty' baseline scores about that. Recall matters more than the "
        "F0.5 framing alone suggests.",
        "",
    ]
    return lines


def coverage_stats(pairs: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> list[str]:
    matched = set(pairs["other"])
    lines = ["#### 3. What share of S2/S3 records is matched to any S1?", ""]
    for name, frame in (("S2", s2), ("S3", s3)):
        ids = set(frame["entity_id"].astype(str))
        hit = len(ids & matched)
        lines.append(
            f"- **{name}**: {len(ids):,} records, {hit:,} matched ({_pct(hit, len(ids))}) "
            f"-> {_pct(len(ids) - hit, len(ids))} are distractors with no S1."
        )
    missing = len(matched - set(s2["entity_id"].astype(str)) - set(s3["entity_id"].astype(str)))
    lines += [
        f"- ground-truth IDs absent from the source files: **{missing:,}** "
        "(should be 0)",
        "",
    ]
    return lines


def field_quality(frames: dict[str, pd.DataFrame]) -> list[str]:
    lines = [
        "#### 4. Field quality",
        "",
        "| frame | rows | empty name | empty addr | non-Latin name | addr w/o digit | postal found |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, df in frames.items():
        n = len(df)
        if n == 0:
            continue
        nm = df["business_name"].astype(str)
        ad = df["business_address"].astype(str)
        # Sample the expensive per-string checks; these are rates, not counts.
        k = min(n, 200_000)
        idx = np.linspace(0, n - 1, k).astype(int)
        nm_s, ad_s = nm.iloc[idx], ad.iloc[idx]
        lines.append(
            f"| {name} | {n:,} | {_pct((nm == '').sum(), n)} | {_pct((ad == '').sum(), n)} "
            f"| {_pct(sum(nz.is_non_latin(v) for v in nm_s), k)} "
            f"| {_pct(sum(not nz.has_digit(v) for v in ad_s), k)} "
            f"| {_pct(sum(bool(nz.postal_codes(v)) for v in ad_s), k)} |"
        )
    lines.append("")
    return lines


def build_id_country(frames) -> pd.Series:
    """entity_id -> country as a Series with a hash index.

    A plain dict over 12.5M records costs several GB; an indexed Series with the country
    as a category is a few hundred MB and joins in C.
    """
    parts = [
        pd.Series(
            f["country"].astype(str).values,
            index=f["entity_id"].astype(str).values,
        )
        for f in frames
    ]
    out = pd.concat(parts)
    return out.astype("category")


def country_agreement(pairs: pd.DataFrame, id_country: pd.Series) -> list[str]:
    """Do true matches share a country? Decides if same_country may gate blocking."""
    left = id_country.reindex(pairs["s1"].values).astype(object).values
    right = id_country.reindex(pairs["other"].values).astype(object).values

    known = pd.notna(left) & pd.notna(right)
    same = int((known & (left == right)).sum())
    cross = int((known & (left != right)).sum())
    unknown = int((~known).sum())
    total = same + cross

    lines = ["#### 5. Do true matches cross country?", ""]
    lines += [
        f"- same country: **{same:,}** ({_pct(same, total)})",
        f"- different country: **{cross:,}** ({_pct(cross, total)})",
        f"- unresolved IDs: {unknown:,}",
    ]
    if cross:
        idx = np.nonzero(known & (left != right))[0][:5]
        ex = [
            f"{pairs['s1'].iloc[i]}({left[i]}) ~ {pairs['other'].iloc[i]}({right[i]})"
            for i in idx
        ]
        lines.append(f"- examples: {'; '.join(ex)}")
    verdict = (
        "blocking within country costs ~0 recall -- partition the problem by country label "
        "(a same_country constraint, not a hard-coded country list)"
        if total and cross / total < 0.001
        else "do NOT block on country; use same_country only as a feature"
    )
    lines += [f"- **verdict: {verdict}**", ""]
    return lines


BLOCKING_KEYS = {
    "exact normalized name": lambda a, b: nz.normalize_name(a[0]) == nz.normalize_name(b[0]),
    "name tokens sorted": lambda a, b: nz.name_sort_key(a[0]) == nz.name_sort_key(b[0]),
    "share >=1 name token": lambda a, b: bool(nz.name_tokens(a[0]) & nz.name_tokens(b[0])),
    "share >=2 name tokens": lambda a, b: len(nz.name_tokens(a[0]) & nz.name_tokens(b[0])) >= 2,
    "share a name trigram": lambda a, b: bool(nz.char_ngrams(a[0]) & nz.char_ngrams(b[0])),
    "name trigram jaccard>=0.3": lambda a, b: _jac(nz.char_ngrams(a[0]), nz.char_ngrams(b[0])) >= 0.3,
    "share a postal code": lambda a, b: bool(nz.postal_codes(a[1]) & nz.postal_codes(b[1])),
    "share a street number": lambda a, b: bool(nz.street_numbers(a[1]) & nz.street_numbers(b[1])),
    "share >=1 addr token": lambda a, b: bool(nz.address_tokens(a[1]) & nz.address_tokens(b[1])),
    "same country": lambda a, b: a[2] == b[2],
}


def _jac(x: set, y: set) -> float:
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def blocking_recall(
    pairs: pd.DataFrame, frames, probe: int, seed: int
) -> list[str]:
    """Per-key recall on true pairs: the recall ceiling each blocking key can deliver.

    Samples the pairs first, then materializes record text for those IDs only, so peak
    memory tracks the probe size rather than the 12.5M-record corpus.
    """
    rng = np.random.default_rng(seed)
    if len(pairs) > probe:
        idx = np.sort(rng.choice(len(pairs), size=probe, replace=False))
        sub = pairs.iloc[idx]
    else:
        sub = pairs

    needed = set(sub["s1"].astype(str)) | set(sub["other"].astype(str))
    lookup: dict[str, tuple[str, str, str]] = {}
    for f in frames:
        ids = f["entity_id"].astype(str)
        hit = f[ids.isin(needed)]
        lookup.update(
            zip(
                hit["entity_id"].astype(str),
                zip(
                    hit["business_name"].astype(str),
                    hit["business_address"].astype(str),
                    hit["country"].astype(str),
                ),
            )
        )

    resolved = [
        (lookup[a], lookup[b])
        for a, b in zip(sub["s1"].astype(str), sub["other"].astype(str))
        if a in lookup and b in lookup
    ]
    del lookup
    n = len(resolved)
    lines = [
        f"#### 6. Blocking-key recall on true pairs (probe n={n:,})",
        "",
        "Share of true pairs each key would keep. A key below ~90% cannot be the sole",
        "blocker; the union of cheap keys sets the achievable recall ceiling.",
        "",
        "| blocking key | recall on true pairs |",
        "| --- | --- |",
    ]
    for key, fn in BLOCKING_KEYS.items():
        h = sum(1 for a, b in resolved if fn(a, b))
        lines.append(f"| {key} | **{_pct(h, n)}** |")

    unions = {
        "name token OR name trigram": ["share >=1 name token", "share a name trigram"],
        "name trigram OR addr token": ["share a name trigram", "share >=1 addr token"],
        "name token OR addr token": ["share >=1 name token", "share >=1 addr token"],
        "name trigram OR addr token OR street number": [
            "share a name trigram", "share >=1 addr token", "share a street number",
        ],
    }
    lines += ["", "**Union schemes** (pair kept if any key fires):", "",
              "| scheme | recall |", "| --- | --- |"]
    best_fns = None
    for label, keys in unions.items():
        fns = [BLOCKING_KEYS[k] for k in keys]
        u = sum(1 for a, b in resolved if any(f(a, b) for f in fns))
        lines.append(f"| {label} | **{_pct(u, n)}** |")
        if label == "name token OR addr token":
            best_fns = fns
    lines.append("")

    miss = [(a, b) for a, b in resolved if not any(f(a, b) for f in best_fns)]
    if miss:
        lines += [
            f"Pairs that survive NO key in 'name token OR addr token' "
            f"({len(miss):,} of {n:,} = {_pct(len(miss), n)}) -- these bound recall:",
            "",
        ]
        for a, b in miss[:10]:
            lines.append(f"- `{a[0]}` @ `{a[1]}` **vs** `{b[0]}` @ `{b[1]}`")
        lines.append("")
    return lines


def token_frequency(s1: pd.DataFrame, probe: int, seed: int) -> list[str]:
    """How discriminative are name tokens? Drives which tokens are usable as block keys."""
    rng = np.random.default_rng(seed)
    n = len(s1)
    k = min(n, probe)
    idx = rng.choice(n, size=k, replace=False) if n > k else np.arange(n)
    counter: Counter[str] = Counter()
    for v in s1["business_name"].astype(str).iloc[np.sort(idx)]:
        counter.update(nz.name_tokens(v))
    total = sum(counter.values())
    lines = [
        f"#### 7. Name-token frequency (probe n={k:,} S1 names)",
        "",
        f"- distinct tokens: **{len(counter):,}**   occurrences: {total:,}",
        f"- tokens appearing once: {_pct(sum(1 for c in counter.values() if c == 1), len(counter))} "
        "of the vocabulary",
        "",
        "Most common tokens (these are near-useless as block keys):",
        "",
        "| token | count |",
        "| --- | --- |",
    ]
    for tok, c in counter.most_common(15):
        lines.append(f"| {tok} | {c:,} |")
    lines.append("")
    return lines


def run(paths, args) -> list[str]:
    """Assemble the report. Uses --sample to stay runnable on a laptop."""
    probe = getattr(args, "probe", 200_000) or 200_000
    seed = args.seed

    if args.sample:
        from io_utils import load_sample

        data = load_sample(paths, args.sample, seed=seed)
        s1, s2, s3, gt = data["source1"], data["source2"], data["source3"], data["ground_truth"]
        scope = f"--sample {args.sample} (train slice)"
    else:
        s1 = read_source(paths.train_s1)
        s2 = read_source(paths.train_s2)
        s3 = read_source(paths.train_s3)
        gt = read_source(paths.train_gt)
        scope = "full train"

    pairs = gt_pairs(gt)
    frames = (s1, s2, s3)

    lines = [f"**scope:** {scope}", ""]
    lines += check_one_s1_per_record(pairs)
    lines += match_count_stats(gt, pairs)
    lines += coverage_stats(pairs, s2, s3)
    lines += field_quality({"train_s1": s1, "train_s2": s2, "train_s3": s3})

    id_country = build_id_country(frames)
    lines += country_agreement(pairs, id_country)
    del id_country

    lines += blocking_recall(pairs, frames, probe, seed)
    lines += token_frequency(s1, probe, seed)
    return lines


# =============================================================================
# p1c_eda -- deep EDA. notebooks/01_eda.ipynb only displays the tables this writes.
#
# Arrow-native: every full-corpus pass runs in pyarrow.compute, and only small sampled
# sets are ever turned into Python objects. p1b peaked at 9.3 GB by converting whole
# columns to object dtype; this step should stay well under that.
# =============================================================================

import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from rapidfuzz import utils as rf_utils  # noqa: E402

from config import SEED  # noqa: E402
from io_utils import count_rows  # noqa: E402
from report import md_table  # noqa: E402

SPLIT_SOURCES = [
    ("train", "s1"), ("train", "s2"), ("train", "s3"),
    ("test", "s1"), ("test", "s2"), ("test", "s3"),
]

# Address formats, detected from the text only (RE2 syntax, run by pyarrow).
FORMAT_PATTERNS = {
    "5-digit": r"(^|[^0-9])[0-9]{5}([^0-9]|$)",
    "6-digit": r"(^|[^0-9])[0-9]{6}([^0-9]|$)",
    "ddd ddd": r"(^|[^0-9])[0-9]{3} [0-9]{3}([^0-9]|$)",
    # A 5-digit run opening the address is usually a house number, not a postal code.
    "5-digit at start": r"^\s*[0-9]{5}([^0-9]|$)",
}
LANDMARK = r"\b(near|nr|opp|opposite|behind)\b"
TOKEN_SPLIT = r"[^\p{L}\p{N}]+"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- loading -----------------------------------------------------------------

@dataclass
class EdaData:
    train: dict
    test: dict
    gt: pa.Table
    scope: str
    sample: bool

    def split(self, name: str) -> dict:
        return self.train if name == "train" else self.test


def _to_arrow(df: pd.DataFrame) -> pa.Table:
    return pa.table({c: pa.array(df[c].astype(str).tolist(), pa.string()) for c in df.columns})


def load_eda_data(paths, sample: int | None = None, seed: int = SEED) -> EdaData:
    from io_utils import load_sample, read_table

    if sample:
        d = load_sample(paths, sample, seed=seed)
        train = {"s1": _to_arrow(d["source1"]), "s2": _to_arrow(d["source2"]),
                 "s3": _to_arrow(d["source3"])}
        gt = _to_arrow(d["ground_truth"])
        k = max(sample * 5, 5000)
        test = {s: _to_arrow(read_source(p, nrows=k))
                for s, p in (("s1", paths.test_s1), ("s2", paths.test_s2), ("s3", paths.test_s3))}
        scope = f"--sample {sample}: train slice + first {k:,} rows of each test file"
    else:
        train = {s: read_table(p) for s, p in
                 (("s1", paths.train_s1), ("s2", paths.train_s2), ("s3", paths.train_s3))}
        test = {s: read_table(p) for s, p in
                (("s1", paths.test_s1), ("s2", paths.test_s2), ("s3", paths.test_s3))}
        gt = read_table(paths.train_gt)
        scope = "full train + full test"
    return EdaData(train, test, gt, scope, bool(sample))


# --- small helpers -------------------------------------------------------------

def _np(x) -> np.ndarray:
    if isinstance(x, pa.ChunkedArray):
        x = x.combine_chunks()
    return x.to_numpy(zero_copy_only=False)


def _rate(mask) -> float:
    n = len(mask)
    return float(pc.sum(mask).as_py() or 0) / n if n else float("nan")


def _codes(col) -> tuple[np.ndarray, list[str]]:
    """Integer code per row plus the label for each code (dictionary encoding)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    enc = pc.dictionary_encode(col)
    return _np(enc.indices), enc.dictionary.to_pylist()


def _records(tables: dict, ids) -> dict:
    """entity_id -> row dict (+ 'source') for a small set of IDs."""
    ids = [i for i in dict.fromkeys(ids) if i]
    if not ids:
        return {}
    vs = pa.array(ids, pa.string())
    out = {}
    for src, t in tables.items():
        for r in t.filter(pc.is_in(t.column("entity_id"), value_set=vs)).to_pylist():
            r["source"] = src.upper()
            out[r["entity_id"]] = r
    return out


def _tsr(a: str, b: str) -> float:
    """rapidfuzz token_set_ratio on the raw strings (lowercased, punctuation stripped)."""
    return round(float(fuzz.token_set_ratio(a or "", b or "", processor=rf_utils.default_process)), 1)


def _tsr_name(a: str, b: str) -> float:
    return round(float(fuzz.token_set_ratio(nz.normalize_name(a or ""), nz.normalize_name(b or ""))), 1)


def _tsr_addr(a: str, b: str) -> float:
    return round(float(fuzz.token_set_ratio(nz.normalize_address(a or ""),
                                            nz.normalize_address(b or ""))), 1)


# --- ground truth ---------------------------------------------------------------

def gt_pairs_arrow(gt: pa.Table) -> pa.Table:
    """(s1, other) for every true pair, built without Python strings."""
    s1 = gt.column("source1_entity_id").combine_chunks()
    lists = pc.split_pattern(gt.column("matched_entity_ids").combine_chunks(), ",")
    flat = pc.utf8_trim_whitespace(pc.list_flatten(lists))
    parent = pc.list_parent_indices(lists)
    keep = pc.not_equal(flat, "")
    return pa.table({"s1": s1.take(parent.filter(keep)), "other": flat.filter(keep)})


def match_counts(gt: pa.Table) -> np.ndarray:
    m = gt.column("matched_entity_ids")
    n = pc.add(pc.count_substring(m, ","), 1)
    return _np(pc.if_else(pc.equal(pc.utf8_trim_whitespace(m), ""), 0, n))


def section_ground_truth(d: EdaData, pairs: pa.Table):
    counts = match_counts(d.gt)
    n = len(counts)
    bins = [("0", counts == 0), ("1", counts == 1), ("2", counts == 2), ("3", counts == 3),
            ("4", counts == 4), ("5+", counts >= 5)]
    hist = pd.DataFrame([{"matches": k, "S1 entities": int(m.sum()),
                          "share %": round(100 * m.mean(), 2)} for k, m in bins])

    other = pairs.column("other").combine_chunks()
    tot = len(other)
    s2 = int(pc.sum(pc.starts_with(other, "S2-")).as_py() or 0)
    rows = [
        ("S1 entities", n),
        ("singletons %", round(100 * (counts == 0).mean(), 2)),
        ("mean matches per non-singleton", round(float(counts[counts > 0].mean()), 2)),
        ("max matches", int(counts.max())),
        ("matched IDs", tot),
        ("matched IDs from S2 %", round(100 * s2 / tot, 2)),
        ("matched IDs from S3 %", round(100 * (tot - s2) / tot, 2)),
    ]
    for src in ("s2", "s3"):
        hit = pc.is_in(d.train[src].column("entity_id"), value_set=other)
        rows.append((f"{src.upper()} records matching no S1 %", round(100 * (1 - _rate(hit)), 2)))
    note = " (sample: filler inflates this)" if d.sample else ""
    summary = pd.DataFrame(rows, columns=["metric", "value"], dtype=object)
    summary["metric"] = [m + (note if "no S1" in m else "") for m in summary["metric"]]
    return {"gt_summary": summary, "gt_match_hist": hist}, counts


def section_ownership(pairs: pa.Table):
    other = pairs.column("other").combine_chunks()
    vc = pc.value_counts(other)
    counts = _np(vc.field("counts"))
    dup = np.nonzero(counts > 1)[0]
    summary = pd.DataFrame(
        [("matched IDs", len(other)), ("IDs under more than one S1", len(dup)),
         ("max owners of a single ID", int(counts.max()) if len(counts) else 0)],
        columns=["metric", "value"], dtype=object,
    )
    examples = pd.DataFrame(columns=["entity_id", "owners", "owner S1 ids"])
    if len(dup):
        dup_ids = vc.field("values").take(pa.array(dup[:10]))
        sub = pairs.filter(pc.is_in(pairs.column("other"), value_set=dup_ids)).to_pandas()
        examples = (sub.groupby("other")["s1"]
                    .agg(owners="count", **{"owner S1 ids": lambda s: ",".join(sorted(s))})
                    .reset_index().rename(columns={"other": "entity_id"}))
    return {"ownership": summary, "ownership_examples": examples}, len(dup)


def section_sizes(paths) -> pd.DataFrame:
    rows = []
    for name, tr, te in (("source1", paths.train_s1, paths.test_s1),
                         ("source2", paths.train_s2, paths.test_s2),
                         ("source3", paths.train_s3, paths.test_s3),
                         ("ground_truth", paths.train_gt, None)):
        a = count_rows(tr)
        b = count_rows(te) if te else None
        rows.append({"file": name, "train rows": a, "test rows": b,
                     "test/train": round(b / a, 3) if b else None,
                     "train MB": round(tr.stat().st_size / 1024**2, 1),
                     "test MB": round(te.stat().st_size / 1024**2, 1) if te else None})
    return pd.DataFrame(rows)


def section_country(d: EdaData, pairs: pa.Table):
    long = []
    for split, src in SPLIT_SOURCES:
        t = d.split(split)[src]
        vc = pc.value_counts(t.column("country").combine_chunks())
        for v, c in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
            long.append({"split": split, "source": src.upper(), "country": v, "count": c,
                         "share %": round(100 * c / t.num_rows, 2)})
    long = pd.DataFrame(long)
    wide = (long.assign(cell=long["count"].map("{:,}".format) + " ("
                        + long["share %"].map("{:.1f}".format) + "%)")
            .pivot_table(index=["split", "source"], columns="country", values="cell",
                         aggfunc="first")
            .fillna("-").reset_index())
    wide.columns.name = None

    train_labels = sorted(set(long.loc[long.split == "train", "country"]))
    new = sorted(set(long.loc[long.split == "test", "country"]) - set(train_labels))
    new_df = long[(long.split == "test") & long.country.isin(new)][
        ["source", "country", "count", "share %"]].reset_index(drop=True)

    s1c = d.train["s1"].select(["entity_id", "country"]).rename_columns(["s1", "c1"])
    oc = pa.concat_tables([d.train[s].select(["entity_id", "country"]) for s in ("s2", "s3")]
                          ).rename_columns(["other", "c2"])
    j = pairs.join(s1c, "s1", join_type="inner").join(oc, "other", join_type="inner")
    differ = int(pc.sum(pc.not_equal(j.column("c1"), j.column("c2"))).as_py() or 0)
    resolved = j.num_rows
    del j, oc
    cross = pd.DataFrame(
        [("true pairs", pairs.num_rows), ("pairs resolved to both records", resolved),
         ("country differs inside pair", differ),
         ("country differs %", round(100 * differ / max(resolved, 1), 4))],
        columns=["metric", "value"], dtype=object,
    )

    agg = long.groupby("country", as_index=False)["count"].sum()
    agg["normalized key"] = agg["country"].map(lambda s: " ".join(s.split()).casefold())
    agg["raw label (repr)"] = agg["country"].map(repr)
    agg["raw spellings of key"] = agg.groupby("normalized key")["country"].transform("count")
    variants = agg[["raw label (repr)", "normalized key", "count", "raw spellings of key"]
                   ].sort_values("count", ascending=False).reset_index(drop=True)

    tables = {"country_counts": wide, "country_new_in_test": new_df,
              "country_in_pairs": cross, "country_variants": variants}
    return tables, train_labels, new, differ


def section_fields(d: EdaData) -> dict:
    empty_rows, len_rows = [], []
    for split, src in SPLIT_SOURCES:
        t = d.split(split)[src]
        row = {"split": split, "source": src.upper(), "rows": t.num_rows}
        for c in ("entity_id", "business_name", "business_address", "country"):
            row[f"{c} empty %"] = round(
                100 * _rate(pc.equal(pc.utf8_trim_whitespace(t.column(c)), "")), 3)
        empty_rows.append(row)

        codes, labels = _codes(t.column("country"))
        nl = _np(pc.utf8_length(t.column("business_name")))
        al = _np(pc.utf8_length(t.column("business_address")))
        for k, lab in enumerate(labels):
            m = codes == k
            a = al[m]
            a = a[a > 0]
            len_rows.append({
                "split": split, "source": src.upper(), "country": lab, "rows": int(m.sum()),
                "name len median": float(np.median(nl[m])),
                "name len p95": float(np.percentile(nl[m], 95)),
                "addr len median": float(np.median(a)) if len(a) else float("nan"),
                "addr len p95": float(np.percentile(a, 95)) if len(a) else float("nan"),
            })
    return {"field_empty": pd.DataFrame(empty_rows), "field_lengths": pd.DataFrame(len_rows)}


def section_formats(d: EdaData) -> pd.DataFrame:
    rows = []
    for split, src in SPLIT_SOURCES:
        t = d.split(split)[src]
        addr = t.column("business_address")
        codes, labels = _codes(t.column("country"))
        flags = {k: _np(pc.match_substring_regex(addr, p)) for k, p in FORMAT_PATTERNS.items()}
        flags["near/opp/behind"] = _np(pc.match_substring_regex(addr, LANDMARK, ignore_case=True))
        for k, lab in enumerate(labels):
            m = codes == k
            row = {"split": split, "source": src.upper(), "country": lab, "rows": int(m.sum())}
            for name, f in flags.items():
                row[f"{name} %"] = round(100 * f[m].mean(), 2)
            rows.append(row)
    return pd.DataFrame(rows)


# --- samples ------------------------------------------------------------------------

def sample_true_pairs(d: EdaData, pairs: pa.Table, n: int = 30, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(pairs.num_rows, size=min(n, pairs.num_rows), replace=False))
    sub = pairs.take(pa.array(idx)).to_pylist()
    rec = _records(d.train, [p["s1"] for p in sub] + [p["other"] for p in sub])
    rows = []
    for p in sub:
        a, b = rec.get(p["s1"]), rec.get(p["other"])
        if not a or not b:
            continue
        rows.append({
            "s1 id": a["entity_id"], "s1 name": a["business_name"],
            "s1 address": a["business_address"], "s1 country": a["country"],
            "match id": b["entity_id"], "match name": b["business_name"],
            "match address": b["business_address"], "match country": b["country"],
            "name tsr": _tsr(a["business_name"], b["business_name"]),
            "addr tsr": _tsr(a["business_address"], b["business_address"]),
            "name tsr (normalized)": _tsr_name(a["business_name"], b["business_name"]),
            "addr tsr (normalized)": _tsr_addr(a["business_address"], b["business_address"]),
        })
    return pd.DataFrame(rows)


def singleton_traps(d: EdaData, pairs: pa.Table, counts: np.ndarray, n_show: int = 15,
                    n_stats: int = 300, seed: int = SEED, chunk: int = 100_000):
    """Nearest S2/S3 record by char-3gram TF-IDF name cosine, within the S1's own country.

    Singletons are the traps: every candidate is wrong. Matched S1s are the control group.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = np.random.default_rng(seed)
    gt_ids = d.gt.column("source1_entity_id")
    single_idx, match_idx = np.nonzero(counts == 0)[0], np.nonzero(counts > 0)[0]
    qs = rng.choice(single_idx, size=min(n_stats, len(single_idx)), replace=False)
    qm = rng.choice(match_idx, size=min(n_stats, len(match_idx)), replace=False)
    q_single = gt_ids.take(pa.array(np.sort(qs))).to_pylist()
    q_match = gt_ids.take(pa.array(np.sort(qm))).to_pylist()
    qrec = _records({"s1": d.train["s1"]}, q_single + q_match)
    queries = ([(q, "singleton") for q in q_single if q in qrec]
               + [(q, "matched") for q in q_match if q in qrec])

    truth = defaultdict(set)
    for r in pairs.filter(pc.is_in(pairs.column("s1"),
                                   value_set=pa.array(q_match, pa.string()))).to_pylist():
        truth[r["s1"]].add(r["other"])

    cand = pa.concat_tables([d.train[s].select(["entity_id", "business_name", "country"])
                             for s in ("s2", "s3")])
    cand_ids, cand_names = cand.column("entity_id"), cand.column("business_name")
    codes, labels = _codes(cand.column("country"))

    fit_idx = np.sort(rng.choice(cand.num_rows, size=min(300_000, cand.num_rows), replace=False))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                          dtype=np.float32, sublinear_tf=True)
    vec.fit([nz.normalize_name(x) for x in cand_names.take(pa.array(fit_idx)).to_pylist()])

    best_score = {q: -1.0 for q, _ in queries}
    best_pos = {q: -1 for q, _ in queries}
    by_country = defaultdict(list)
    for q, _ in queries:
        by_country[qrec[q]["country"]].append(q)

    for country, qids in by_country.items():
        if country not in labels:
            continue
        pos = np.nonzero(codes == labels.index(country))[0]
        qt = vec.transform([nz.normalize_name(qrec[q]["business_name"]) for q in qids]).T.tocsr()
        _log(f"  traps: {country}: {len(qids)} queries vs {len(pos):,} candidates")
        for start in range(0, len(pos), chunk):
            p = pos[start:start + chunk]
            names = [nz.normalize_name(x) for x in cand_names.take(pa.array(p)).to_pylist()]
            sims = (vec.transform(names) @ qt).tocsc()
            if sims.nnz == 0:
                continue
            mx = sims.max(axis=0).toarray().ravel()
            am = np.asarray(sims.argmax(axis=0)).ravel()
            for j, q in enumerate(qids):
                if mx[j] > best_score[q]:
                    best_score[q], best_pos[q] = float(mx[j]), int(p[am[j]])
    del cand, codes

    found = [q for q, _ in queries if best_pos[q] >= 0]
    nn_ids = dict(zip(found, cand_ids.take(pa.array([best_pos[q] for q in found])).to_pylist()))
    owned = set()
    if nn_ids:
        vals = pa.array(list(nn_ids.values()), pa.string())
        mask = pc.is_in(vals, value_set=pairs.column("other").combine_chunks())
        owned = {v for v, m in zip(vals.to_pylist(), mask.to_pylist()) if m}
    nrec = _records({"s2": d.train["s2"], "s3": d.train["s3"]}, list(nn_ids.values()))

    rows = []
    for q, grp in queries:
        a, nid = qrec[q], nn_ids.get(q)
        b = nrec.get(nid, {}) if nid else {}
        is_true = nid in truth[q] if grp == "matched" else False
        rows.append({
            "group": grp, "s1 id": q, "s1 name": a["business_name"],
            "s1 address": a["business_address"], "country": a["country"],
            "nn id": nid, "nn source": b.get("source"), "nn name": b.get("business_name"),
            "nn address": b.get("business_address"),
            "cosine": round(best_score[q], 4) if nid else None,
            "name tsr": _tsr(a["business_name"], b.get("business_name", "")),
            "addr tsr": _tsr(a["business_address"], b.get("business_address", "")),
            "nn is true match": is_true if grp == "matched" else None,
            "nn owned by another S1": bool(nid in owned and not is_true) if nid else None,
        })
    all_rows = pd.DataFrame(rows)

    stats = []
    for grp, g in all_rows.groupby("group"):
        c = g["cosine"].dropna().astype(float)
        stats.append({
            "group": grp, "queries": len(g),
            "cosine median": round(c.median(), 3), "cosine p90": round(c.quantile(0.9), 3),
            "cosine >= 0.8 %": round(100 * (c >= 0.8).mean(), 1),
            "cosine >= 0.9 %": round(100 * (c >= 0.9).mean(), 1),
            "nn is true match %": (round(100 * g["nn is true match"].astype(bool).mean(), 1)
                                   if grp == "matched" else None),
            "nn owned by another S1 %": round(
                100 * g["nn owned by another S1"].fillna(False).astype(bool).mean(), 1),
        })
    top = (all_rows[all_rows.group == "singleton"].sort_values("cosine", ascending=False)
           .head(n_show).drop(columns=["group", "nn is true match"]).reset_index(drop=True))
    return {"traps_stats": pd.DataFrame(stats), "traps_top15": top, "traps_all": all_rows}


def _category(name_sim: float, addr_sim: float, addr_missing: bool) -> str:
    if addr_missing:
        return "no address, same name" if name_sim >= 90 else "no address, name differs"
    if name_sim >= 90 and addr_sim >= 90:
        return "duplicate (same name + address)"
    if addr_sim >= 90:
        return "name variant at same address"
    if name_sim >= 90 and addr_sim < 60:
        return "same name, different address (branch?)"
    if name_sim < 60 and addr_sim < 60:
        return "different name and address"
    return "partial variant"


def multi_match_entities(d: EdaData, pairs: pa.Table, counts: np.ndarray, n_show: int = 20,
                         n_stats: int = 3000, seed: int = SEED):
    """S1 entities with 3+ matches: what are the extra records?"""
    rng = np.random.default_rng(seed)
    elig = np.nonzero(counts >= 3)[0]
    pick = rng.choice(elig, size=min(n_stats, len(elig)), replace=False)
    s1s = d.gt.column("source1_entity_id").take(pa.array(pick)).to_pylist()
    sub = pairs.filter(pc.is_in(pairs.column("s1"), value_set=pa.array(s1s, pa.string()))).to_pylist()
    rec = _records(d.train, s1s + [r["other"] for r in sub])
    groups = defaultdict(list)
    for r in sub:
        groups[r["s1"]].append(r["other"])

    rec_rows, ent_rows, examples, same_src_rows = [], [], [], []
    for i, s in enumerate(s1s):
        a = rec.get(s)
        if not a:
            continue
        others = sorted((rec[o] for o in groups[s] if o in rec), key=lambda r: r["entity_id"])
        if i < n_show:
            examples.append({"entity": i + 1, "role": "S1", "id": s, "name": a["business_name"],
                             "address": a["business_address"], "name sim": None,
                             "addr sim": None, "category": ""})
        cats = []
        for o in others:
            ns = _tsr_name(a["business_name"], o["business_name"])
            ad = _tsr_addr(a["business_address"], o["business_address"])
            cat = _category(ns, ad, not o["business_address"].strip())
            cats.append(cat)
            rec_rows.append({"category": cat})
            if i < n_show:
                examples.append({"entity": i + 1, "role": o["source"], "id": o["entity_id"],
                                 "name": o["business_name"], "address": o["business_address"],
                                 "name sim": ns, "addr sim": ad, "category": cat})
        addrs = {nz.normalize_address(o["business_address"]) for o in others
                 if o["business_address"].strip()}
        # Records from the SAME source under one entity: are they copies of each other?
        for src in ("S2", "S3"):
            same = [o for o in others if o["source"] == src]
            for x in range(len(same)):
                for y in range(x + 1, len(same)):
                    u, v = same[x], same[y]
                    has_addr = u["business_address"].strip() and v["business_address"].strip()
                    same_src_rows.append(bool(
                        has_addr
                        and _tsr_name(u["business_name"], v["business_name"]) >= 90
                        and _tsr_addr(u["business_address"], v["business_address"]) >= 90))
        n2 = sum(o["source"] == "S2" for o in others)
        ent_rows.append({"n_S2": n2, "n_S3": len(others) - n2, "distinct addresses": len(addrs),
                         "branch-like": any("branch" in c for c in cats)})

    rr, ent = pd.DataFrame(rec_rows), pd.DataFrame(ent_rows)
    breakdown = rr["category"].value_counts().rename_axis("category").reset_index(name="records")
    breakdown["share %"] = (100 * breakdown["records"] / breakdown["records"].sum()).round(1)
    mix = ent.groupby(["n_S2", "n_S3"]).size().reset_index(name="entities")
    mix = mix.sort_values("entities", ascending=False).head(10).reset_index(drop=True)
    mix["share %"] = (100 * mix["entities"] / len(ent)).round(1)
    distinct = ent["distinct addresses"].value_counts().sort_index().rename_axis(
        "distinct addresses among matches").reset_index(name="entities")
    facts = {
        "entities": len(ent),
        "same-source pairs": len(same_src_rows),
        "same-source near-duplicate %": (round(100 * np.mean(same_src_rows), 1)
                                          if same_src_rows else float("nan")),
        "branch-like %": round(100 * ent["branch-like"].mean(), 1),
        "duplicate share %": float(breakdown.loc[
            breakdown.category.str.startswith("duplicate"), "share %"].sum()),
    }
    return {"multi_breakdown": breakdown, "multi_source_mix": mix,
            "multi_distinct_addresses": distinct,
            "multi_examples": pd.DataFrame(examples)}, facts


def new_country_records(d: EdaData, new: list[str], n: int = 25, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for c in new:
        for src in ("s1", "s2", "s3"):
            t = d.test[src]
            sub = t.filter(pc.equal(t.column("country"), c))
            if sub.num_rows == 0:
                continue
            idx = np.sort(rng.choice(sub.num_rows, size=min(n, sub.num_rows), replace=False))
            df = sub.take(pa.array(idx)).to_pandas()
            df.insert(0, "source", src.upper())
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["source", "entity_id", "business_name", "business_address", "country"])


def _top_tokens(arr, k: int):
    flat = pc.list_flatten(pc.split_pattern_regex(pc.utf8_lower(arr), TOKEN_SPLIT))
    flat = flat.filter(pc.not_equal(flat, ""))
    if isinstance(flat, pa.ChunkedArray):
        flat = flat.combine_chunks()
    if len(flat) == 0:
        return []
    vc = pc.value_counts(flat)
    cnt = _np(vc.field("counts"))
    order = np.argsort(-cnt, kind="stable")[:k]
    vals = vc.field("values").take(pa.array(order)).to_pylist()
    total = cnt.sum()
    return [(v, int(cnt[o]), 100 * cnt[o] / total) for v, o in zip(vals, order)]


def top_tokens(d: EdaData, train_labels: list[str], new: list[str], k: int = 40,
               cap: int = 1_500_000, seed: int = SEED) -> dict:
    """Top name/address tokens per train country (S1+S2+S3) and per new test country."""
    rng = np.random.default_rng(seed)
    groups = [(f"train {c}", d.train, c) for c in train_labels] + \
             [(f"test {c}", d.test, c) for c in new]
    long = []
    for label, tables, c in groups:
        for fname, col in (("name", "business_name"), ("address", "business_address")):
            chunks = []
            for t in tables.values():
                chunks += t.column(col).filter(pc.equal(t.column("country"), c)).chunks
            arr = pa.chunked_array(chunks, type=pa.string())
            if len(arr) > cap:
                arr = arr.take(pa.array(np.sort(rng.choice(len(arr), size=cap, replace=False))))
            for rank, (tok, cnt, sh) in enumerate(_top_tokens(arr, k), 1):
                long.append({"group": label, "field": fname, "rank": rank, "token": tok,
                             "count": cnt, "share %": round(sh, 2)})
    long = pd.DataFrame(long)
    out = {"top_tokens_long": long}
    for fname in ("name", "address"):
        f = long[long.field == fname]
        wide = (f.assign(cell=f["token"] + " (" + f["count"].map("{:,}".format) + ")")
                .pivot(index="rank", columns="group", values="cell").reset_index())
        wide.columns.name = None
        out[f"top_{fname}_tokens"] = wide
    return out


# --- evidence for the assumptions --------------------------------------------------

def name_vs_address_signal(d: EdaData, pairs: pa.Table, n: int = 50_000, seed: int = SEED):
    """How much of the match signal is in the name vs the address?

    Positives are true pairs; negatives are random S2/S3 records from the S1's own country.
    Random negatives are easy, so the AUCs are upper bounds -- the agreement split among
    true pairs is the more telling number.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(pairs.num_rows, size=min(n, pairs.num_rows), replace=False))
    pos = pairs.take(pa.array(idx)).to_pylist()
    rec_s1 = _records({"s1": d.train["s1"]}, [p["s1"] for p in pos])

    cand = pa.concat_tables([d.train[s].select(["entity_id", "country"]) for s in ("s2", "s3")])
    codes, labels = _codes(cand.column("country"))
    pool = {lab: np.nonzero(codes == k)[0] for k, lab in enumerate(labels)}
    neg_pos = []
    for p in pos:
        a = rec_s1.get(p["s1"])
        pl = pool.get(a["country"]) if a else None
        neg_pos.append(int(rng.choice(pl)) if pl is not None and len(pl) else -1)
    valid = [i for i in neg_pos if i >= 0]
    neg_ids = iter(cand.column("entity_id").take(pa.array(valid, pa.int64())).to_pylist())
    del cand, codes, pool
    neg_list = list(neg_ids)
    rec_o = _records({"s2": d.train["s2"], "s3": d.train["s3"]},
                     [p["other"] for p in pos] + neg_list)

    rows, it = [], iter(neg_list)
    for p, ni in zip(pos, neg_pos):
        a = rec_s1.get(p["s1"])
        nid = next(it) if ni >= 0 else None
        if not a:
            continue
        for y, b in ((1, rec_o.get(p["other"])), (0, rec_o.get(nid) if nid else None)):
            if b:
                rows.append((y, _tsr_name(a["business_name"], b["business_name"]),
                             _tsr_addr(a["business_address"], b["business_address"]),
                             not b["business_address"].strip()))
    df = pd.DataFrame(rows, columns=["y", "name", "addr", "addr_missing"])
    auc_name = float(roc_auc_score(df.y, df.name))
    auc_addr = float(roc_auc_score(df.y, df.addr))
    p = df[df.y == 1]
    nm, ad = p.name >= 80, p.addr >= 80
    facts = {
        "auc_name": auc_name, "auc_addr": auc_addr,
        "both_agree": 100 * (nm & ad).mean(), "name_only": 100 * (nm & ~ad).mean(),
        "addr_only": 100 * (~nm & ad).mean(), "neither": 100 * (~nm & ~ad).mean(),
        "name_lt50": 100 * (p.name < 50).mean(), "addr_lt50": 100 * (p.addr < 50).mean(),
        "addr_missing": 100 * p.addr_missing.mean(), "n_pos": len(p), "n_neg": int((df.y == 0).sum()),
    }
    table = pd.DataFrame([
        ("true pairs / random same-country negatives", f"{facts['n_pos']:,} / {facts['n_neg']:,}"),
        ("AUC name tsr (normalized)", round(auc_name, 4)),
        ("AUC address tsr (normalized)", round(auc_addr, 4)),
        ("true pairs: name AND address agree (>=80) %", round(facts["both_agree"], 2)),
        ("true pairs: name only agrees %", round(facts["name_only"], 2)),
        ("true pairs: address only agrees %", round(facts["addr_only"], 2)),
        ("true pairs: neither agrees %", round(facts["neither"], 2)),
        ("true pairs: name tsr < 50 %", round(facts["name_lt50"], 2)),
        ("true pairs: address tsr < 50 %", round(facts["addr_lt50"], 2)),
        ("true pairs: match record has no address %", round(facts["addr_missing"], 2)),
    ], columns=["metric", "value"], dtype=object)
    return {"evidence_name_vs_address": table}, facts


postal_codes_in = nz.postal_codes_in
naive_codes_in = nz.naive_codes_in


def postal_agreement(d: EdaData, pairs: pa.Table, n: int = 200_000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(pairs.num_rows, size=min(n, pairs.num_rows), replace=False))
    sub = pairs.take(pa.array(idx)).to_pylist()
    rec = _records(d.train, [p["s1"] for p in sub] + [p["other"] for p in sub])

    # per key: pairs, both, agree, both6, agree6, both5, agree5, naive_both, naive_agree
    stat = defaultdict(lambda: np.zeros(9, dtype=np.int64))
    conflicts = []
    for p in sub:
        a, b = rec.get(p["s1"]), rec.get(p["other"])
        if not a or not b:
            continue
        ca, cb = postal_codes_in(a["business_address"]), postal_codes_in(b["business_address"])
        na, nb = naive_codes_in(a["business_address"]), naive_codes_in(b["business_address"])
        agree = bool(set(ca) & set(cb))
        for key in (a["country"], "ALL"):
            s_ = stat[key]
            s_[0] += 1
            if ca and cb:
                s_[1] += 1
                s_[2] += agree
            for kind, i in (("6-digit", 3), ("5-digit", 5)):
                ka = {c for c, k in ca.items() if k == kind}
                kb = {c for c, k in cb.items() if k == kind}
                if ka and kb:
                    s_[i] += 1
                    s_[i + 1] += bool(ka & kb)
            if na and nb:
                s_[7] += 1
                s_[8] += bool(na & nb)
        if ca and cb and not agree and len(conflicts) < 30:
            conflicts.append({"country": a["country"], "s1 codes": ",".join(sorted(ca)),
                              "s1 address": a["business_address"],
                              "match codes": ",".join(sorted(cb)),
                              "match address": b["business_address"]})

    def pct(x, y):
        return round(100 * x / y, 2) if y else None

    rows = []
    for key in sorted(stat, key=lambda k: (k == "ALL", k)):
        s_ = stat[key]
        rows.append({
            "country": key, "true pairs": int(s_[0]),
            "both have code %": pct(s_[1], s_[0]), "agree %": pct(s_[2], s_[1]),
            "both 6-digit": int(s_[3]), "6-digit agree %": pct(s_[4], s_[3]),
            "both 5-digit": int(s_[5]), "5-digit agree %": pct(s_[6], s_[5]),
            "naive: both %": pct(s_[7], s_[0]), "naive: agree %": pct(s_[8], s_[7]),
        })
    df = pd.DataFrame(rows)
    a = stat["ALL"]
    facts = {"both": pct(a[1], a[0]) or 0.0, "agree": pct(a[2], a[1]), "n_both": int(a[1]),
             "agree6": pct(a[4], a[3]), "n6": int(a[3]),
             "agree5": pct(a[6], a[5]), "n5": int(a[5]),
             "naive_agree": pct(a[8], a[7])}
    conflicts_df = pd.DataFrame(conflicts, columns=["country", "s1 codes", "s1 address",
                                                    "match codes", "match address"])
    return {"evidence_postal": df, "evidence_postal_conflicts": conflicts_df}, facts


_SUFFIX_RE = r"\b(" + "|".join(sorted(nz.LEGAL_SUFFIXES)) + r")\b"
_ABBREV_RE = r"\b(rd|st|ave|av|blvd|bd|r|nr|opp|bldg|flr|fl|pl|chem|imp|fbg|apt|sq|no)\b"
NOISE = {
    "name: legal suffix": ("business_name",
                           lambda a: pc.match_substring_regex(a, _SUFFIX_RE, ignore_case=True)),
    "name: junk prefix": ("business_name",
                          lambda a: pc.match_substring_regex(a, r"^[^\p{L}\p{N}]+")),
    "name: has digit": ("business_name", lambda a: pc.match_substring_regex(a, r"[0-9]")),
    "name: non-ASCII": ("business_name", lambda a: pc.invert(pc.string_is_ascii(a))),
    "name: ALL CAPS": ("business_name", pc.utf8_is_upper),
    "addr: empty": ("business_address", lambda a: pc.equal(pc.utf8_trim_whitespace(a), "")),
    "addr: ALL CAPS": ("business_address", pc.utf8_is_upper),
    "addr: abbreviation": ("business_address",
                           lambda a: pc.match_substring_regex(a, _ABBREV_RE, ignore_case=True)),
    "addr: landmark": ("business_address",
                       lambda a: pc.match_substring_regex(a, LANDMARK, ignore_case=True)),
    "addr: zero-padded number": ("business_address",
                                 lambda a: pc.match_substring_regex(a, r"(^|[^0-9])0[0-9]{3,}")),
    "addr: no digit": ("business_address",
                       lambda a: pc.invert(pc.match_substring_regex(a, r"[0-9]"))),
}


def noise_signature(d: EdaData) -> pd.DataFrame:
    rows = []
    for split, src in SPLIT_SOURCES:
        t = d.split(split)[src]
        codes, labels = _codes(t.column("country"))
        flags = {name: _np(fn(t.column(col))) for name, (col, fn) in NOISE.items()}
        for k, lab in enumerate(labels):
            m = codes == k
            row = {"split": split, "source": src.upper(), "country": lab, "rows": int(m.sum())}
            for name, f in flags.items():
                row[f"{name} %"] = round(100 * f[m].mean(), 2)
            rows.append(row)
    return pd.DataFrame(rows)


def noise_direction(sig: pd.DataFrame, ref_countries: list[str], new: list[str],
                    min_gap: float = 1.0):
    """Does each new country's S2/S3-vs-S1 noise gap point the same way as the known ones?

    Uses the test split for every country so the comparison is like-for-like. A check
    only counts where all reference countries agree on direction by at least min_gap pp.
    """
    t = sig[sig.split == "test"].set_index(["source", "country"])
    metrics = [c for c in sig.columns if c.endswith(" %")]
    checks = []
    for m in metrics:
        for src in ("S2", "S3"):
            try:
                ref = {c: t.loc[(src, c), m] - t.loc[("S1", c), m] for c in ref_countries}
            except KeyError:
                continue
            if not ref or any(abs(v) < min_gap for v in ref.values()):
                continue
            signs = {np.sign(v) for v in ref.values()}
            if len(signs) != 1:
                continue
            want = signs.pop()
            for c in new:
                try:
                    gap = t.loc[(src, c), m] - t.loc[("S1", c), m]
                except KeyError:
                    continue
                row = {"metric": m.removesuffix(" %"), "source": src}
                row.update({f"{r} gap (pp)": round(v, 2) for r, v in ref.items()})
                row.update({"new country": c, "new gap (pp)": round(gap, 2),
                            "same direction": bool(np.sign(gap) == want)})
                checks.append(row)
    df = pd.DataFrame(checks)
    share = float(df["same direction"].mean()) if len(df) else float("nan")
    return df, share, len(df)


def _verdict_b(f: dict) -> str:
    if f["auc_name"] > f["auc_addr"] + 0.01 and f["name_only"] > f["addr_only"]:
        return "holds"
    if f["auc_addr"] > f["auc_name"] + 0.01 and f["addr_only"] > f["name_only"]:
        return "fails"
    return "partly"


def _verdict_share(x: float, hi: float, lo: float) -> str:
    if x != x:  # NaN
        return "untested"
    return "holds" if x >= hi else "partly" if x >= lo else "fails"


def assumptions(n_dup: int, n_ids: int, nv: dict, pv: dict, dir_share: float,
                dir_n: int, new: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"assumption": "(a) each S2/S3 has at most one owner",
         "verdict": "holds" if n_dup == 0 else "fails",
         "evidence": f"{n_dup:,} of {n_ids:,} matched IDs appear under more than one S1"},
        {"assumption": "(b) names carry most of the signal",
         "verdict": _verdict_b(nv),
         "evidence": (f"AUC name {nv['auc_name']:.3f} vs address {nv['auc_addr']:.3f}; "
                      f"true pairs agreeing on name only {nv['name_only']:.1f}% vs address only "
                      f"{nv['addr_only']:.1f}%; name tsr<50 in {nv['name_lt50']:.1f}%")},
        {"assumption": "(c) postal codes are reliable when present",
         "verdict": ("untested (too rare)" if pv["n_both"] < 30
                     else _verdict_share(pv["agree"] or 0.0, 95, 80)),
         "evidence": (f"a genuine code is on both records in only {pv['both']:.2f}% of true pairs "
                      f"(n={pv['n_both']:,}); they agree in {pv['agree']}% "
                      f"(6-digit {pv['agree6']}% n={pv['n6']:,}; 5-digit {pv['agree5']}% "
                      f"n={pv['n5']:,}). Naive digit-run matching agrees only "
                      f"{pv['naive_agree']}% -- most 5/6-digit runs are house numbers.")},
        {"assumption": "(d) France shows the same noise patterns in a new language",
         "verdict": _verdict_share(100 * dir_share, 80, 50) if dir_n else "untested",
         "evidence": (f"{round(dir_share * dir_n)}/{dir_n} S2/S3-vs-S1 noise gaps in "
                      f"{', '.join(new) or 'new countries'} point the same way as in the train "
                      "countries (test split). Descriptive only: France has no labels.")},
    ])


# --- orchestration ------------------------------------------------------------------

def write_tables(tables: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        df.to_csv(out_dir / f"{key}.csv", index=False)


def load_tables(report_dir) -> dict:
    """What the notebook calls: every CSV the step wrote, keyed by name."""
    out = {}
    for f in sorted(Path(report_dir).glob("*.csv")):
        try:
            out[f.stem] = pd.read_csv(f, keep_default_na=False)
        except pd.errors.EmptyDataError:  # a section that legitimately found nothing
            out[f.stem] = pd.DataFrame()
    return out


def show(tables: dict, *keys: str, max_rows: int | None = 100) -> None:
    """Notebook display helper: one DataFrame per key, or a note if the step has not run."""
    from IPython.display import Markdown, display

    for k in keys:
        df = tables.get(k)
        if df is None:
            display(Markdown(f"_`{k}` not found -- run step `p1c_eda` first._"))
            continue
        display(df if max_rows is None else df.head(max_rows))


def _kv(df: pd.DataFrame) -> list[str]:
    return [md_table(df), ""]


def run_deep(paths, args) -> list[str]:
    seed = args.seed
    t0 = time.perf_counter()
    _log("p1c: loading data")
    d = load_eda_data(paths, args.sample, seed)
    pairs = gt_pairs_arrow(d.gt)
    T: dict = {}

    _log("p1c: 1 ground truth")
    t, counts = section_ground_truth(d, pairs)
    T.update(t)
    _log("p1c: 2 ownership")
    t, n_dup = section_ownership(pairs)
    T.update(t)
    _log("p1c: 3 sizes")
    T["sizes"] = section_sizes(paths)
    _log("p1c: 4 country")
    t, train_labels, new, differ = section_country(d, pairs)
    T.update(t)
    _log("p1c: 5 fields")
    T.update(section_fields(d))
    _log("p1c: 6 formats")
    T["formats"] = section_formats(d)

    _log("p1c: samples - 30 true pairs")
    T["true_pairs_30"] = sample_true_pairs(d, pairs, 30, seed)
    _log("p1c: samples - singleton traps (TF-IDF nearest neighbour)")
    T.update(singleton_traps(d, pairs, counts, seed=seed))
    _log("p1c: samples - 3+ match entities")
    t, mfacts = multi_match_entities(d, pairs, counts, seed=seed)
    T.update(t)
    _log("p1c: samples - new-country test records")
    T["new_country_records"] = new_country_records(d, new, 25, seed)
    _log("p1c: top tokens")
    T.update(top_tokens(d, train_labels, new, seed=seed))

    _log("p1c: evidence - name vs address")
    t, nv = name_vs_address_signal(d, pairs, seed=seed)
    T.update(t)
    _log("p1c: evidence - postal agreement")
    t, pv = postal_agreement(d, pairs, seed=seed)
    T.update(t)
    _log("p1c: evidence - noise signature")
    T["noise_signature"] = noise_signature(d)
    T["noise_direction"], dir_share, dir_n = noise_direction(T["noise_signature"], train_labels, new)
    T["assumptions"] = assumptions(n_dup, pairs.num_rows, nv, pv, dir_share, dir_n, new)

    gs = dict(zip(T["gt_summary"]["metric"], T["gt_summary"]["value"]))
    sz = T["sizes"].set_index("file")
    fe = T["field_empty"].set_index(["split", "source"])["business_address empty %"]
    ts = T["traps_stats"].set_index("group")
    new_share = T["country_new_in_test"]
    new_s1 = new_share[new_share.source == "S1"]
    T["summary"] = pd.DataFrame([
        ("scope", d.scope),
        ("train S1 / test S1 entities", f"{sz.loc['source1', 'train rows']:,} / "
                                         f"{int(sz.loc['source1', 'test rows']):,}"),
        ("test/train ratio S1 / S2 / S3", " / ".join(
            f"{sz.loc[s, 'test/train']:.2f}" for s in ("source1", "source2", "source3"))),
        ("singletons (train)", f"{gs['singletons %']}%"),
        ("matches per non-singleton: mean / max",
         f"{gs['mean matches per non-singleton']} / {gs['max matches']}"),
        ("matched IDs from S2 / S3",
         f"{gs['matched IDs from S2 %']}% / {gs['matched IDs from S3 %']}%"),
        ("IDs under more than one S1", f"{n_dup:,}"),
        ("true pairs whose country differs", f"{differ:,}"),
        ("countries new in test", ", ".join(
            f"{r.country} ({r['share %']}% of test S1)" for _, r in new_s1.iterrows()) or "none"),
        ("empty address: train S2 / S3 / test S2 / S3", " / ".join(
            f"{fe.loc[(sp, s)]:.1f}%" for sp, s in
            (("train", "S2"), ("train", "S3"), ("test", "S2"), ("test", "S3")))),
        ("true pairs where both carry a postal code / codes agree",
         f"{pv['both']:.2f}% / {pv['agree']}% (naive digit runs: {pv['naive_agree']}%)"),
        ("AUC name vs address (random negatives)", f"{nv['auc_name']:.3f} vs {nv['auc_addr']:.3f}"),
        ("true pairs agreeing on name only / address only",
         f"{nv['name_only']:.1f}% / {nv['addr_only']:.1f}%"),
        ("singleton traps: NN cosine >= 0.8",
         f"{ts.loc['singleton', 'cosine >= 0.8 %']}% of singletons"
         if "singleton" in ts.index else "n/a"),
        ("matched S1: TF-IDF NN is a true match",
         f"{ts.loc['matched', 'nn is true match %']}%" if "matched" in ts.index else "n/a"),
        ("3+ match entities: extras that duplicate the S1 (name+address)",
         f"{mfacts['duplicate share %']:.1f}% of records; branch-like in "
         f"{mfacts['branch-like %']}% of entities"),
        ("same-source records within an entity that are near-copies",
         f"{mfacts['same-source near-duplicate %']}% of {mfacts['same-source pairs']:,} pairs"),
        ("new-country noise gaps matching train direction",
         f"{round(dir_share * dir_n)}/{dir_n}" if dir_n else "n/a"),
    ], columns=["fact", "value"])

    report_dir = getattr(args, "report_dir_resolved", None)
    if report_dir is not None and not args.no_report:
        write_tables(T, Path(report_dir) / "p1c_eda")
        _log(f"p1c: wrote {len(T)} tables to {Path(report_dir) / 'p1c_eda'}")
    _log(f"p1c: done in {time.perf_counter() - t0:,.0f}s")

    L = [f"**scope:** {d.scope}  ", "Full tables: `reports/p1c_eda/*.csv`, "
         "displayed by `notebooks/01_eda.ipynb`.", ""]
    L += ["#### One-screen summary", ""] + _kv(T["summary"])
    L += ["#### Assumptions", ""] + _kv(T["assumptions"])
    L += ["#### 1. Ground truth", ""] + _kv(T["gt_match_hist"]) + _kv(T["gt_summary"].iloc[4:])
    L += ["#### 2. Ownership", ""] + _kv(T["ownership"])
    if len(T["ownership_examples"]):
        L += _kv(T["ownership_examples"].head(5))
    L += ["#### 3. Sizes", ""] + _kv(T["sizes"])
    L += ["#### 4. Country", ""] + _kv(T["country_counts"]) + _kv(T["country_variants"])
    L += ["#### 5. Fields", ""] + _kv(T["field_empty"].drop(columns=["entity_id empty %"]))
    L += _kv(T["field_lengths"])
    L += ["#### 6. Formats (from text, per country)", ""] + _kv(T["formats"])
    L += ["#### Singleton traps / 3+ match entities", ""] + _kv(T["traps_stats"])
    L += _kv(T["multi_breakdown"])
    L += ["#### Evidence", ""] + _kv(T["evidence_name_vs_address"]) + _kv(T["evidence_postal"])
    if len(T["evidence_postal_conflicts"]):
        L += ["Postal conflicts that survive the refined extractor (first 5):", ""]
        L += _kv(T["evidence_postal_conflicts"].head(5))
    return L
