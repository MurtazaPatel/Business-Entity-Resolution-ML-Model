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


def country_agreement(pairs: pd.DataFrame, lookup: dict[str, tuple[str, str, str]]) -> list[str]:
    """Do true matches share a country? Decides if same_country may gate blocking."""
    same = cross = unknown = 0
    cross_examples: list[str] = []
    for s1, other in zip(pairs["s1"], pairs["other"]):
        a, b = lookup.get(s1), lookup.get(other)
        if a is None or b is None:
            unknown += 1
            continue
        if a[2] == b[2]:
            same += 1
        else:
            cross += 1
            if len(cross_examples) < 5:
                cross_examples.append(f"{s1}({a[2]}) ~ {other}({b[2]})")
    total = same + cross
    lines = ["#### 5. Do true matches cross country?", ""]
    lines += [
        f"- same country: **{same:,}** ({_pct(same, total)})",
        f"- different country: **{cross:,}** ({_pct(cross, total)})",
        f"- unresolved IDs: {unknown:,}",
    ]
    if cross:
        lines.append(f"- examples: {'; '.join(cross_examples)}")
    verdict = (
        "safe to block within country (costs ~0 recall)"
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
    pairs: pd.DataFrame, lookup: dict[str, tuple[str, str, str]], probe: int, seed: int
) -> list[str]:
    """Per-key recall on true pairs: the recall ceiling each blocking key can deliver."""
    rng = np.random.default_rng(seed)
    if len(pairs) > probe:
        idx = rng.choice(len(pairs), size=probe, replace=False)
        sub = pairs.iloc[np.sort(idx)]
    else:
        sub = pairs

    resolved = [
        (lookup[s1], lookup[o])
        for s1, o in zip(sub["s1"], sub["other"])
        if s1 in lookup and o in lookup
    ]
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
    hits = {}
    for key, fn in BLOCKING_KEYS.items():
        h = sum(1 for a, b in resolved if fn(a, b))
        hits[key] = h
        lines.append(f"| {key} | **{_pct(h, n)}** |")

    # Candidate blocking schemes: each is the union of keys, i.e. a pair survives if ANY
    # key fires. The cheapest scheme clearing ~99% is the one to build.
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
        if label == "name trigram OR addr token":
            best_fns = fns
    lines.append("")

    fns = best_fns
    miss = [(a, b) for a, b in resolved if not any(f(a, b) for f in fns)]
    if miss:
        lines += [
            f"Pairs that survive NO key in 'name trigram OR addr token' "
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

    # id -> (name, address, country) for every record we might touch.
    lookup: dict[str, tuple[str, str, str]] = {}
    for frame in (s1, s2, s3):
        lookup.update(
            zip(
                frame["entity_id"].astype(str),
                zip(
                    frame["business_name"].astype(str),
                    frame["business_address"].astype(str),
                    frame["country"].astype(str),
                ),
            )
        )

    lines = [f"**scope:** {scope}", ""]
    lines += check_one_s1_per_record(pairs)
    lines += match_count_stats(gt, pairs)
    lines += coverage_stats(pairs, s2, s3)
    lines += field_quality({"train_s1": s1, "train_s2": s2, "train_s3": s3})
    lines += country_agreement(pairs, lookup)
    lines += blocking_recall(pairs, lookup, probe, seed)
    lines += token_frequency(s1, probe, seed)
    return lines
