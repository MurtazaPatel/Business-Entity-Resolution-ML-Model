"""p2b step: tune the name-only baseline on train, then predict the test set.

CV is grouped by S1 via artifacts/folds.parquet, LOCO trains on one country and evaluates
on another. Both only need a threshold, so "training" is a grid search over cosine.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import baseline as B
import splits as S
from io_utils import gzip_file, parse_id_list, read_source, write_submission
from metrics import macro_f05
from report import md_table


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _truth_map(gt: pd.DataFrame) -> dict[str, list[str]]:
    return {k: parse_id_list(v)
            for k, v in zip(gt["source1_entity_id"].astype(str), gt["matched_entity_ids"])}


def _pool(paths, split: str) -> pd.DataFrame:
    a, b = ((paths.train_s2, paths.train_s3) if split == "train"
            else (paths.test_s2, paths.test_s3))
    cols = ["entity_id", "business_name", "country"]
    return pd.concat([read_source(a)[cols], read_source(b)[cols]], ignore_index=True)


def run(paths, args) -> list[str]:
    seed = args.seed
    k = getattr(args, "top_k", B.TOP_K)
    max_df = getattr(args, "max_df", B.MAX_DF)
    cv_sample = getattr(args, "cv_sample", 200_000)
    jobs = getattr(args, "jobs", 1) or 1
    chunk = getattr(args, "query_chunk", B.QUERY_CHUNK) or B.QUERY_CHUNK
    t_start = time.perf_counter()

    cols = ["entity_id", "business_name", "country"]
    s1 = read_source(paths.train_s1)[cols]
    gt = read_source(paths.train_gt)
    if args.sample:
        s1 = s1.head(args.sample)
        keep = set(s1["entity_id"].astype(str))
        gt = gt[gt["source1_entity_id"].astype(str).isin(keep)]
    truth = _truth_map(gt)

    folds = S.make_folds(paths, seed=seed, force=args.force)
    folds = folds[folds["source1_entity_id"].isin(set(gt["source1_entity_id"].astype(str)))]

    # Tuning runs on a stratified subsample: the threshold is one scalar, so the extra
    # precision from all 2.2M entities is not worth the hours it would cost.
    if cv_sample and len(folds) > cv_sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(folds), size=cv_sample, replace=False))
        folds = folds.iloc[idx]
    eval_ids = set(folds["source1_entity_id"])
    s1_eval = s1[s1["entity_id"].astype(str).isin(eval_ids)]
    truth_eval = {k_: v for k_, v in truth.items() if k_ in eval_ids}
    _log(f"tuning pool: {len(s1_eval):,} S1 entities")

    _log("building train candidates")
    pool = B.split_pool_by_country(_pool(paths, "train"))   # frame freed before searching
    _log(f"pool by country: {', '.join(f'{c} {len(v[0]):,}' for c, v in sorted(pool.items()))}")
    cands = B.generate_candidates(s1_eval, pool, k=k, max_df=max_df, jobs=jobs, chunk=chunk, log=_log)
    del pool
    _log(f"candidates: {len(cands.frame):,} pairs")

    block = B.blocking_recall(cands, truth_eval)
    tuning = B.tune_threshold(cands, truth_eval)
    thr = B.best_threshold(tuning)
    _log(f"best threshold {thr}")

    # --- CV: tune on the training folds, score the held-out fold --------------
    cv_rows = []
    for fold in sorted(folds["fold"].unique()):
        tr_mask, va_mask = S.fold_masks(folds, fold)
        tr_ids = set(folds.loc[tr_mask, "source1_entity_id"])
        va_ids = set(folds.loc[va_mask, "source1_entity_id"])
        tr_t = {i: truth_eval[i] for i in tr_ids if i in truth_eval}
        va_t = {i: truth_eval[i] for i in va_ids if i in truth_eval}
        tr_c = B.Candidates(cands.frame[cands.frame["s1"].isin(tr_ids)], len(tr_t), 0)
        t_fold = B.best_threshold(B.tune_threshold(tr_c, tr_t))
        va_pred = B.Candidates(cands.frame[cands.frame["s1"].isin(va_ids)],
                               len(va_t), 0).above(t_fold)
        sc = macro_f05(va_t, va_pred)
        cv_rows.append({"fold": int(fold), "threshold": t_fold, "eval S1": sc.n,
                        "macro F0.5": round(sc.macro_f, 5),
                        "singletons": round(sc.singleton_f, 5),
                        "non-singletons": round(sc.non_singleton_f, 5),
                        "mean precision": round(sc.mean_precision, 4),
                        "mean recall": round(sc.mean_recall, 4)})
    cv = pd.DataFrame(cv_rows)
    mean_row = {"fold": "mean", "threshold": round(cv["threshold"].mean(), 4),
                "eval S1": int(cv["eval S1"].sum())}
    for c in ("macro F0.5", "singletons", "non-singletons", "mean precision", "mean recall"):
        mean_row[c] = round(cv[c].mean(), 5)
    cv = pd.concat([cv, pd.DataFrame([mean_row])], ignore_index=True)
    _log(f"CV macro F0.5 {mean_row['macro F0.5']}")

    # --- LOCO ----------------------------------------------------------------
    loco_rows = []
    for spl in S.loco_splits(folds):
        tr_m, ev_m = S.loco_masks(folds, spl)
        tr_ids = set(folds.loc[tr_m, "source1_entity_id"])
        ev_ids = set(folds.loc[ev_m, "source1_entity_id"])
        tr_t = {i: truth_eval[i] for i in tr_ids if i in truth_eval}
        ev_t = {i: truth_eval[i] for i in ev_ids if i in truth_eval}
        if not tr_t or not ev_t:
            continue
        t_loco = B.best_threshold(B.tune_threshold(
            B.Candidates(cands.frame[cands.frame["s1"].isin(tr_ids)], len(tr_t), 0), tr_t))
        pred = B.Candidates(cands.frame[cands.frame["s1"].isin(ev_ids)],
                            len(ev_t), 0).above(t_loco)
        sc = macro_f05(ev_t, pred)
        loco_rows.append({"split": spl["name"], "threshold": t_loco, "eval S1": sc.n,
                          "macro F0.5": round(sc.macro_f, 5),
                          "singletons": round(sc.singleton_f, 5),
                          "non-singletons": round(sc.non_singleton_f, 5),
                          "mean precision": round(sc.mean_precision, 4),
                          "mean recall": round(sc.mean_recall, 4)})
    loco = pd.DataFrame(loco_rows)
    country_stats = (pd.DataFrame(cands.stats).T.reset_index(names="country")
                     if cands.stats else pd.DataFrame())
    del cands

    lines = [
        f"**config:** char {B.NGRAM[0]}-gram TF-IDF on core_name, top-{k} by cosine, "
        f"max_df={max_df}, one global threshold. Tuned on {len(s1_eval):,} S1 entities.",
        "",
        "#### Blocking (before any threshold)", "",
        md_table(pd.DataFrame([block]).T.reset_index().set_axis(["metric", "value"], axis=1)), "",
        "#### Per-country blocking", "",
        md_table(country_stats) if len(country_stats) else "_none_", "",
        "#### Threshold sweep", "", md_table(tuning[tuning["threshold"] % 0.1 < 1e-9]), "",
        f"Best threshold: **{thr}**", "",
        "#### CV (grouped by S1)", "", md_table(cv), "",
        "#### LOCO", "", md_table(loco) if len(loco) else "_no LOCO splits available_", "",
    ]

    # --- test inference ------------------------------------------------------
    if not getattr(args, "skip_test", False):
        _log("test inference")
        test_s1 = read_source(paths.test_s1)[cols]
        test_pool = B.split_pool_by_country(_pool(paths, "test"))
        _log(f"test pool: {', '.join(f'{c} {len(v[0]):,}' for c, v in sorted(test_pool.items()))}")
        tc = B.generate_candidates(test_s1, test_pool, k=k, max_df=max_df, jobs=jobs, chunk=chunk, log=_log)
        del test_pool

        all_ids = test_s1["entity_id"].astype(str).tolist()
        cand_map = tc.id_lists()
        pred_map = tc.above(thr)
        cand_df = pd.DataFrame({"source1_entity_id": all_ids,
                                "candidate_entity_ids": [cand_map.get(i, []) for i in all_ids]})
        pred_df = pd.DataFrame({"source1_entity_id": all_ids,
                                "matched_entity_ids": [pred_map.get(i, []) for i in all_ids]})
        write_submission(cand_df, paths.candidate_pairs, "candidate")
        write_submission(pred_df, paths.matching_results, "matched")
        # A gzipped copy of the scored file travels back through git; output/*.tsv itself
        # is gitignored, which is how the first full run's predictions got stranded on
        # Kaggle. candidate_pairs is not compressed here -- it is only needed for the
        # final zip, and committing both would add ~96 MB per run.
        gz = gzip_file(paths.matching_results)
        n_pred = sum(len(v) for v in pred_map.values())
        lines += [
            "#### Test submission", "",
            md_table(pd.DataFrame([
                ("test S1 rows", len(all_ids)),
                ("candidate pairs", len(tc.frame)),
                ("predicted pairs (>= threshold)", n_pred),
                ("S1 with a prediction", len(pred_map)),
                ("S1 predicted empty", len(all_ids) - len(pred_map)),
                ("S1 predicted empty %", round(100 * (len(all_ids) - len(pred_map)) / len(all_ids), 2)),
                ("matching_results.tsv MB", round(paths.matching_results.stat().st_size / 1024**2, 1)),
                ("matching_results.tsv.gz MB", round(gz.stat().st_size / 1024**2, 1)),
            ], columns=["metric", "value"], dtype=object)), "",
            f"Submittable file is committed as `output/{gz.name}`; expand with "
            "`python -c \"import sys; sys.path.insert(0,'code/business_entity_resolution/src'); "
            "import io_utils; io_utils.gunzip_file('output/matching_results.tsv.gz')\"`.", "",
        ]
        _log(f"wrote {paths.matching_results} and {paths.candidate_pairs}")

    _log(f"p2b done in {time.perf_counter() - t_start:,.0f}s")
    return lines
