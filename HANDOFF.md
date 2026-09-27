# Handoff — Business Entity Resolution (Amazon ML Challenge)

For the engineer (and their Claude Code) picking this up at **p3**. Read `CLAUDE.md` first for
the hard rules; this file is **state**: what is built, what is measured, and what the numbers
imply for p3. Every figure here comes from a full-data run recorded in `reports/`.

---

## 1. Where things stand

| | |
|---|---|
| Repo root | `student_resource/` — run every command from here |
| Pipeline | `code/business_entity_resolution/src/` (10 modules, ~3.2k lines) |
| Tests | **142 passing** — `.venv/bin/python -m pytest code/business_entity_resolution/tests/ -q` |
| Steps done | p1a_check, p1b_eda, p1c_eda, p2a_splits, p2b_baseline |
| Best CV | **macro F0.5 0.3269** (name-only TF-IDF baseline) |
| Leaderboard | **not yet submitted** — see §6 |

Everything runs through one entry point:

```bash
python code/business_entity_resolution/src/run_step.py --step <id> [flags]
```

Step ids: `p1a_check`, `p1b_eda`, `p1c_eda`, `p2a_splits`, `p2b_baseline`.
A typo suggests the nearest id. Full-data runs happen on Kaggle via `notebooks/runner.ipynb`.

---

## 2. The five findings that should drive p3

All from `reports/p1c_eda.md` on full train + test.

**1. A name alone cannot identify a business at this scale — this is the headline.**
Searching all 10.3M S2/S3 names, the median nearest-neighbour cosine is **1.0 for singletons
as well as matched entities**. 92.7% of singletons have a name twin at cosine ≥ 0.8, and for a
matched S1 the name-nearest-neighbour is the true match only **48.3%** of the time. On a 10k
sample the same measurement said 8.6% and 92% — *the opposite conclusion*. **Do not trust any
similarity measurement taken on a subsample.**

**2. The address decides more matches than the name.** Among true pairs, 16.1% agree on
address while disagreeing on name, versus 10.1% the other way. The name's higher AUC
(0.976 vs 0.956) is against *random* negatives, which are trivially easy, and does not survive
the hard negatives finding 1 shows are everywhere.

**3. Exclusivity holds exactly and is the strongest unused lever.** Not one of the
**7,638,365** matched IDs appears under two S1 entities. It also bites: 37% of matched
entities' and 47.7% of singletons' nearest neighbours are already owned by *another* S1. A
global assignment can reject candidates a per-pair classifier would accept. **p2b does not use
this at all.**

**4. Postal codes are accurate and nearly absent.** A genuine code sits on both records in
only **0.16%** of true pairs, though it agrees 98.75% when present. Matching raw 5/6-digit
runs looks 36× more available but agrees just 93.5% — most such runs are house numbers. Good
feature when present, useless as a blocking key.

**5. Most matches are near-duplicates, not branches.** 52.3% of matched records repeat the
S1's name *and* address, 18.2% are a name variant at the same address, and only 0.5% are a
same-name different-address branch. Within one entity, 47% of same-source record pairs are
near-copies of each other.

**France (15% of test, absent from train) behaves like the training countries** — all 17
S2-vs-S1 and S3-vs-S1 noise gaps that US and India agree on point the same way for France.

---

## 3. Data shape

| file | rows | note |
|---|---:|---|
| train_source1 / ground_truth | 2,206,821 | one GT row per S1 |
| train_source2 | 5,034,616 | 26.64% match no S1 (distractors) |
| train_source3 | 5,285,603 | 25.37% distractors |
| test_source1 | 1,732,544 | every row needs a submission row |
| test_source2 | 4,887,273 | |
| test_source3 | 5,082,316 | |

- **7,638,365** true pairs; mean **3.67** per non-singleton, median 4, **max 11**.
- **Singletons: 5.58%** (123,247). So "predict empty" scores exactly **0.05585**, not more.
- Countries: train = US 59.98% / India 40.02%. Test = India 46.8% / US 38.3% / **France 14.98%**.
- **100% of true pairs are same-country** (0 of 7,638,365 cross). Partitioning by country costs
  no recall and is a `same_country` constraint, not a hard-coded country list.
- Empty addresses: ~3.4% of S2/S3 train, ~2.7% test. Non-Latin (Devanagari) names: 11–15% of S2/S3.

---

## 4. The metric, measured on real ground truth

`src/metrics.py` implements macro F0.5 exactly as `PROBLEM.md` states it. Verified on all
2.2M train entities:

| baseline | macro F0.5 |
|---|---|
| perfect | 1.00000 |
| **always predict empty** | **0.05585** (= the singleton rate) |
| drop one true match per entity | 0.87105 |
| **add one false ID per entity** | **0.75154** |

**A false positive costs ~2× a miss** (−0.25 vs −0.13). Adding one false ID also wipes the
singleton bucket entirely (1.0 → 0.0), which alone is 5.58% of the score. Every threshold
decision should be made against these numbers.

`Score` also reports the macro split by singleton / non-singleton — use it, because changes
that help one usually hurt the other.

---

## 5. Validation

`artifacts/folds.parquet` (built by `p2a_splits`, 2,206,821 rows):

- 5 folds of 441,364–441,365 entities, `StratifiedGroupKFold` grouped on S1 id, stratified on
  (country, singleton flag). Singleton rate 5.58% in every fold, **spread 0.0002pp**.
- Grouping on S1 is what stops one entity's pairs spanning folds. `splits.check_no_leakage()`
  asserts this; re-run it after any change.
- LOCO: `US→India` and `India→US`, derived from the labels present in the data so France needs
  no code change.

**Pick models by CV + LOCO, not the leaderboard** (it scores only a subset of test).

---

## 6. p2b baseline — what it is and its ceiling

Char 3-gram TF-IDF on `core_name`, top-5 by cosine per S1 within country, one global threshold.
**There is no learned model** — the only thing fit from labels is one float, `0.975`.

| | |
|---|---|
| CV macro F0.5 | **0.3269** (folds 0.3255–0.3288) |
| singletons / non-singletons | 0.1358 / 0.3383 |
| LOCO India→US / US→India | 0.3674 / **0.2660** |
| blocking pair-recall | 35.4%; only 57.8% of entities get ≥1 true candidate |
| test: candidates / predicted | 8,413,358 / 7,283,409 |
| test: predicted empty | 67,614 (**3.9%**, vs ~5.6% true singletons) |
| runtime | 5.8 h, 9.34 GB peak (4.5 h with `--threshold 0.975`) |

**Its ceiling is measured, not guessed.** On the real 6.2M-row US pool, *exact* name-only top-5
— no pruning, ~42 h of compute — recovers only **74.3%** of true matches. `max_df=0.01` gives
65.0%. So ~26% of true pairs are unreachable by name similarity at any compute budget.

The 3.9% vs 5.6% gap is pure false merges on singletons, which is why the singleton bucket
scores 0.136 where predict-nothing scores 1.0.

**No submission has been scored yet.** The run wrote a valid file twice, but the Kaggle session
was recycled before it could be retrieved (see §8). The fix is in place: the step now gzips
`matching_results.tsv` and the runner commits `output/matching_results.tsv.gz`, so `git pull`
yields a submittable file. Expand with `io_utils.gunzip_file(...)` or `gunzip -k`.

---

## 7. What p3 should do

The measurements point one way: **the name is exhausted, the address and exclusivity are not.**

1. **Blocking that actually reaches the matches.** `p1b_eda` measured union recall on 300k true
   pairs: name-token OR address-token = **99.99%**, versus 35.4% for the current top-5. Build an
   inverted index over rare name tokens *and* address tokens. This is the single biggest win —
   the classifier cannot fix candidates it never sees.
2. **Pair features + LightGBM.** rapidfuzz ratios on `core_name` and normalized address,
   token/trigram Jaccard, shared postal code (present in only 0.16% of pairs, so keep an
   is-present flag), shared street number, `same_country`, empty-address flags, and
   *competition* features (this candidate's score relative to the S1's best other candidate).
3. **Exploit exclusivity.** Each S2/S3 record belongs to at most one S1 — verified on all
   7,638,365 pairs. Add a global assignment or mutual-best step after scoring; 37–48% of
   nearest neighbours are already owned by another S1.
4. **Tune the decision for F0.5, per entity.** A false positive costs 2× a miss, and singletons
   are 5.58% of the average. Consider a per-entity "predict empty" decision rather than one
   global cosine threshold.
5. **Check LOCO every time.** US→India already drops to 0.266 against 0.367 the other way, so
   thresholds transfer badly between countries — and France was never validated at all.

---

## 8. Gotchas that cost us time

- **Never pin `numpy` below the Kaggle/Colab image's version.** Pinning `numpy==1.26.4`
  downgraded it under pandas/sklearn binaries compiled against numpy 2.x and every import died
  with `ValueError: numpy.dtype size changed`. `requirements.txt` uses floors for anything
  compiled; only pure-Python packages are pinned. The runner installs missing modules with
  `--no-deps` and never touches the preinstalled stack.
- **Kaggle flattens dataset uploads.** The runner locates the 7 TSVs by *filename* anywhere
  under `/kaggle/input` and rebuilds `train/`/`test/` as symlinks, so either layout works.
- **`output/*.tsv` is gitignored; only `output/matching_results.tsv.gz` is committed.** A
  46 MB gz per run accumulates in history forever — `PUSH_SUBMISSION = False` turns it off.
  gzip is the floor here: parquet with int32 ids was *larger* (47 MB), because the ids are
  high-entropy integers.
- **Kaggle "Save Version" was failing with `ConcurrencyViolation`.** It is not needed — the
  runner pushes reports and the submission during the run. `notebooks/recover_submission.py`
  rescues a finished run from a live session (pushes to a side branch, never `main`).
- **Peak RAM tracks `--query-chunk`, not the index.** ~1.1 MB per query per worker, so
  chunk 500 × 4 workers ≈ 2 GB. It was 2000 (8.5 GB) and would have fallen over.
- **Do not add a GPU accelerator.** TF-IDF and `scipy.sparse` have no GPU path; a GPU session
  costs CPU cores and RAM, which are the binding constraints. `--jobs` is the only lever and
  the config cell sizes it from the actual machine.
- **A two-stage rare-trigram search was *slower*** — 26–72s/1k queries vs 15s single-stage for
  64.3% vs 65.0% recall. Kept behind `RARE_TRIGRAMS=0` in `baseline.py` with the numbers
  recorded, so nobody re-derives it.
- **Postal extraction is position-aware.** The discriminator is what *follows* the digit run in
  its comma segment: a street word or bare digits mean a house number; a plain word or segment
  end mean a postal code. A naive rule reported 72% "disagreement" on US pairs that were all
  house numbers, and a purely positional rule threw away real French codes (`33000 BORDEAUX`
  has the same shape as `11244 Westfall Road`). 15 real addresses pin this in
  `tests/test_normalize.py`.
- **Measure similarity at full scale.** Finding 1 inverted between a 10k sample and 10.3M.

---

## 9. Reproducing locally

```bash
cd student_resource
python3.11 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
.venv/bin/python -m pytest code/business_entity_resolution/tests/ -q     # 142 pass

# smoke-run any step on a sample; never run full data on a laptop
.venv/bin/python code/business_entity_resolution/src/run_step.py \
    --step p1c_eda --sample 2000 --report-dir /tmp/smoke
```

Local env is numpy 2.4 / pandas 2.3 to mirror Kaggle. `lightgbm` needs `brew install libomp`
on macOS.

Reports: `reports/<id>.md` plus `reports/p1c_eda/*.csv` (30 tables, displayed by
`notebooks/01_eda.ipynb` — the notebook only loads and displays, no logic).
Experiment log: `EXPERIMENTS.md`.
