# Handoff — Business Entity Resolution (Amazon ML Challenge)

**Written for: the engineer (and their Claude Code) taking over from p3.**

Read `CLAUDE.md` first — it is the operating contract and its hard rules are not
negotiable. Then `PROBLEM.md` for the official statement. This file is what those two do
not tell you: what has been measured, what was tried and rejected, and where the traps are.

State at handoff: **p0–p2b complete, 142 tests passing, one unsubmitted baseline.**
Latest commit on `main` is the source of truth; `EXPERIMENTS.md` is the running log.

---

## 1. The task in one paragraph

For every Source-1 entity, output which S2/S3 records are the same real business. Scored by
**macro-averaged F0.5 per S1 entity** — precision weighted 2× over recall, averaged over
*all* S1 entities including singletons. Two output files: `output/matching_results.tsv`
(scored) and `output/candidate_pairs.tsv` (your blocking set; matches must be a subset).

---

## 2. Facts established by measurement

All of these are from full-data runs, not samples. Reports are in `reports/`.

### Structural (exploit these)

| Fact | Evidence |
|---|---|
| **Each S2/S3 record belongs to at most one S1.** Exclusivity holds exactly. | 0 of 7,638,365 matched IDs appear under two S1s (`reports/p1c_eda.md`) |
| **True matches never cross country.** | 0 of 7,638,365 pairs (same-country blocking costs zero recall) |
| Singletons are **5.58%** of train entities | 123,247 of 2,206,821 |
| Matches per non-singleton: mean **3.67**, max **11** | — |
| ~26% of S2 and ~25% of S3 records match no S1 at all | distractors |

Exclusivity is the strongest unused lever. A per-pair classifier will happily assign one
record to several S1s; a **global assignment** cannot. 37% of matched entities' and 47.7%
of singletons' nearest neighbours are already owned by another S1, so the constraint
actively rejects wrong candidates.

### The finding that should shape p3

**A business name cannot identify a business at this scale.** Searching all 10.3M S2/S3
names by char-3gram TF-IDF cosine:

- median nearest-neighbour cosine is **1.0 for singletons as well as matched entities**
- **92.7%** of singletons have a name twin at cosine ≥ 0.8
- for a matched S1, the name-nearest-neighbour is the true match only **48.3%** of the time

On a 10k sample the same measurement said 8.6% and 92% — the *opposite* conclusion.
**Do not trust a candidate-ranking measurement taken on a subsample.**

Corollary: among true pairs, **16.1% agree on address while disagreeing on name**, versus
10.1% the other way. The name's higher AUC (0.976 vs 0.956) is measured against random
negatives, which are trivially easy. Address is not a tiebreaker here; it is co-primary.

### Field quality

- Empty addresses: ~3.4% of S2/S3 (train), ~2.6% (test). Real values, never NaN.
- Non-Latin names (Devanagari): ~15% of S2, ~11% of S3. `unidecode` transliterates them.
- **Postal codes are accurate but nearly absent**: a genuine code sits on both records in
  **0.16%** of true pairs, agreeing 98.75% when present. Naive 5/6-digit matching looks 36×
  more available but agrees only 93.5% because most runs are house numbers. Useful as a
  feature when present; useless as a blocking key.
- 52.3% of matched records repeat the S1's name *and* address; only 0.5% are a same-name
  different-address branch. The real job is recognising corrupted restatements.

### France (unseen in training)

15% of test S1. All 17 S2/S3-vs-S1 noise gaps that US and India agree on point the same way
for France, so the corruption process looks language-independent. Its addresses are shorter
(median 49 vs 76 for India) and it never uses the `near/opp/behind` landmarks that appear in
~12% of Indian addresses. Both detected from text, never from the country label.

---

## 3. The metric, and what it implies

`metrics.py` implements it exactly; 21 tests including the PROBLEM.md 0.714 example.
Validated against real ground truth on all 2.2M entities:

| Baseline | macro F0.5 |
|---|---|
| Perfect | 1.00000 |
| **Always predict empty** | **0.05585** — exactly the singleton rate |
| Drop one true match per entity | 0.87105 |
| **Add one false ID per entity** | **0.75154** |

**One false positive costs ~2× one miss.** Adding a single false ID also wipes the
singleton bucket from 1.0 to 0.0, which alone is 5.58% of the score. Any threshold or
decision rule should be tuned against this asymmetry, not against F1 intuition.

---

## 4. Where p2b got to, and its ceiling

Name-only char-3gram TF-IDF on `core_name`, top-5 by cosine, one global threshold (0.975).

| | |
|---|---|
| CV macro F0.5 (grouped by S1, 200k entities) | **0.3269** |
| LOCO India→US / US→India | 0.3674 / 0.2660 |
| Singleton bucket | **0.136** (vs 1.0 for predict-nothing) |
| Blocking pair recall | 35.4% |
| Entities with ≥1 true candidate | 57.8% |
| Runtime (full test inference) | ~5 h on 4 Kaggle cores |

**This is a floor, not a starting point to tune.** The hard ceiling was measured: an
*exact* name-only top-5 over the whole pool — no pruning, ~42 h of compute — recovers only
**74.3%** of true matches. No model on top of name-only candidates can beat that.

The LOCO spread (0.367 vs 0.266) says the single global threshold transfers badly across
countries. France has no training rows at all, so its cosine scale was never validated.

---

## 5. What p3 should probably do

Not prescriptive, but these follow directly from the measurements:

1. **Blocking must use address, not just name.** p1b measured union recall: `name token OR
   addr token` = **99.99%** of true pairs, versus 85.6% for name tokens alone. That is the
   recall ceiling worth building on. `reports/p1_eda.md` §6 has per-key numbers.
2. **A supervised pair classifier** (LightGBM is pinned and installed) over features from
   both fields: rapidfuzz ratios on `core_name` and normalized address, token/trigram
   Jaccard, shared postal code, shared street number, `same_country`, length ratios, plus
   *blocking rank and cosine* as features.
3. **Then a global assignment** enforcing exclusivity — each S2/S3 record to at most one S1.
   This is what converts a good classifier into a precision win, and F0.5 pays 2× for it.
4. **Calibrate the empty-prediction decision separately.** We currently predict empty for
   3.9% of test entities while ~5.6% are true singletons; that gap is pure false merges.
5. **Validate with CV *and* LOCO.** CV alone will mislead you about France.

---

## 6. Things already tried and rejected — do not redo

| Tried | Result | Where |
|---|---|---|
| Two-stage rare-trigram blocking + exact rescore | **Slower**, 26–72s/1k vs 15s/1k single-stage, for 64.3% vs 65.0% recall. Per-query Python rescore costs more than pruning saves. | `baseline.py`, `RARE_TRIGRAMS=0` |
| Parquet/int32 encoding of the submission | 47 MB vs gzip's 46 MB. IDs are high-entropy; ~46 MB is the floor. | — |
| Positional-only postal extraction | Discarded real French codes, which precede the city exactly where a US house number sits. | `normalize.py` |
| `sentence-transformers` / FAISS over the full corpus | 26M records × 384 dims ≈ 40 GB of vectors. Infeasible; commented out of requirements. | `requirements.txt` |
| `numpy<2` pin | Breaks the C ABI on Kaggle/Colab images built against numpy 2.x. Use floors for compiled packages. | `requirements.txt` |

---

## 7. How to run anything

Everything goes through one entry point, from the repo root:

```bash
python code/business_entity_resolution/src/run_step.py --step <id> [flags]
```

Step ids: `p1a_check`, `p1b_eda`, `p1c_eda`, `p2a_splits`, `p2b_baseline`.
Common flags: `--data-dir`, `--out-dir`, `--sample N`, `--jobs N`, `--query-chunk N`,
`--threshold F`, `--skip-test`, `--report-dir`, `--force`, `--seed`. No absolute paths
anywhere; every step writes `reports/<id>.md` with runtime and peak RAM.

**Compute model (from CLAUDE.md):** the laptop only writes code, runs pytest, and
smoke-runs with `--sample`. Full runs happen on Kaggle via `notebooks/runner.ipynb`, which
clones the repo, stages the dataset, runs the step, validates, and pushes reports plus the
gzipped submission back. Set `STEP` in its config cell and Run All.

```bash
.venv/bin/python -m pytest code/business_entity_resolution/tests/ -q     # 142 tests
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

Always run the validator with `--check-ids` before uploading. A submission that fails
validation is not evaluated at all.

---

## 8. Traps that have already cost us time

- **`output/*.tsv` is gitignored.** A 5.8 h run's predictions were stranded on Kaggle
  because only the report was pushed. The runner now commits
  `output/matching_results.tsv.gz`; `notebooks/recover_submission.py` rescues a live
  session if needed.
- **Kaggle "Save Version" throws `ConcurrencyViolation`.** Do not depend on it — the runner
  pushes to git during the run, and drops copies at the top of `/kaggle/working`.
- **A silent step looks hung.** Steps now print progress with ETA about once a minute; the
  runner sets `PYTHONUNBUFFERED=1` and reads with `readline`. A run was killed at ~4 h
  because it printed nothing.
- **Never accelerate with a GPU.** Nothing here has a GPU path; a GPU session costs cores
  and RAM, the actual constraints.
- **Peak memory is the sparse product, not the index** — ~1.1 MB per query per worker.
  `--query-chunk 500` keeps 4 workers near 2 GB; 2000 would be 8.5 GB.
- **Measure candidate quality at full scale.** The 10k-sample result was the opposite of the
  10.3M-record truth.

---

## 9. Still outstanding

1. **No submission has been uploaded yet.** The p2b run finished (`reports/p2b_baseline.md`,
   7,283,409 predicted pairs) but the `.gz` was not committed because the Kaggle notebook
   in use had a stale push cell. Either re-run `p2b_baseline --threshold 0.975` with the
   current notebook (~4.5 h), or skip straight to p3 — the baseline is only worth ~0.33.
2. **`EXPERIMENTS.md` has no LB column filled in**, because of (1). Getting one real
   leaderboard number is worth doing early to calibrate CV.
3. **`Documentation_template.md` is untouched** — required in the final zip.
4. **`candidate_pairs.tsv` is not committed** (only the matching file), by design; it is
   needed for the final submission zip, so retrieve it from the last run's output.
