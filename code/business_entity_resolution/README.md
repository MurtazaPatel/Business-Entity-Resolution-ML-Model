# Business Entity Resolution — pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the given
`dataset/` files. Run everything from the repo root (`student_resource/`).

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Run a step

```bash
python code/business_entity_resolution/src/run_step.py --step <id> [--sample N]
```

Every step accepts `--data-dir`, `--out-dir`, `--sample`, `--force` and `--seed`. Paths
default to `<repo>/dataset` and `<repo>/output`, resolved relative to this file rather
than the cwd, so a clone on Kaggle works unchanged. Nothing is hard-coded to an absolute
path.

Each run writes `reports/<id>.md` with the step's metrics, runtime and peak RAM.

| step | what it does |
| --- | --- |
| `p0_load` | Loads all 7 given files; reports shape, columns and 3 sample rows each. |

## Where the work happens

```
src/config.py     paths, SEED=42, shared CLI flags
src/io_utils.py   TSV reading (all-string, no NA coercion) and spec-exact submission writing
src/run_step.py   step registry + reports/<id>.md
tests/            contract tests, incl. a run against utils/validate_submission.py
```

## Scale note

The full data is ~26M records / 2.5 GB. Development machines smoke-run with `--sample N`;
full runs happen on Kaggle. Frames are backed by pyarrow strings, which keeps a 5M-row
source near 0.6 GB instead of ~2 GB under object dtype.

## Tests

```bash
.venv/bin/python -m pytest code/business_entity_resolution/tests/ -q
```

## Validate a submission

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
