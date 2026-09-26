# Step `p2b_baseline`

- date: 2026-09-26
- runtime: 18215.0s   peak RAM: 7.18 GB
- data-dir: `dataset`   sample: full

**config:** char 3-gram TF-IDF on core_name, top-5 by cosine, max_df=0.01. Threshold **0.975** supplied via `--threshold`; tuning, CV and LOCO skipped -- see an earlier `p2b_baseline` report for those.

#### Test submission

| metric | value |
| --- | --- |
| test S1 rows | 1,732,544 |
| candidate pairs | 8,413,358 |
| predicted pairs (>= threshold) | 7,283,409 |
| S1 with a prediction | 1,664,930 |
| S1 predicted empty | 67,614 |
| S1 predicted empty % | 3.9 |
| matching_results.tsv MB | 110.9 |
| matching_results.tsv.gz MB | 46.3 |

Submittable file is committed as `output/matching_results.tsv.gz`; expand with `python -c "import sys; sys.path.insert(0,'code/business_entity_resolution/src'); import io_utils; io_utils.gunzip_file('output/matching_results.tsv.gz')"`.

