# Step `p2a_splits`

- date: 2026-09-25
- runtime: 209.8s   peak RAM: 0.80 GB
- data-dir: `dataset`   sample: full

Folds: `artifacts/folds.parquet` (2,206,821 S1 entities, 5 folds, seed 42)

#### Leakage checks

| check | pass | detail |
| --- | --- | --- |
| every S1 in exactly one fold | True | 0 span >1 fold |
| no duplicate S1 rows | True | 0 duplicates |
| all folds non-empty | True |  |
| singleton rate spread <= 0.050pp | True | spread 0.0002pp |
| no unlabelled country | True | 0 blank |

#### Fold sizes and singleton rate

| fold | S1 entities | singletons | singleton % | mean matches | true pairs | India % | US % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 441,365 | 24,649 | 5.58 | 3.46 | 1,526,609 | 40.02 | 59.98 |
| 1 | 441,364 | 24,650 | 5.58 | 3.46 | 1,527,140 | 40.02 | 59.98 |
| 2 | 441,364 | 24,650 | 5.58 | 3.46 | 1,528,058 | 40.02 | 59.98 |
| 3 | 441,364 | 24,649 | 5.58 | 3.46 | 1,528,353 | 40.02 | 59.98 |
| 4 | 441,364 | 24,649 | 5.58 | 3.46 | 1,528,205 | 40.02 | 59.98 |
| ALL | 2,206,821 | 123,247 | 5.58 | 3.46 | 7,638,365 | 40.02 | 59.98 |

#### Leave-one-country-out

| split | train S1 | train true pairs | train singleton % | eval S1 | eval true pairs | eval singleton % |
| --- | --- | --- | --- | --- | --- | --- |
| India->US | 883,188 | 3,059,843 | 5.59 | 1,323,633 | 4,578,522 | 5.58 |
| US->India | 1,323,633 | 4,578,522 | 5.58 | 883,188 | 3,059,843 | 5.59 |

