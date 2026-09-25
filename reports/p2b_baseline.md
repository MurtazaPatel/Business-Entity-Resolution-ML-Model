# Step `p2b_baseline`

- date: 2026-09-25
- runtime: 20933.7s   peak RAM: 9.34 GB
- data-dir: `dataset`   sample: full

**config:** char 3-gram TF-IDF on core_name, top-5 by cosine, max_df=0.01, one global threshold. Tuned on 200,000 S1 entities.

#### Blocking (before any threshold)

| metric | value |
| --- | --- |
| pair recall % | 35.4 |
| entities with >=1 true candidate % | 57.77 |
| candidate pairs | 983,200 |
| S1 with no candidate | 3,360 |
| S1 with no candidate % | 1.68 |

#### Per-country blocking

| country | queries | pool | vocab | nnz per doc | no candidate | no candidate % | build s | search s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| India | 79,860 | 4,133,346 | 21,429 | 6.65 | 1,644 | 2.06 | 102.7 | 646.8 |
| US | 120,140 | 6,186,873 | 23,091 | 7.52 | 1,716 | 1.43 | 153.3 | 2,782.8 |

#### Threshold sweep

| threshold | macro F0.5 | singletons | non-singletons | mean precision | mean recall | predicted empty % | pairs kept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.2552 | 0.01681 | 0.2695 | 0.2461 | 0.3436 | 1.68 | 983,200 |
| 0.1 | 0.2552 | 0.01681 | 0.2695 | 0.2461 | 0.3436 | 1.68 | 983,200 |
| 0.2 | 0.2552 | 0.01681 | 0.2695 | 0.2461 | 0.3436 | 1.68 | 983,200 |
| 0.4 | 0.2553 | 0.01681 | 0.2695 | 0.2461 | 0.3436 | 1.68 | 983,173 |
| 0.8 | 0.2853 | 0.04535 | 0.2996 | 0.2845 | 0.3346 | 2.05 | 911,101 |

Best threshold: **0.975**

#### CV (grouped by S1)

| fold | threshold | eval S1 | macro F0.5 | singletons | non-singletons | mean precision | mean recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.975 | 39,988 | 0.3284 | 0.1479 | 0.3393 | 0.3493 | 0.3132 |
| 1 | 0.975 | 40,136 | 0.3262 | 0.1153 | 0.3384 | 0.3466 | 0.313 |
| 2 | 0.975 | 39,792 | 0.3256 | 0.1353 | 0.3374 | 0.3457 | 0.3118 |
| 3 | 0.975 | 40,049 | 0.3255 | 0.1446 | 0.3362 | 0.3455 | 0.3119 |
| 4 | 0.975 | 40,035 | 0.3288 | 0.1356 | 0.3401 | 0.3493 | 0.315 |
| mean | 0.975 | 200,000 | 0.3269 | 0.1358 | 0.3383 | 0.3473 | 0.313 |

#### LOCO

| split | threshold | eval S1 | macro F0.5 | singletons | non-singletons | mean precision | mean recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| India->US | 0.975 | 120,140 | 0.3674 | 0.154 | 0.3801 | 0.3921 | 0.3481 |
| US->India | 0.975 | 79,860 | 0.266 | 0.1086 | 0.2753 | 0.2799 | 0.2601 |

#### Test submission

| metric | value |
| --- | --- |
| test S1 rows | 1,732,544 |
| candidate pairs | 8,413,358 |
| predicted pairs (>= threshold) | 7,283,409 |
| S1 with a prediction | 1,664,930 |
| S1 predicted empty | 67,614 |
| S1 predicted empty % | 3.9 |

