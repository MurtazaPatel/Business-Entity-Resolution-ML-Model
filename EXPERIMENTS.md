# Experiments

One row per completed step. Appended only after full-run results come back from Kaggle,
never at smoke-run time. CV = grouped-by-S1 K-fold on train; LOCO = leave-one-country-out;
LB = public leaderboard.

| date | step | change | CV F0.5 | LOCO F0.5 | LB | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-25 | p0_load | Repo scaffold: config/io_utils/run_step, pinned env, load-check | – | – | – | Baseline only, no model. 26M records across 7 files; train singleton rate 5.6%. |
| 2026-09-25 | p1b_eda | Full-train EDA: invariants + blocking-key recall (300k probe) | – | – | – | Exclusivity HOLDS (7,638,365 IDs, 0 dupes). 100% of true pairs same country. Singletons 5.58%. Union `name token OR addr token` = 99.99% recall; misses are empty-address + typo'd short names. Postal code in ~7% of addresses, only 5% pair recall. 273s, 9.26 GB peak (object-dtype conversions; p1c goes Arrow-native). |
| 2026-09-25 | p1c_eda | Deep EDA on full train+test; 30 tables to reports/p1c_eda/ | – | – | – | Name alone is NOT discriminative at scale: median name-NN cosine 1.0 even for singletons, 92.7% of singletons have a twin >=0.8, and a matched S1's NN is the true match only 48.3% of the time. True pairs agree on address-only (16.1%) more than name-only (10.1%). Postal codes agree 98.75% but both records carry one in 0.16% of pairs. 52.3% of matches duplicate the S1 outright. France: 17/17 noise gaps match train direction. 448s, 6.12 GB. |
| 2026-09-25 | p2a_splits | metrics.py (macro F0.5) + splits.py (5-fold StratifiedGroupKFold, LOCO) | – | – | – | 84 tests pass incl. PROBLEM.md 0.714 example. Folds: 441,364-441,365 S1 each, singleton rate 5.58% (spread 0.0002pp), no S1 spans folds. LOCO US->India / India->US. Metric validated on real GT: perfect 1.0, always-empty 0.05585 (= singleton rate), drop-1-match 0.871, add-1-false 0.752 -> a false positive costs ~2x a miss. 210s, 0.80 GB. |
| 2026-09-25 | p2b_baseline | Name-only char 3-gram TF-IDF on core_name, top-5 cosine, one global threshold | 0.3327 | 0.3728 / 0.2711 | – | **Local smoke only (6,000 train S1); full Kaggle run pending.** Threshold tunes to 0.975 (F0.5 pushes it very high). Singleton bucket only 0.156 vs 1.0 for always-empty, so the baseline destroys singletons to win on matches. Blocking pair-recall 35.6%; 57.8% of entities get >=1 true candidate. Measured ceiling: exact name-only top-5 recall@5 is 74.3% and costs ~42h, so the name alone cannot go much further -- p2c needs address features. LOCO gap 0.37 vs 0.27 (India->US easier than US->India). |
