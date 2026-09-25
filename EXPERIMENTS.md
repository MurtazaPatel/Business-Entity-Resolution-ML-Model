# Experiments

One row per completed step. Appended only after full-run results come back from Kaggle,
never at smoke-run time. CV = grouped-by-S1 K-fold on train; LOCO = leave-one-country-out;
LB = public leaderboard.

| date | step | change | CV F0.5 | LOCO F0.5 | LB | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-25 | p0_load | Repo scaffold: config/io_utils/run_step, pinned env, load-check | – | – | – | Baseline only, no model. 26M records across 7 files; train singleton rate 5.6%. |
| 2026-09-25 | p1b_eda | Full-train EDA: invariants + blocking-key recall (300k probe) | – | – | – | Exclusivity HOLDS (7,638,365 IDs, 0 dupes). 100% of true pairs same country. Singletons 5.58%. Union `name token OR addr token` = 99.99% recall; misses are empty-address + typo'd short names. Postal code in ~7% of addresses, only 5% pair recall. 273s, 9.26 GB peak (object-dtype conversions; p1c goes Arrow-native). |
