# Experiments

One row per completed step. Appended only after full-run results come back from Kaggle,
never at smoke-run time. CV = grouped-by-S1 K-fold on train; LOCO = leave-one-country-out;
LB = public leaderboard.

| date | step | change | CV F0.5 | LOCO F0.5 | LB | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-25 | p0_load | Repo scaffold: config/io_utils/run_step, pinned env, load-check | – | – | – | Baseline only, no model. 26M records across 7 files; train singleton rate 5.6%. |
