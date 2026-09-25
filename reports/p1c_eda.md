# Step `p1c_eda`

- date: 2026-09-25
- runtime: 448.3s   peak RAM: 6.12 GB
- data-dir: `dataset`   sample: full

**scope:** full train + full test  
Full tables: `reports/p1c_eda/*.csv`, displayed by `notebooks/01_eda.ipynb`.

#### One-screen summary

| fact | value |
| --- | --- |
| scope | full train + full test |
| train S1 / test S1 entities | 2,206,821 / 1,732,544 |
| test/train ratio S1 / S2 / S3 | 0.79 / 0.97 / 0.96 |
| singletons (train) | 5.58% |
| matches per non-singleton: mean / max | 3.67 / 11 |
| matched IDs from S2 / S3 | 48.36% / 51.64% |
| IDs under more than one S1 | 0 |
| true pairs whose country differs | 0 |
| countries new in test | France (14.98% of test S1) |
| empty address: train S2 / S3 / test S2 / S3 | 3.4% / 3.3% / 2.6% / 2.7% |
| true pairs where both carry a postal code / codes agree | 0.16% / 98.75% (naive digit runs: 93.49%) |
| AUC name vs address (random negatives) | 0.976 vs 0.956 |
| true pairs agreeing on name only / address only | 10.1% / 16.1% |
| singleton traps: NN cosine >= 0.8 | 92.7% of singletons |
| matched S1: TF-IDF NN is a true match | 48.3% |
| 3+ match entities: extras that duplicate the S1 (name+addre… | 52.3% of records; branch-like in 2.0% of entities |
| same-source records within an entity that are near-copies | 47.0% of 10,652 pairs |
| new-country noise gaps matching train direction | 17/17 |

#### Assumptions

| assumption | verdict | evidence |
| --- | --- | --- |
| (a) each S2/S3 has at most one owner | holds | 0 of 7,638,365 matched IDs appear under more than one S1 |
| (b) names carry most of the signal | partly | AUC name 0.976 vs address 0.956; true pairs agreeing on nam… |
| (c) postal codes are reliable when present | holds | a genuine code is on both records in only 0.16% of true pai… |
| (d) France shows the same noise patterns in a new language | holds | 17/17 S2/S3-vs-S1 noise gaps in France point the same way a… |

#### 1. Ground truth

| matches | S1 entities | share % |
| --- | --- | --- |
| 0 | 123,247 | 5.58 |
| 1 | 119,157 | 5.4 |
| 2 | 375,212 | 17 |
| 3 | 530,841 | 24.05 |
| 4 | 484,115 | 21.94 |
| 5+ | 574,249 | 26.02 |

| metric | value |
| --- | --- |
| matched IDs | 7,638,365 |
| matched IDs from S2 % | 48.36 |
| matched IDs from S3 % | 51.64 |
| S2 records matching no S1 % | 26.64 |
| S3 records matching no S1 % | 25.37 |

#### 2. Ownership

| metric | value |
| --- | --- |
| matched IDs | 7,638,365 |
| IDs under more than one S1 | 0 |
| max owners of a single ID | 1 |

#### 3. Sizes

| file | train rows | test rows | test/train | train MB | test MB |
| --- | --- | --- | --- | --- | --- |
| source1 | 2,206,821 | 1,732,544 | 0.785 | 200.3 | 166.9 |
| source2 | 5,034,616 | 4,887,273 | 0.971 | 466.6 | 485.9 |
| source3 | 5,285,603 | 5,082,316 | 0.962 | 480.4 | 482.6 |
| ground_truth | 2,206,821 |  |  | 121.1 |  |

#### 4. Country

| split | source | France | India | US |
| --- | --- | --- | --- | --- |
| test | S1 | 259,452 (15.0%) | 809,986 (46.8%) | 663,106 (38.3%) |
| test | S2 | 703,378 (14.4%) | 2,312,565 (47.3%) | 1,871,330 (38.3%) |
| test | S3 | 731,615 (14.4%) | 2,405,000 (47.3%) | 1,945,701 (38.3%) |
| train | S1 | - | 883,188 (40.0%) | 1,323,633 (60.0%) |
| train | S2 | - | 2,017,799 (40.1%) | 3,016,817 (59.9%) |
| train | S3 | - | 2,115,547 (40.0%) | 3,170,056 (60.0%) |

| raw label (repr) | normalized key | count | raw spellings of key |
| --- | --- | --- | --- |
| 'US' | us | 11,990,643 | 1 |
| 'India' | india | 10,544,085 | 1 |
| 'France' | france | 1,694,445 | 1 |

#### 5. Fields

| split | source | rows | business_name empty % | business_address empty % | country empty % |
| --- | --- | --- | --- | --- | --- |
| train | S1 | 2,206,821 | 0 | 0 | 0 |
| train | S2 | 5,034,616 | 0 | 3.36 | 0 |
| train | S3 | 5,285,603 | 0 | 3.33 | 0 |
| test | S1 | 1,732,544 | 0 | 0 | 0 |
| test | S2 | 4,887,273 | 0 | 2.65 | 0 |
| test | S3 | 5,082,316 | 0 | 2.68 | 0 |

| split | source | country | rows | name len median | name len p95 | addr len median | addr len p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | S1 | US | 1,323,633 | 22 | 35 | 34 | 48 |
| train | S1 | India | 883,188 | 27 | 38 | 76 | 116 |
| train | S2 | India | 2,017,799 | 27 | 42 | 69 | 110 |
| train | S2 | US | 3,016,817 | 23 | 39 | 32 | 44 |
| train | S3 | US | 3,170,056 | 23 | 40 | 39 | 53 |
| train | S3 | India | 2,115,547 | 27 | 43 | 60 | 106 |
| test | S1 | US | 663,106 | 22 | 35 | 34 | 48 |
| test | S1 | France | 259,452 | 19 | 29 | 48 | 65 |
| test | S1 | India | 809,986 | 27 | 38 | 76 | 116 |
| test | S2 | India | 2,312,565 | 28 | 44 | 69 | 110 |
| test | S2 | France | 703,378 | 20 | 35 | 41 | 60 |
| test | S2 | US | 1,871,330 | 24 | 39 | 32 | 44 |
| test | S3 | India | 2,405,000 | 28 | 44 | 59 | 106 |
| test | S3 | France | 731,615 | 20 | 36 | 41 | 60 |
| test | S3 | US | 1,945,701 | 24 | 41 | 39 | 54 |

#### 6. Formats (from text, per country)

| split | source | country | rows | 5-digit % | 6-digit % | ddd ddd % | 5-digit at start % | near/opp/behind % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | S1 | US | 1,323,633 | 10.95 | 0.13 | 0.6 | 9.27 | 0.01 |
| train | S1 | India | 883,188 | 0.3 | 0.02 | 0.34 | 0.05 | 13.06 |
| train | S2 | India | 2,017,799 | 1.13 | 0.02 | 1.16 | 0.22 | 11.16 |
| train | S2 | US | 3,016,817 | 10.45 | 1.39 | 0.49 | 8.5 | 0.01 |
| train | S3 | US | 3,170,056 | 10.48 | 1.33 | 0.51 | 8.48 | 0.01 |
| train | S3 | India | 2,115,547 | 1.05 | 0.02 | 1.1 | 0.21 | 8.48 |
| test | S1 | US | 663,106 | 10.95 | 0.12 | 0.6 | 9.28 | 0.01 |
| test | S1 | France | 259,452 | 0.41 | 0.01 | 0 | 0.03 | 0 |
| test | S1 | India | 809,986 | 0.28 | 0.02 | 0.35 | 0.04 | 13.04 |
| test | S2 | India | 2,312,565 | 1.1 | 0.02 | 1.17 | 0.21 | 11.22 |
| test | S2 | France | 703,378 | 0.51 | 0.02 | 0 | 0.23 | 0 |
| test | S2 | US | 1,871,330 | 10.68 | 1.43 | 0.5 | 8.67 | 0.01 |
| test | S3 | India | 2,405,000 | 1.01 | 0.02 | 1.13 | 0.2 | 8.42 |
| test | S3 | France | 731,615 | 0.53 | 0.02 | 0 | 0.23 | 0 |
| test | S3 | US | 1,945,701 | 10.72 | 1.37 | 0.51 | 8.69 | 0.01 |

#### Singleton traps / 3+ match entities

| group | queries | cosine median | cosine p90 | cosine >= 0.8 % | cosine >= 0.9 % | nn is true match % | nn owned by another S1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched | 300 | 1 | 1 | 99.3 | 96.7 | 48.3 | 37 |
| singleton | 300 | 1 | 1 | 92.7 | 79.7 |  | 47.7 |

| category | records | share % |
| --- | --- | --- |
| duplicate (same name + address) | 6,710 | 52.3 |
| partial variant | 3,148 | 24.5 |
| name variant at same address | 2,331 | 18.2 |
| no address, same name | 484 | 3.8 |
| no address, name differs | 70 | 0.5 |
| same name, different address (branch?) | 66 | 0.5 |
| different name and address | 21 | 0.2 |

#### Evidence

| metric | value |
| --- | --- |
| true pairs / random same-country negatives | 50,000 / 50,000 |
| AUC name tsr (normalized) | 0.9762 |
| AUC address tsr (normalized) | 0.9562 |
| true pairs: name AND address agree (>=80) % | 71.99 |
| true pairs: name only agrees % | 10.14 |
| true pairs: address only agrees % | 16.12 |
| true pairs: neither agrees % | 1.75 |
| true pairs: name tsr < 50 % | 7.21 |
| true pairs: address tsr < 50 % | 4.58 |
| true pairs: match record has no address % | 4.38 |

| country | true pairs | both have code % | agree % | both 6-digit | 6-digit agree % | both 5-digit | 5-digit agree % | naive: both % | naive: agree % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| India | 80,442 | 0.31 | 98.8 | 148 | 100 | 102 | 97.06 | 0.54 | 98.38 |
| US | 119,558 | 0.06 | 98.59 | 7 | 100 | 64 | 98.44 | 9.19 | 93.29 |
| ALL | 200,000 | 0.16 | 98.75 | 155 | 100 | 166 | 97.59 | 5.71 | 93.49 |

Postal conflicts that survive the refined extractor (first 5):

| country | s1 codes | s1 address | match codes | match address |
| --- | --- | --- | --- | --- |
| India | 22244 | 22244, B2T2, Prestige, Jindal City, Tumkur Road, Bangalore … | 72244 | Bangalore North, 72244, ಕರ್ನಾಟಕ |
| US | 12059 | 20349 Cotton Slash Road, Bldg 12059, Marysville, OH | 20349 | 20349-20351 COTTON SLASH RD, MARYSVILLE, OH |
| India | 11024 | 11024, S/F, Gali Peepal Wali, Motia Khan, Pahar Ganj, Centr… | 11022 | 11022, S/F, GALI PEEPAL WALI, MOTIA KHAN, PAHAR GANJ, CENTR… |
| India | 11112 | 11112, Ats Prestine Sector-150, Noida, Gautam Buddha Nagar,… | 11110 | 11110, Ats Prestine Sector-150, Gautam Buddha Nagar, Noida,… |

