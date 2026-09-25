# Step `p1_eda`

- date: 2026-09-25
- runtime: 273.0s   peak RAM: 9.26 GB
- data-dir: `dataset`   sample: full

**scope:** full train

#### 1. Exclusivity: does each S2/S3 record belong to at most one S1?

- **HOLDS.** All 7,638,365 matched IDs are distinct.
- Exploitable: a record may be assigned to at most one S1, so scoring can be
  a global assignment rather than independent per-pair decisions.

#### 2. How many matches per S1 entity?

- S1 entities: **2,206,821**   matched IDs: **7,638,365** (S2 3,693,619 / S3 3,944,746)
- singletons (no true match): **123,247** = **5.58%**
- matches per non-singleton: mean **3.67**, median 4, max **11**

| matches | S1 entities | share |
| --- | --- | --- |
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

> Singletons are only 5.58% of entities, so a blanket 'predict empty' baseline scores about that. Recall matters more than the F0.5 framing alone suggests.

#### 3. What share of S2/S3 records is matched to any S1?

- **S2**: 5,034,616 records, 3,693,619 matched (73.36%) -> 26.64% are distractors with no S1.
- **S3**: 5,285,603 records, 3,944,746 matched (74.63%) -> 25.37% are distractors with no S1.
- ground-truth IDs absent from the source files: **0** (should be 0)

#### 4. Field quality

| frame | rows | empty name | empty addr | non-Latin name | addr w/o digit | postal found |
| --- | --- | --- | --- | --- | --- | --- |
| train_s1 | 2,206,821 | 0.00% | 0.00% | 0.00% | 3.47% | 6.79% |
| train_s2 | 5,034,616 | 0.00% | 3.36% | 15.16% | 9.37% | 7.49% |
| train_s3 | 5,285,603 | 0.00% | 3.33% | 11.41% | 9.26% | 7.43% |

#### 5. Do true matches cross country?

- same country: **7,638,365** (100.00%)
- different country: **0** (0.00%)
- unresolved IDs: 0
- **verdict: blocking within country costs ~0 recall -- partition the problem by country label (a same_country constraint, not a hard-coded country list)**

#### 6. Blocking-key recall on true pairs (probe n=300,000)

Share of true pairs each key would keep. A key below ~90% cannot be the sole
blocker; the union of cheap keys sets the achievable recall ceiling.

| blocking key | recall on true pairs |
| --- | --- |
| exact normalized name | **47.82%** |
| name tokens sorted | **52.62%** |
| share >=1 name token | **85.62%** |
| share >=2 name tokens | **72.32%** |
| share a name trigram | **96.86%** |
| name trigram jaccard>=0.3 | **85.91%** |
| share a postal code | **5.03%** |
| share a street number | **76.24%** |
| share >=1 addr token | **95.55%** |
| same country | **100.00%** |

**Union schemes** (pair kept if any key fires):

| scheme | recall |
| --- | --- |
| name token OR name trigram | **97.05%** |
| name trigram OR addr token | **99.99%** |
| name token OR addr token | **99.99%** |
| name trigram OR addr token OR street number | **99.99%** |

Pairs that survive NO key in 'name token OR addr token' (23 of 300,000 = 0.01%) -- these bound recall:

- `L 7 Prime Hudson` @ `13318 Lakeshore Road, Town Of Centerville, WI` **vs** `L 7` @ ``
- `Synva Corp` @ `617 7, Manchester, NY` **vs** `Sybova Corp` @ `617B 7, Clifton Springs, New York`
- `P+ Purpose` @ `4365 Harris Road, Oxford, OH` **vs** `P+ Pupmsoe` @ ``
- `J & O Grupo, Inc` @ `1114 Pleasant Valley Road, Garland, TX` **vs** `J & O Gnup,  Inc` @ ``
- `OF India Pvt Ltd` @ `Abgila, Jagdishpur Buniyadganj, Manpur, Gaya, Bihar` **vs** `OF Pvt Ltd Cénter` @ ``
- `#2 Transport` @ `3818 Mizell Road, Unit F, Greensboro, NC` **vs** `#2 Traneport` @ ``
- `E/L Seafood` @ `17 Berkshire Street, Norfolk, MA` **vs** `E/L Seaf0od` @ ``
- `L & I Fusion` @ `4111 Highgrove Drive, Dallas, TX` **vs** `L & I` @ ``
- `SA Solutions Private Limited` @ `At Lakadi Po Nirgude Tal Indapur Baramati, Baramati, Pune, Maharashtra` **vs** `SA Stomurtons Private Limited` @ ``
- `N+ Otg Inc.` @ `1268 Buenos Aires, AZ, Sahuarita` **vs** `N+ 0tg` @ ``

#### 7. Name-token frequency (probe n=300,000 S1 names)

- distinct tokens: **56,154**   occurrences: 757,390
- tokens appearing once: 43.76% of the vocabulary

Most common tokens (these are near-useless as block keys):

| token | count |
| --- | --- |
| india | 8,140 |
| care | 7,196 |
| pc | 6,022 |
| associates | 5,880 |
| group | 5,484 |
| center | 4,833 |
| partners | 4,797 |
| services | 4,174 |
| health | 3,819 |
| clinic | 3,723 |
| solutions | 3,567 |
| global | 3,334 |
| trading | 3,118 |
| international | 2,843 |
| technologies | 2,796 |

