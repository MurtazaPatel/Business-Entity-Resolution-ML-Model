# Step `p0_load`

- date: 2026-09-25
- runtime: 2.3s   peak RAM: 0.09 GB
- data-dir: `dataset`   sample: full

### train_source1

- shape: **(2,206,821, 4)**   size: 200.3 MB
- columns: `['entity_id', 'business_name', 'business_address', 'country']`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S1-925783039 | Orelee's Barbershop | 1795 Westchester Drive, High Point, NC | US |
| S1-773889195 | Prime Money | 17560 Ellis Road, Tahlequah, OK | US |
| S1-377745466 | B+ Retail Inc | 1712 Montebello Avenue, Phoenix, AZ | US |

### train_source2

- shape: **(5,034,616, 4)**   size: 466.6 MB
- columns: `['entity_id', 'business_name', 'business_address', 'country']`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S2-166376419 | राम मार्केटिंग प्राइवेट लिमिटेड | KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi | India |
| S2-764573417 | -- Holloway Peak Inc Seafood | 105 ELM ST, MORGANTON, NC | US |
| S2-639257739 | आदित्य प्रॉपर्टीज एलएलपी | G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh | India |

### train_source3

- shape: **(5,285,603, 4)**   size: 480.4 MB
- columns: `['entity_id', 'business_name', 'business_address', 'country']`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S3-202863386 | wilfordhancock.com | Mack Rd, Haltom City, Texas | US |
| S3-859268022 | International South Consultants Private Ltd |  | India |
| S3-22467283 | LLC Moncada Léarning Center | 5780 Fawn Ct, Fort Worth, Texas | US |

### train_ground_truth

- shape: **(2,206,821, 2)**   size: 121.1 MB
- columns: `['source1_entity_id', 'matched_entity_ids']`

| source1_entity_id | matched_entity_ids |
| --- | --- |
| S1-965667 | S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-86044… |
| S1-55344266 | S2-249013014,S2-197070651,S3-478195123,S3-384364074 |
| S1-343815751 | S2-790675320,S2-479876582,S3-878454467 |

### test_source1

- shape: **(1,732,544, 4)**   size: 166.9 MB
- columns: `['entity_id', 'business_name', 'business_address', 'country']`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S1-714132312 | Zephay Labs Inc | 2621 Cotten Road, Tyler, TX | US |
| S1-106407869 | Vision Partners Corp | IA, Iowa City, 1064 Newton Rd, Unit 11 | US |
| S1-156285671 | << Team Ecole | 175 Boulevard du Président Franklin Roosevelt, Bordeaux, No… | France |

### test_source2

- shape: **(4,887,273, 4)**   size: 485.9 MB
- columns: `['entity_id', 'business_name', 'business_address', 'country']`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S2-192345572 | Brahma Infosoft | COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka | India |
| S2-566025912 | Marina Ecole France Sarl | 63 R. DE DIEPPE, LILLE, Hauts-de-France | France |
| S2-158121477 | SCI Ptit Àmicale | 18 RUE JEN ZAY, Dunkerque, Nord | France |

### test_source3

- shape: **(5,082,316, 4)**   size: 482.6 MB
- columns: `['entity_id', 'business_name', 'business_address', 'country']`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S3-462677478 | मॉडर्न फाइनेंस | No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gu… | India |
| S3-374810425 | Shri Sai Infratech Co | 3/115, East Delhi, DL | India |
| S3-198586129 | Fractales Amis Groupe S.A.S | 23 Rue Icmre, La Teste-de-buch, Gironde | France |

