# Attack-packet study (generative-model inputs)

Real attack packets profiled: **105**

## Constant fingerprint fields

| field | value |
|---|---|
| `tag` | `attack` |
| `direction` | `RESPONSE` |
| `service` | `UNCONFIRMED` |
| `variable_list_name` | `RPT` |
| `octet_identities` | `IEDEXPLORER` |
| `dst_ip` | `10.0.19.39` |
| `origin_category` | `(empty)` |

## Categorical distributions

- `src_ip`: {'10.0.19.49': 55, '10.0.19.47': 50}
- IED: {'E03A103_CTRL': 55, 'E01A103_CTRL': 50}
- report members per packet: {'1': 79, '3': 4, '4': 5, '2': 17}
- distinct controllable objects: **15** across 2 IEDs

### Per-IED controllable objects

- **E03A103_CTRL** — CSWI: {'Q1CSWI1': 27, 'Q0CSWI1': 22, 'Q2CSWI1': 6} · CILO: {'Q0CILO1': 9, 'Q8CILO1': 5, 'Q2CILO1': 4, 'Q1CILO1': 4}
- **E01A103_CTRL** — CSWI: {'Q0CSWI1': 21, 'Q1CSWI1': 18, 'Q8CSWI1': 5, 'Q2CSWI1': 6} · CILO: {'Q1CILO1': 6, 'Q0CILO1': 7, 'Q2CILO1': 3, 'Q8CILO1': 2}

## Per-stream report sequence

- `10.0.19.39:56638-10.0.19.49:102`: start 51..241, increment PMF {3: 5, 1: 26, 2: 12, 18: 1, 8: 1, 4: 3, 35: 1, 5: 2, 9: 1, 22: 1, 11: 1}
- `10.0.19.39:56633-10.0.19.47:102`: start 78..211, increment PMF {0: 1, 3: 7, 1: 22, 2: 13, 14: 2, 18: 1, 9: 1, 4: 1, 5: 1}

## Per-stream timing

- `10.0.19.39:56638-10.0.19.49:102`: median gap 18.213s, mean 3454.814s, n_gaps 54
- `10.0.19.39:56633-10.0.19.47:102`: median gap 16.605s, mean 2106.385s, n_gaps 49

## Model note

Synthetic attacks are produced by `src/augmentation/attack_model.py`, which samples these distributions and applies them onto real attack scaffolds (preserving protocol structure and the CSWI/`$ST$Pos` vs CILO/`$ST$EnaCls` relationship). See `docs/dataset_description.md`.
