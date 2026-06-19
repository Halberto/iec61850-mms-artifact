# Augmentation fidelity report

Real attack packets: 105 · synthetic sampled: 5000

Chi-square goodness-of-fit of synthetic marginals vs the real-attack PMF (lower normalized statistic = closer; coverage check flags any value the model emitted that never occurred in real data).

| field | real categories | synthetic categories | out-of-vocab | chi2/df |
|---|---|---|---|---|
| src_ip | 2 | 2 | NONE | 1.217 |
| stream_id | 2 | 2 | NONE | 1.217 |
| controllable_object | 8 | 8 | NONE | 3.738 |

- Fingerprint constants identical on all synthetic rows: **True**
- No out-of-vocabulary values in any checked field: **True**

_Faithful by construction: the model samples only real, observed values and applies them onto real attack scaffolds._
