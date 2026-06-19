# Request-side vs. report-side MMS detection comparison

Derived directly from released artifacts by `src/evaluation/compare_request_vs_report_detection.py`. Compares the deterministic protocol-rule families emitted by the hybrid baseline; the orthogonal statistical branch is excluded by design.

## Inputs

- Alerts: `results/strict/hybrid_ids_alerts_full_capture_strict.csv`
- Attack ground truth: `data/raw/mms_capture_attack_tags.csv` (105 attack packets)
- Scenario-window labels: `data/labels/mms_full_capture_supervised_labels.csv` (core=192, core+context=1312)

## Ground-truth composition (why request-side detection is structurally blind)

- Attack packets: **105**
- Direction: RESPONSE=105
- Service: UNCONFIRMED=105
- Attack-labelled Write **Requests** in the analyzed station-bus capture/export: **0**. Every labeled attack packet is an unconfirmed Information Report; therefore, request-side rules have no attack-labeled Write Request to inspect in the exported records.

## Headline comparison

| Detector configuration | Attack packets detected | Recall | Total alert rows | Alerts in clean traffic |
|---|---|---|---|---|
| Request-side rules only | 0/105 | 0.0% | 0 | 0 |
| Report-side rules only (proposed) | 105/105 | 100.0% | 376 | 1 |
| Combined (request + report) | 105/105 | 100.0% | 376 | 1 |

*Alerts in clean traffic* = alert rows falling outside every labelled attack scenario window. The request-side configuration raises no clean-traffic alerts, while the report-side and combined configurations each raise one. The main difference between the configurations is detection coverage: request-side rules detect 0/105 attack packets, whereas report-side rules detect 105/105.

## Per-rule recall over the attack packets

| Rule | Family | Attacks fired on |
|---|---|---|
| `report_origin_not_seen_in_writes` | report | 105/105 |
| `report_without_matching_write` | report | 105/105 |
| `unexpected_octet_identity_in_control_context` | report | 105/105 |
| `report_ctlnum_regression` | report | 70/105 |

## What the request-side alerts actually are

The request-side family produced 0 alert rows in the strict configuration and detected 0/105 attack packets. This supports the visibility argument: the evaluated attacks are represented in the exported ground truth as unconfirmed Information Reports, not as attack-labelled Write Requests.

## Misses and honesty notes

- Report-side recall: 105/105. Missed attack lines: none.

## Interpretation

Request-side rules detect 0/105 attack packets because no attack-labelled Write Requests are present in the analyzed station-bus export. Report-side rules detect 105/105 packets using unmatched reports, `origin.orIdent`, and `ctlNum`, with one clean-traffic alert; there are no misses. The combined configuration does not improve recall over report-side rules alone, showing that report-side features materially drive detection in this cross-session attack setting.
