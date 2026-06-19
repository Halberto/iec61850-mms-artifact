# Request-side vs. report-side MMS detection comparison

Derived directly from released artifacts by `src/evaluation/compare_request_vs_report_detection.py`. Compares the deterministic protocol-rule families emitted by the hybrid baseline; the orthogonal statistical branch is excluded by design.

## Inputs

- Alerts: `data/raw/hybrid_ids_alerts_full_capture.csv`
- Attack ground truth: `data/raw/mms_capture_attack_tags.csv` (105 attack packets)
- Scenario-window labels: `data/labels/mms_full_capture_supervised_labels.csv` (core=192, core+context=1312)

## Ground-truth composition (why request-side detection is structurally blind)

- Attack packets: **105**
- Direction: RESPONSE=105
- Service: UNCONFIRMED=105
- Attack-labelled Write **Requests** on the monitored stream: **0**. Every attack packet is an unconfirmed Information Report, because the adversary's writes originate from a separate off-stream `IEDEXPLORER` association. A request-side-only detector therefore has no attack request to inspect.

## Headline comparison

| Detector configuration | Attack packets detected | Recall | Total alert rows | Alerts in clean traffic |
|---|---|---|---|---|
| Request-side rules only | 0/105 | 0.0% | 10 | 0 |
| Report-side rules only (proposed) | 101/105 | 96.2% | 175 | 0 |
| Combined (request + report) | 101/105 | 96.2% | 185 | 0 |

*Alerts in clean traffic* = alert rows falling outside every labelled attack scenario window (the operationally meaningful false-alarm count). It is 0 for all three configurations: neither family fires in clean supervisory traffic. The entire difference between the configurations is in **detections**, not false alarms.

## Per-rule recall over the attack packets

| Rule | Family | Attacks fired on |
|---|---|---|
| `report_origin_not_seen_in_writes` | report | 101/105 |
| `report_without_matching_write` | report | 101/105 |
| `unexpected_octet_identity_in_control_context` | report | 101/105 |
| `report_ctlnum_regression` | report | 70/105 |

## What the request-side alerts actually are

The request-side family produced 10 alert row(s), of which **0 are true attack packets**. They all fire `sbow_without_matching_oper` on legitimate Select-Before-Operate requests whose paired Operate was not observed within the matching window -- a benign supervisory-workflow artifact in the surrounding context traffic, not the attack.

## Misses and honesty notes

- Report-side recall: 101/105. Missed attack lines: [422, 433, 503, 165404].
- The shortfall from 100% is **not** a report-side blind spot; each missed packet is accounted for by the detector's own design:
  - **Warm-up window (3)**: lines [422, 433, 503] fall in the first-5-minutes unlabelled training prefix the detector uses to seed its baselines; rows there are not scored by design (these are the earliest attack packets).
  - **Alert de-duplication (1)**: lines [165404] were detected but collapsed by alert suppression -- an identical-fingerprint attack alert (same report origin/`ctlNum`/reasons) fired within the suppression window seconds earlier, so the duplicate is not re-emitted. Visible with `--emit-all` or a shorter suppression window.

## Interpretation

Request-side-only MMS detection is not merely weaker on this corpus -- it is structurally blind: it records zero true detections, because the attack is observable solely through Information Reports. The report-side family flags 96.2% of attack packets without raising a single alert in clean traffic; the remaining 3.8% are not detection failures but warm-up exclusion and alert de-duplication (see above), so every attack campaign is in fact surfaced. Adding the request-side family to the report-side family changes neither recall nor the clean-traffic alert count, confirming that the report-side features are the ones that materially drive detection.
