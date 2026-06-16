# Data Layout

- `sample/`: Git-safe runnable sample files and small processed tables
- `raw/`: normalized full-corpus MMS capture files (compressed, labeled, deduplicated) plus supporting raw artifacts such as attack tags, alert exports, and analysis notes
  - `mms_capture_normalized.csv.gz` — primary dataset, all 428,204 records, enriched flat schema with network, protocol, IEC 61850 control fields, and ground-truth `tag` label (52 MB compressed)
  - `mms_capture_normalized.jsonl.gz` — same corpus in JSONL format (58 MB compressed)
  - `mms_capture_attack_tags.csv` — attack-focused 21-column view with `tag` (attack/normal), control fields, and decoded `access_result`; measurement report noise cleared
  - `attack_tags_full_capture.csv` — raw attack tag annotations (IEDEXPLORER-origin packets labeled `attack`)
  - `hybrid_ids_alerts_full_capture.csv` — IDS alert export
  - `analysis.txt` — capture-session analysis notes
- `labels/`: supervised labels, scenario summaries, and manual review exports. Join to the normalized corpus by `line_number`.
