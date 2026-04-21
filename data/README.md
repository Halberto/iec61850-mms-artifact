# Data Layout

- `sample/`: Git-safe runnable sample files and small processed tables
- `raw/`: normalized full-corpus MMS capture files (compressed, labeled, deduplicated) plus supporting raw artifacts such as attack tags, alert exports, and analysis notes
  - `mms_capture_normalized.csv.gz` — primary dataset, all 428,204 records, flat schema with ground-truth labels (20 MB compressed)
  - `mms_capture_normalized.jsonl.gz` — same corpus in JSONL format (63 MB compressed)
  - `attack_tags_full_capture.csv` — raw attack tag annotations
  - `hybrid_ids_alerts_full_capture.csv` — IDS alert export
  - `analysis.txt` — capture-session analysis notes
- `labels/`: supervised labels, scenario summaries, and manual review exports. Join to the normalized corpus by `line_number`.
