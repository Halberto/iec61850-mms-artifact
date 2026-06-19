# Artifact Overview

This release contains the baseline implementation, feature-building code, reviewed labels and summaries, a sample dataset, expected outputs for sanity checking, and the normalized full-corpus MMS capture.

The full 428,204-record corpus is provided as two compact gzip-compressed files in `data/raw/`:

- `mms_capture_normalized.csv.gz` (52 MB) — enriched flat CSV with network, protocol, and IEC 61850 control fields; ground-truth `tag` label (`attack`/`normal`) based on IEDEXPLORER-origin detection
- `mms_capture_normalized.jsonl.gz` (58 MB) — same corpus in JSONL format
- `mms_capture_attack_tags.csv` (62 MB) — attack-focused 21-column view with control and report fields; measurement report noise cleared from `access_result`

These files were produced from the raw MMS dissector output, extracting and flattening all relevant network, protocol, and IEC 61850 fields. Attack packets (105 records) are identified by IEDEXPLORER as the origin identifier in the decoded MMS report (`octet_identities = IEDEXPLORER`).

The anonymous archive at https://anonymous.4open.science/r/iec61850-mms-artifact-7B05/ remains useful as an alternate delivery channel for very large assets and for omitted full-capture processed tables listed in `data/metadata/release_inventory.csv`.

This repository does not include the live acquisition and protocol-dissection tooling used to build the full corpus from the laboratory traffic stream.
