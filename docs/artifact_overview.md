# Artifact Overview

This release contains the baseline implementation, feature-building code, reviewed labels and summaries, a sample dataset, expected outputs for sanity checking, and the normalized full-corpus MMS capture.

The full 428,204-record corpus is provided as two compact gzip-compressed files in `data/raw/`:

- `mms_capture_normalized.csv.gz` (20 MB) — flat, deduplicated CSV with all network, protocol, and IEC 61850 control fields plus ground-truth labels merged in by `line_number`
- `mms_capture_normalized.jsonl.gz` (63 MB) — same corpus in JSONL format

These files were produced by `src/utils/normalize_dataset.py`, which strips structural redundancy from the original raw capture (nested `dissection` objects, typed `control_parameters`, duplicate fields) and merges in the reviewed labels from `data/labels/mms_full_capture_supervised_labels.csv`.

The anonymous archive at https://anonymous.4open.science/r/iec61850-mms-artifact-7B05/ remains useful as an alternate delivery channel for very large assets and for omitted full-capture processed tables listed in `data/metadata/release_inventory.csv`.

This repository does not include the live acquisition and protocol-dissection tooling used to build the full corpus from the laboratory traffic stream.
