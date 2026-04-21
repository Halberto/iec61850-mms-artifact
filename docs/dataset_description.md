# Dataset Description

The repository includes a Git-safe MMS sample subset in `data/sample/`, a normalized full-corpus capture in `data/raw/`, and reviewed labels in `data/labels/`.

## Full Corpus

The manuscript reports a full corpus of 428,204 MMS records with 105 malicious packets across the four evaluated scenario classes. The full corpus is provided in two compact files:

- `data/raw/mms_capture_normalized.csv.gz` (20 MB compressed) — primary analysis file
- `data/raw/mms_capture_normalized.jsonl.gz` (63 MB compressed) — same corpus in JSONL

Both files are produced by `src/utils/normalize_dataset.py` from the original raw capture. The normalization:
- Removes the nested `dissection` object (which duplicated `src_ip`, `dst_ip`, `direction`, `timestamp`, etc.) and extracts only the unique fields it contained: MAC addresses, PRP sequence info, TCP seq/ack/flags/window, MMS payload size
- Flattens typed `control_parameters` objects into direct scalars (`ctl_num`, `ctl_timestamp`, `origin_identifier`, `ctl_test`, `ctl_check`)
- Merges ground-truth labels from `data/labels/mms_full_capture_supervised_labels.csv` by `line_number`
- Drops inaccurate embedded label placeholders from the original capture

The original unprocessed raw capture CSV is retained as a private archive and is not included in this repository.

## Labels

Ground-truth labels are in `data/labels/mms_full_capture_supervised_labels.csv`. Key label columns:

| Column | Description |
|---|---|
| `final_tag` | `normal` or `attack` |
| `seed_is_attack` | 1 if this record is a direct attack PDU |
| `supervised_is_anomaly` | 1 if flagged as anomalous in the reviewed scenario |
| `scenario_id` | Identifier for the attack scenario this record belongs to |
| `scenario_role` | `seed`, `context`, or `normal` |

Join key: `line_number` (1-based row index, identical between the normalized files and the labels CSV).

## Sample Subset

The sample files in `data/sample/` remain useful for lightweight validation and can be used without the full corpus.

Large full-capture processed tables remain omitted from the repository. See `data/metadata/release_inventory.csv` for the manifest.
