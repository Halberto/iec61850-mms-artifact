# Dataset Description

The repository includes a Git-safe MMS sample subset in `data/sample/`, a normalized full-corpus capture in `data/raw/`, and reviewed labels in `data/labels/`.

## Full Corpus

The manuscript reports a full corpus of 428,204 MMS records with 105 malicious packets across the four evaluated scenario classes. The full corpus is provided in two compact files:

- `data/raw/mms_capture_normalized.csv.gz` (52 MB compressed) — primary analysis file
- `data/raw/mms_capture_normalized.jsonl.gz` (58 MB compressed) — same corpus in JSONL
- `data/raw/mms_capture_attack_tags.csv` (62 MB) — attack-focused 21-column view

The normalized files contain an enriched flat schema extracted from the raw MMS dissector output:
- Network fields: `frame_number`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `src_mac`, `dst_mac`, `ipv4_ttl`, `tcp_flags`, `tcp_window`
- Protocol fields: `direction`, `service`, `invoke_id`, `summary`, `raw_mms_hex`
- IEC 61850 control fields: `control_object`, `control_action`, `control_value`, `ctl_num`, `origin_identifier`, `origin_category`
- Report fields: `variable_list_name`, `access_result`, `octet_identities`
- Ground-truth label: `tag` (`attack` for IEDEXPLORER-origin packets, `normal` otherwise)

The attack tags file (`mms_capture_attack_tags.csv`) provides the same fields minus network/hex columns, with `access_result` cleared for measurement report noise patterns (`urcbMeasFlt`, `MEAS_RTU`, `RCB_RTU`, `URCB_1`, `brcbStatNrml`).

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
