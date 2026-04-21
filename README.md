# From MMS Command Misuse to Process Disruption

This repository accompanies the paper "From MMS Command Misuse to Process Disruption: Threat Modeling and Experimental Analysis in IEC 61850 Digital Substations". It contains the released baseline IDS, labels, sample data, and the restored full raw CSV and JSONL capture exports used for the paper-scale corpus.

Anonymous review archive: https://anonymous.4open.science/r/iec61850-mms-artifact-7B05/

## Scope

This Git repository contains the baseline IDS implementation, reviewed labels, sample data, and the normalized full-corpus MMS capture. The full 428,204-record corpus has been deduplicated and normalized into compact gzip-compressed files — all structural redundancy removed, ground-truth labels merged in. The sample workflow assets remain usable for lightweight validation.

## Included In This Git Repository

- MMS-aware baseline implementation under `src/baseline/`
- Feature generation and sequence-window construction under `src/features/`
- Minimal orchestration under `src/pipeline/`
- Evaluation and reporting scripts under `src/evaluation/`
- Dataset normalization utility under `src/utils/normalize_dataset.py`
- Baseline configuration files under `configs/`
- A Git-safe sample dataset under `data/sample/`
- Normalized full-corpus capture files under `data/raw/`:
  - `mms_capture_normalized.csv.gz` — all 428,204 MMS records, flat+labeled (20 MB)
  - `mms_capture_normalized.jsonl.gz` — same corpus in JSONL format (63 MB)
- Supporting raw artifacts under `data/raw/`: attack tags, IDS alerts, analysis notes
- Reviewed labels and scenario summaries under `data/labels/`
- Expected output files under `results/expected_outputs/`
- Release manifest for omitted large artifacts under `data/metadata/release_inventory.csv`

## Still Better Served Via Release Assets Or Archives

- Large full-capture processed tables
- Additional model checkpoints and experiment artifacts omitted from the Git subset
- The original unprocessed raw capture CSV (kept as a private archive)

- Local environment folders and editor settings
- The live capture and protocol-dissection tooling used to build the full corpus from the lab capture stream

See `docs/artifact_overview.md`, `docs/dataset_description.md`, and `docs/reproduction.md` for the artifact boundary and reproduction path.
