# Reproduction

This artifact supports two levels of reproduction:

- Sample workflow reproduction from the Git repository alone
- Paper-scale reproduction using the normalized full-corpus files in this repository

## Sample Workflow

Recommended quick-start commands:

```bash
pip install -r requirements.txt
bash scripts/run_baseline.sh
bash scripts/build_features.sh
bash scripts/reproduce_tables.sh
```

These commands validate the shipped sample subset under `data/sample/`, the reviewed labels under `data/labels/`, and the expected-output checks under `results/expected_outputs/`.

## Paper-Scale Reproduction

To reproduce the larger corpus statistics and full-capture analyses reported in Sections 6 and 7 of the paper, use the normalized full-corpus files:

- `data/raw/mms_capture_normalized.csv.gz` — all 428,204 records, flat schema, ground-truth labels included
- `data/raw/mms_capture_normalized.jsonl.gz` — same corpus in JSONL format

Load with pandas:

```python
import pandas as pd
df = pd.read_csv("data/raw/mms_capture_normalized.csv.gz")
```

To regenerate the normalized files from a new raw capture:

```bash
python src/utils/normalize_dataset.py \
  --input-csv <path-to-raw-capture.csv> \
  --labels-csv data/labels/mms_full_capture_supervised_labels.csv
```

If you prefer not to work from the Git working tree, you can obtain the anonymous archive:

https://anonymous.4open.science/r/iec61850-mms-artifact-7B05/

Use `data/metadata/release_inventory.csv` as the manifest for large files. The repository is sufficient for inspecting the normalized corpus and baseline workflow, but does not contain the live acquisition and protocol-dissection tooling used to build the full corpus from laboratory traffic.
