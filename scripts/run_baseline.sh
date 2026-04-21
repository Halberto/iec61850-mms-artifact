#!/usr/bin/env bash
set -euo pipefail

python src/baseline/hybrid_mms_ids.py \
  --input-csv data/sample/mms_sample_100k.csv \
  --output-csv results/expected_outputs/hybrid_ids_alerts_sample.csv \
  --baseline-json configs/baseline/hybrid_ids_baseline.json
