#!/usr/bin/env bash
set -euo pipefail

python src/pipeline/run_minimal_baseline.py \
  --feature-csv data/sample/mms_ml_features_100k.csv \
  --sequence-csv data/sample/mms_sequence_windows.csv \
  --label-csv data/labels/mms_prepared_supervised_labels.csv \
  --scenario-summary-csv data/labels/mms_prepared_scenario_summary.csv \
  --output-dir results/expected_outputs/minimal_baseline
