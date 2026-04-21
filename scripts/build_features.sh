#!/usr/bin/env bash
set -euo pipefail

python src/features/feature_synthesizer.py \
  --input-csv data/sample/mms_sample_100k.csv \
  --results-csv results/expected_outputs/hybrid_ids_alerts_sample.csv \
  --output-csv data/sample/mms_ml_features_100k_rebuilt.csv \
  --label-csv data/labels/mms_prepared_supervised_labels.csv \
  --label-key-column line_number \
  --feature-key-column line_number \
  --label-value-column supervised_is_anomaly

python src/features/build_sequence_windows.py \
  --feature-csv data/sample/mms_ml_features_100k_rebuilt.csv \
  --output-csv data/sample/mms_sequence_windows_rebuilt.csv
