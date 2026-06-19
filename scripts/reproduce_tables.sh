#!/usr/bin/env bash
set -euo pipefail

python src/pipeline/run_minimal_baseline.py \
  --feature-csv data/sample/mms_ml_features_100k.csv \
  --sequence-csv data/sample/mms_sequence_windows.csv \
  --label-csv data/labels/mms_prepared_supervised_labels.csv \
  --scenario-summary-csv data/labels/mms_prepared_scenario_summary.csv \
  --output-dir results/expected_outputs/minimal_baseline

# Request-side-only vs. report-side-only vs. combined detection comparison.
# Derives the comparison from released full-capture artifacts (no heavy re-run).
python src/evaluation/compare_request_vs_report_detection.py \
  --alerts-csv data/raw/hybrid_ids_alerts_full_capture.csv \
  --attack-tags-csv data/raw/mms_capture_attack_tags.csv \
  --labels-csv data/labels/mms_full_capture_supervised_labels.csv \
  --output-md results/request_vs_report_comparison.md \
  --output-json results/request_vs_report_comparison.json
