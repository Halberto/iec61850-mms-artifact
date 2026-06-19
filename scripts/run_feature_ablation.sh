#!/usr/bin/env bash
set -euo pipefail

# Report-side feature ablation: does surfacing report-side origin/ctlNum into the
# ML features materially improve detection? (Reviewer comment #2.)
#
# Design: build features and train the SAME model twice from the SAME input and the
# SAME hybrid context, differing only by one feature_synthesizer flag:
#   * request-side : default (origin/ctlNum features are request-side only)
#   * report-aware : --use-report-side-control (reports' origin/ctlNum, which the
#                    detector already emits as report_origins/report_ctl_nums, feed
#                    the origin/ctlNum features too)
# The recall delta between the two models is the report-side contribution.
#
# Heavy steps (hybrid pass + feature build over the 100k sample + two RandomForest
# fits) are left for you to run. The script is self-contained: it regenerates the
# hybrid-context file with --emit-all (feature_synthesizer inner-joins on line_number,
# so every row must be present, not just alerts).

SAMPLE_IN="data/sample/mms_sample_100k.csv"
LABELS="data/labels/mms_prepared_supervised_labels.csv"
OUT_DIR="results/ablation"
RESULTS_CSV="$OUT_DIR/hybrid_ids_alerts_sample_allrows.csv"
mkdir -p "$OUT_DIR"

FEAT_REQUEST="$OUT_DIR/features_request_side.csv"
FEAT_COMBINED="$OUT_DIR/features_report_aware.csv"

build_features () {  # $1=output features csv  $2...=extra flags
  out="$1"; shift
  python src/features/feature_synthesizer.py \
    --input-csv "$SAMPLE_IN" \
    --results-csv "$RESULTS_CSV" \
    --output-csv "$out" \
    --label-csv "$LABELS" \
    --label-key-column line_number \
    --feature-key-column line_number \
    --label-value-column supervised_is_anomaly "$@"
}

train () {  # $1=features csv  $2=report path  $3=model path
  python src/baseline/train_fusion_ml.py \
    --feature-csv "$1" \
    --report-path "$2" \
    --model-path "$3" \
    --synthetic-policy exclude
}

# 0. Hybrid context for every sample row (--emit-all so the inner-join keeps all rows).
python src/baseline/hybrid_mms_ids.py \
  --input-csv "$SAMPLE_IN" \
  --output-csv "$RESULTS_CSV" \
  --baseline-json "$OUT_DIR/hybrid_ids_baseline_sample.json" \
  --emit-all

# 1. Build both feature sets from the same input/context; only the flag differs.
build_features "$FEAT_REQUEST"
build_features "$FEAT_COMBINED" --use-report-side-control

# 3. Train the same model on each.
train "$FEAT_REQUEST"  "$OUT_DIR/report_request_side.txt" "$OUT_DIR/model_request_side.joblib"
train "$FEAT_COMBINED" "$OUT_DIR/report_report_aware.txt" "$OUT_DIR/model_report_aware.joblib"

echo
echo "[done] compare recall/F1 between:"
echo "  request-side : $OUT_DIR/report_request_side.txt"
echo "  report-aware : $OUT_DIR/report_report_aware.txt"
echo "The improvement in the report-aware model is the report-side feature contribution."
