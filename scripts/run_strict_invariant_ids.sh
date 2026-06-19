#!/usr/bin/env bash
set -euo pipefail

# Strict deterministic-invariant run of the hybrid MMS IDS on the full capture.
#
# Versus the default (online-learned) run, --strict-deterministic-invariants:
#   1. provisions the authorized-origin / ctlNum baseline from the trained model
#      (instead of learning it online), so the report-side invariants are armed
#      from the very first packet;
#   2. evaluates the protocol/deterministic rules from row 1 (no warm-up gate for
#      hard violations) -- recovers attacks that land inside the 5-min warm-up;
#   3. never suppresses hard-rule likely-attack alerts -- recovers attacks that the
#      time-based de-duplication would otherwise fold into a prior identical alert.
#
# The statistical branch is unchanged and still uses the warm-up window. Outputs go
# to results/strict/ so the released default-mode artifacts under data/raw/ are kept.
#
# The detector reads the gzipped capture directly (no manual decompression needed).

IN="data/raw/mms_capture_normalized.csv.gz"
OUT_DIR="results/strict"
ALERTS="$OUT_DIR/hybrid_ids_alerts_full_capture_strict.csv"
# The warm-up window contains no SCADA control WRITEs (they appear later in the
# capture), so re-training would yield known_origins=0. Load the pre-built baseline
# which has the correct known_origins provisioned.
PREBUILT_BASELINE="configs/experiments/hybrid_ids_baseline_full_capture.json"

mkdir -p "$OUT_DIR"

python src/baseline/hybrid_mms_ids.py \
  --input-csv "$IN" \
  --output-csv "$ALERTS" \
  --baseline-json "$OUT_DIR/hybrid_ids_baseline_full_capture_strict.json" \
  --load-baseline-json "$PREBUILT_BASELINE" \
  --strict-deterministic-invariants

# Recompute the request-side vs report-side vs combined comparison against the
# strict alerts. With the warm-up and de-duplication gaps closed, the report-side
# (and combined) recall should reach 105/105 with no change to request-side (0/105).
python src/evaluation/compare_request_vs_report_detection.py \
  --alerts-csv "$ALERTS" \
  --attack-tags-csv data/raw/mms_capture_attack_tags.csv \
  --labels-csv data/labels/mms_full_capture_supervised_labels.csv \
  --baseline-json "$BASELINE" \
  --output-md "$OUT_DIR/request_vs_report_comparison_strict.md" \
  --output-json "$OUT_DIR/request_vs_report_comparison_strict.json"

echo "[done] strict alerts: $ALERTS"
echo "[done] comparison:    $OUT_DIR/request_vs_report_comparison_strict.md"
