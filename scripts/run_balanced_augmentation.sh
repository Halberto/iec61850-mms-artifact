#!/usr/bin/env bash
# Balanced MMS augmentation: generate both normal and attack traffic from
# empirical protocol, timing, and class-specific distributions.
#
# Usage:
#   bash scripts/run_balanced_augmentation.sh [scale] [attack_ratio] [output]
#
# Example:
#   bash scripts/run_balanced_augmentation.sh 2 0.40 data/raw/mms_capture_balanced_augmented.csv.gz
set -euo pipefail

SCALE="${1:-2}"
RATIO="${2:-0.40}"
OUTPUT="${3:-data/raw/mms_capture_balanced_augmented.csv.gz}"
SEED="${SEED:-42}"

python src/augmentation/mms_dataset_augmentor.py \
  --input data/raw/mms_capture_attack_tags.csv \
  --output "$OUTPUT" \
  --scale "$SCALE" \
  --ratio "$RATIO" \
  --seed "$SEED"

echo "Done. Balanced augmented corpus: $OUTPUT"
