#!/usr/bin/env bash
# Generative attack-augmentation pipeline: study -> validate -> reproduce.
#
# Stages 1-2 are light (read the 105 real attacks, fit + check the model).
# Stage 3 is heavy: it extends the full ~428k-row capture to the configured
# attack ratio (default 40/60), writing ~285k synthetic attack rows.
#
# Tip: preview the planned counts first without writing anything:
#   python src/augmentation/generate_attacks.py --dry-run
set -euo pipefail

# 1. Study the real attacks -> results/attack_profile.json (+ templates + report).
python src/augmentation/profile_attacks.py \
  --corpus data/raw/mms_capture_normalized.csv.gz

# 2. Fit + validate the generative model against the real distributions.
python src/augmentation/validate_augmentation.py

# 3. Reproduce: extend the corpus to the target ratio (driven by the config).
python src/augmentation/generate_attacks.py \
  --corpus data/raw/mms_capture_normalized.csv.gz \
  --config configs/experiments/attack_augmentation.json \
  --output-corpus data/raw/mms_capture_normalized_augmented.csv.gz \
  --output-labels data/labels/mms_full_capture_supervised_labels_augmented.csv \
  --summary-json results/augmentation_summary.json

echo "Done. Extended corpus: data/raw/mms_capture_normalized_augmented.csv.gz"
echo "Summary: results/augmentation_summary.json | Fidelity: results/augmentation_fidelity.md"
echo
echo "When training on features built from the extended corpus, pass"
echo "  --synthetic-policy train-only"
echo "to src/baseline/train_fusion_ml.py so validation/test stay real-only."
