# Limitations

This Git repository does not include the full unprocessed raw capture files, most large processed tables, or the full set of experiment checkpoints. Those artifacts are better published as release archives or DOI-backed deposits.

The normalized corpus files (`mms_capture_normalized.csv.gz`, `mms_capture_normalized.jsonl.gz`) are derived from the original raw capture by removing structural redundancy and merging reviewed labels. The following information from the original capture is not present in the normalized files:

- Full per-layer protocol dissection (TPKT, COTP, Session, Presentation layer detail)
- Response-side raw MMS hex bytes (`response_raw_mms_hex`)
- Response packet dissection (`response_dissection`)

These fields are available in the original raw capture CSV, which is retained as a private archive and can be regenerated using `src/utils/normalize_dataset.py` with different column selections if needed.

The repository is intended to be intentional, reviewable, and reproducible — not a mirror of the original working directory.

## Attack-class scarcity and augmentation

The captured attack subset is small by nature: **105 real malicious packets** (all IEDExplorer-origin IEC 61850 report PDUs), spanning **2 IEDs** and **7 controllable switch objects** (of which `Q8CSWI1` is the earth switch). This limits the statistical power of purely supervised detection trained on real positives alone.

To mitigate this without misrepresenting captured traffic, the repository provides a **generative augmentation framework** (`src/augmentation/`, documented in `dataset_description.md`) that studies the real attacks, fits a statistical model, and reproduces new attack packets to extend the corpus to a configurable ratio (default 40/60). Its design constraints are deliberately conservative:

- The model (`AttackGenerativeModel`) samples **only real, observed values** applied onto real attack scaffolds — no interpolation (SMOTE) or unconstrained generation (GAN), and no invented protocol values. Fidelity is quantified in `results/augmentation_fidelity.md`.
- Synthetic rows are always flagged (`event_source = "synthetic"`, with `gen_rule`/`synthetic_seed_line` provenance) and excluded from the training feature set.
- With `--synthetic-policy train-only`, validation and test sets contain **real packets only**, so reported detection metrics remain a faithful measure of generalization to real, held-out attacks.

Augmentation **does not change the real captured ground truth** — the original corpus and labels are untouched; the extended corpus is written to separate `*_augmented` files. Any claim built on augmented training should be reported alongside the non-augmented baseline (`--synthetic-policy exclude`).

**Inherent ceiling.** Because the generator only reuses values seen in 105 real packets (2 IEDs, 15 objects, 2 channels), synthetic diversity is bounded by that real vocabulary: oversampling to 40% multiplies *volume and class balance*, not the variety of attack behaviors. It is a mitigation for class imbalance, not a substitute for capturing more distinct real attacks.
