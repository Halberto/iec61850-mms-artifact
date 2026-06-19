# Dataset Description

The repository includes a Git-safe MMS sample subset in `data/sample/`, a normalized full-corpus capture in `data/raw/`, and reviewed labels in `data/labels/`.

## Full Corpus

The manuscript reports a full corpus of 428,204 MMS records with 105 malicious packets across the four evaluated scenario classes. The full corpus is provided in two compact files:

- `data/raw/mms_capture_normalized.csv.gz` (52 MB compressed) — primary analysis file
- `data/raw/mms_capture_normalized.jsonl.gz` (58 MB compressed) — same corpus in JSONL
- `data/raw/mms_capture_attack_tags.csv` (62 MB) — attack-focused 21-column view

The normalized files contain an enriched flat schema extracted from the raw MMS dissector output:
- Network fields: `frame_number`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `src_mac`, `dst_mac`, `ipv4_ttl`, `tcp_flags`, `tcp_window`
- Protocol fields: `direction`, `service`, `invoke_id`, `summary`, `raw_mms_hex`
- IEC 61850 control fields: `control_object`, `control_action`, `control_value`, `ctl_num`, `origin_identifier`, `origin_category`
- Report fields: `variable_list_name`, `access_result`, `octet_identities`
- Ground-truth label: `tag` (`attack` for IEDEXPLORER-origin packets, `normal` otherwise)

The attack tags file (`mms_capture_attack_tags.csv`) provides the same fields minus network/hex columns, with `access_result` cleared for measurement report noise patterns (`urcbMeasFlt`, `MEAS_RTU`, `RCB_RTU`, `URCB_1`, `brcbStatNrml`).

The original unprocessed raw capture CSV is retained as a private archive and is not included in this repository.

## Labels

Ground-truth labels are in `data/labels/mms_full_capture_supervised_labels.csv`. Key label columns:

| Column | Description |
|---|---|
| `final_tag` | `normal` or `attack` |
| `seed_is_attack` | 1 if this record is a direct attack PDU |
| `supervised_is_anomaly` | 1 if flagged as anomalous in the reviewed scenario |
| `scenario_id` | Identifier for the attack scenario this record belongs to |
| `scenario_role` | `seed`, `context`, or `normal` |

Join key: `line_number` (1-based row index, identical between the normalized files and the labels CSV).

## Attack-Class Augmentation (generative)

Captured malicious traffic is intrinsically scarce: the corpus holds **105 real IEDExplorer report packets** across 2 IEDs and 15 controllable switch/interlock objects. To support class-balanced training, the repository ships a **generative augmentation framework** that *studies* the real attacks, *fits* a statistical model to them, and *reproduces* new attack packets to extend the corpus to a configurable attack ratio (default **40/60 attack/normal**).

Pipeline (`src/augmentation/`, orchestrated by `scripts/run_attack_augmentation.sh`):

1. **Study** — `profile_attacks.py` reads the corpus and learns, from the 105 real attacks: the constant fingerprint fields (`RESPONSE/UNCONFIRMED/RPT/IEDEXPLORER`, dst `10.0.19.39`); the `src_ip` / IED / report-member PMFs; per-IED controllable-object vocabulary split by CSWI vs CILO; a first-order Markov chain over object tokens; per-stream report-sequence increments; and per-stream inter-arrival timing. Outputs `results/attack_profile.json`, `attack_templates.jsonl`, and a human-readable `attack_profile_report.md`.
2. **Model** — `attack_model.py` defines `AttackGenerativeModel`, a **conditional-empirical + Markov sampler**. It draws a real attack row as a structural scaffold (guaranteeing a valid URCB report), resamples *which* objects the report carries (preserving the CSWI/`$ST$Pos` vs CILO/`$ST$EnaCls` relationship), and advances monotone per-stream sequence counters and clocks. It samples **only real, observed values** — no interpolation (unlike SMOTE) and no unconstrained generator (unlike a GAN), which is the defensible choice at n=105.
3. **Reproduce** — `generate_attacks.py` samples the model to hit `target_attack_ratio`, writing an extended corpus `data/raw/mms_capture_normalized_augmented.csv.gz`, an aligned label file, and `results/augmentation_summary.json`. Default `normal_subsample: null` keeps **all** real normals and oversamples synthetic attacks (~285k) to reach 40%; setting `normal_subsample` builds a compact balanced training corpus instead.
4. **Validate** — `validate_augmentation.py` compares synthetic vs real per-field marginals (chi-square + out-of-vocabulary check) → `results/augmentation_fidelity.md`.

**Integrity guarantees.**
- Every synthetic row carries `event_source = "synthetic"` (+ `gen_rule`, `synthetic_seed_line`); real rows are `event_source = "packet"`.
- Provenance is passed through the feature pipeline as **metadata, excluded from the training feature set** (see `DEFAULT_METADATA_COLUMNS` in `src/baseline/train_fusion_ml.py`), so a model cannot key on it.
- Training with `--synthetic-policy train-only` computes the split on **real packets only** and confines synthetic rows to the training fold, so reported validation/test metrics measure detection of *real, held-out* attacks. `--synthetic-policy exclude` reproduces the non-augmented baseline.
- Fidelity is quantified, not assumed: the model emits no out-of-vocabulary protocol values.

## Sample Subset

The sample files in `data/sample/` remain useful for lightweight validation and can be used without the full corpus.

Large full-capture processed tables remain omitted from the repository. See `data/metadata/release_inventory.csv` for the manifest.
