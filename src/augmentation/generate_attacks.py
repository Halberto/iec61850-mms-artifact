"""Reproduce stage: extend the captured corpus to a target attack ratio.

Loads the fitted ``AttackGenerativeModel`` and writes an *extended* normalized
corpus plus an aligned label file. Synthetic rows are always flagged
(``event_source = synthetic``) so they can be excluded from features and kept
out of validation/test downstream (see ``--synthetic-policy train-only`` in
``src/baseline/train_fusion_ml.py``).

Default regime (config ``normal_subsample: null``): keep every real normal row
and oversample synthetic attacks until attack >= ``target_attack_ratio`` of the
whole corpus -- i.e. literally extend the lab capture. Set ``normal_subsample``
to instead build a compact balanced training corpus.

Ratio math (A_real real attacks, N normals kept, S synthetic):
    (A_real + S) / (N + A_real + S) >= r  =>  S = ceil(r/(1-r) * N) - A_real
"""

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack_model import AttackGenerativeModel  # noqa: E402  (src/augmentation on path)

csv.field_size_limit(10**8)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "utils"))
from report_control_fields import enrich_report_row  # noqa: E402  (src/utils on path)
PROVENANCE_COLUMNS = ["event_source", "gen_rule", "synthetic_seed_line"]


def parse_args() -> argparse.Namespace:
    raw = os.path.join(REPO_ROOT, "data", "raw")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=os.path.join(raw, "mms_capture_normalized.csv.gz"))
    parser.add_argument("--profile-json", default=os.path.join(REPO_ROOT, "results", "attack_profile.json"))
    parser.add_argument("--templates-jsonl", default=os.path.join(REPO_ROOT, "results", "attack_templates.jsonl"))
    parser.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "experiments", "attack_augmentation.json"))
    parser.add_argument("--output-corpus", default=os.path.join(raw, "mms_capture_normalized_augmented.csv.gz"))
    parser.add_argument(
        "--output-labels",
        default=os.path.join(REPO_ROOT, "data", "labels", "mms_full_capture_supervised_labels_augmented.csv"),
    )
    parser.add_argument("--summary-json", default=os.path.join(REPO_ROOT, "results", "augmentation_summary.json"))
    parser.add_argument("--tag-column", default="tag")
    parser.add_argument("--attack-value", default="attack")
    # Optional overrides (useful for smoke tests; otherwise config drives the run).
    parser.add_argument("--target-attack-ratio", type=float, default=None)
    parser.add_argument("--normal-subsample", type=int, default=None)
    parser.add_argument("--max-synthetic", type=int, default=None, help="Hard cap on synthetic rows (smoke tests).")
    parser.add_argument("--dry-run", action="store_true", help="Report planned counts without writing the corpus.")
    return parser.parse_args()


def open_maybe_gzip(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def required_synthetic(ratio: float, normals: int, real_attacks: int) -> int:
    if ratio <= 0:
        return 0
    if ratio >= 1:
        raise ValueError("target_attack_ratio must be < 1.")
    needed = math.ceil(ratio / (1.0 - ratio) * normals) - real_attacks
    return max(0, needed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ratio = args.target_attack_ratio if args.target_attack_ratio is not None else float(config.get("target_attack_ratio", 0.40))
    normal_subsample = args.normal_subsample if args.normal_subsample is not None else config.get("normal_subsample")
    seed = int(config.get("random_seed", 0))
    rng = random.Random(seed)

    model = AttackGenerativeModel.from_files(args.profile_json, args.templates_jsonl)
    corpus_fields = model.profile.get("corpus_fields")
    if not corpus_fields:
        with open_maybe_gzip(args.corpus) as handle:
            corpus_fields = next(csv.reader(handle))
    out_fields = list(corpus_fields) + [c for c in PROVENANCE_COLUMNS if c not in corpus_fields]
    is_attack = lambda r: (r.get(args.tag_column) or "").strip().lower() == args.attack_value.lower()  # noqa: E731

    # ---- Pass 1: count normals / real attacks (and reservoir-sample normals
    #      when a subsample is requested). Attacks are always kept. ----
    real_attacks: list[dict] = []
    kept_normals: list[dict] | None = [] if normal_subsample is not None else None
    n_normal_total = 0
    seen_normal = 0
    with open_maybe_gzip(args.corpus) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if is_attack(row):
                real_attacks.append(row)
                continue
            n_normal_total += 1
            if normal_subsample is None:
                continue
            # Reservoir sampling to keep exactly `normal_subsample` normals.
            seen_normal += 1
            if len(kept_normals) < normal_subsample:
                kept_normals.append(row)
            else:
                j = rng.randint(0, seen_normal - 1)
                if j < normal_subsample:
                    kept_normals[j] = row

    n_normal_kept = n_normal_total if normal_subsample is None else len(kept_normals)
    n_real_attacks = len(real_attacks)
    synth_needed = required_synthetic(ratio, n_normal_kept, n_real_attacks)
    if args.max_synthetic is not None:
        synth_needed = min(synth_needed, args.max_synthetic)

    achieved = (n_real_attacks + synth_needed) / max(1, n_normal_kept + n_real_attacks + synth_needed)
    print(f"normals kept: {n_normal_kept} | real attacks: {n_real_attacks}")
    print(f"target ratio: {ratio:.3f} -> synthetic needed: {synth_needed} (achieved attack share {achieved:.3f})")

    summary = {
        "config": os.path.relpath(args.config, REPO_ROOT),
        "random_seed": seed,
        "target_attack_ratio": ratio,
        "normal_subsample": normal_subsample,
        "corpus_total_rows": model.profile.get("corpus_total_rows"),
        "normals_total": n_normal_total,
        "normals_kept": n_normal_kept,
        "real_attacks": n_real_attacks,
        "synthetic_generated": synth_needed,
        "extended_total_rows": n_normal_kept + n_real_attacks + synth_needed,
        "achieved_attack_share": round(achieved, 4),
    }
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Summary -> {args.summary_json}")

    if args.dry_run:
        print("Dry run: no corpus written.")
        return

    # ---- Pass 2: write the extended corpus + aligned labels. ----
    os.makedirs(os.path.dirname(args.output_corpus), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_labels), exist_ok=True)
    line_number = 0

    with open_maybe_gzip(args.output_corpus, "wt") as out_corpus, open(
        args.output_labels, "w", newline="", encoding="utf-8"
    ) as out_labels:
        corpus_writer = csv.DictWriter(out_corpus, fieldnames=out_fields, extrasaction="ignore")
        corpus_writer.writeheader()
        label_writer = csv.DictWriter(
            out_labels,
            fieldnames=["line_number", "timestamp", "final_tag", "seed_is_attack", "event_source"],
        )
        label_writer.writeheader()

        def emit(row: dict, event_source: str, gen_rule: str = "", seed_ref: str = "") -> None:
            nonlocal line_number
            line_number += 1
            row = dict(row)
            row["event_source"] = event_source
            row["gen_rule"] = gen_rule
            row["synthetic_seed_line"] = seed_ref
            # Surface report-side control fields into the flat columns. Synthetic
            # rows have a freshly sampled controlled object, so re-derive (overwrite);
            # real rows only get empty report columns filled (never request-side data).
            enrich_report_row(row, overwrite=(event_source == "synthetic"))
            corpus_writer.writerow(row)
            attack = is_attack(row)
            label_writer.writerow({
                "line_number": line_number,
                "timestamp": row.get("timestamp", ""),
                "final_tag": "attack" if attack else "normal",
                "seed_is_attack": 1 if attack else 0,
                "event_source": event_source,
            })

        if normal_subsample is None:
            # Stream the real corpus verbatim (preserves chronological order).
            with open_maybe_gzip(args.corpus) as handle:
                for row in csv.DictReader(handle):
                    emit(row, "packet")
        else:
            real = kept_normals + real_attacks
            real.sort(key=lambda r: r.get("timestamp", ""))
            for row in real:
                emit(row, "packet")

        # Append synthetic attacks.
        for index, row in enumerate(model.sample(synth_needed, rng)):
            emit(row, "synthetic", gen_rule="model", seed_ref=row.get("stream_id", ""))

    print(f"Extended corpus -> {args.output_corpus}")
    print(f"Labels -> {args.output_labels}")
    print(f"Wrote {line_number} rows total ({synth_needed} synthetic).")


if __name__ == "__main__":
    main()
