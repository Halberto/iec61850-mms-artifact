import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import train_fusion_ml as fusion_train
import train_sequence_branch as sequence_train
import mms_deterministic_checker as det_checker


BASE_PATH = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE_CSV = str(BASE_PATH / "data" / "raw" / "mms_capture_normalized.csv.gz")
DEFAULT_FEATURE_CSV = str(BASE_PATH / "data" / "sample" / "mms_ml_features_100k.csv")
DEFAULT_SEQUENCE_CSV = str(BASE_PATH / "data" / "sample" / "mms_sequence_windows.csv")
DEFAULT_LABEL_CSV = str(BASE_PATH / "data" / "labels" / "mms_full_capture_supervised_labels.csv")
DEFAULT_SCENARIO_SUMMARY_CSV = str(BASE_PATH / "data" / "labels" / "mms_full_capture_scenario_summary.csv")
DEFAULT_RESULTS_CSV = str(BASE_PATH / "data" / "raw" / "hybrid_ids_alerts_full_capture.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the minimal MMS IDS baseline workflow: GRU sequence branch, optional rules-only "
            "baseline, and a scenario review export."
        )
    )
    parser.add_argument("--capture-csv", default=DEFAULT_CAPTURE_CSV)
    parser.add_argument("--feature-csv", default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--sequence-csv", default=DEFAULT_SEQUENCE_CSV)
    parser.add_argument("--label-csv", default=DEFAULT_LABEL_CSV)
    parser.add_argument(
        "--scenario-summary-csv",
        default=DEFAULT_SCENARIO_SUMMARY_CSV,
    )
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV)
    parser.add_argument(
        "--output-dir",
        default=str(BASE_PATH / "results" / "minimal_baseline_full_capture"),
    )
    parser.add_argument("--label-column", default="supervised_is_anomaly")
    parser.add_argument("--split-mode", choices=("scenario", "time", "group"), default="scenario")
    parser.add_argument(
        "--threshold-objective",
        choices=("row_f1", "scenario_f1"),
        default="row_f1",
    )
    parser.add_argument("--model-type", choices=("gru", "tcn", "rf"), default="gru")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-rules", action="store_true", help="Skip the rules-only reference baseline.")
    parser.add_argument(
        "--run-strict-rules",
        action="store_true",
        help="Run an additional deterministic strict-rules reference branch (no ML threshold fitting).",
    )
    parser.add_argument(
        "--strict-protocol-threshold",
        type=float,
        default=45.0,
        help="Protocol score threshold used by strict deterministic rules.",
    )
    parser.add_argument(
        "--strict-score-threshold",
        type=float,
        default=1.0,
        help="Decision threshold applied to strict_rule_score.",
    )
    parser.add_argument(
        "--strict-enable-sequence",
        action="store_true",
        help="Enable sequence-based strict rule contribution (disabled by default due high false-positive risk).",
    )
    parser.add_argument(
        "--strict-sequence-requires-protocol",
        action="store_true",
        help="Require protocol_score >= strict-protocol-threshold for sequence discrepancy to contribute.",
    )
    parser.add_argument(
        "--rebuild-derived-inputs",
        action="store_true",
        help="Force rebuilding feature and sequence CSVs before training.",
    )
    # ------------------------------------------------------------------ #
    # Deterministic protocol state-machine branch                         #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--run-state-machine",
        action="store_true",
        help=(
            "Run the deterministic MMS protocol state-machine branch. "
            "Hard protocol invariants are evaluated first (no ML, no threshold fitting). "
            "Scenario-level escalation is applied so that any scenario containing a "
            "hard-rule hit flags all of its member rows."
        ),
    )
    parser.add_argument(
        "--sm-no-moderate",
        action="store_true",
        help="Exclude Tier-2 moderate violations (last_appl_error) from the state-machine final prediction.",
    )
    return parser.parse_args()


def is_default_path(actual_path: str, default_path: str) -> bool:
    return os.path.abspath(actual_path) == os.path.abspath(default_path)


def should_rebuild_inputs(args: argparse.Namespace) -> bool:
    return (
        args.rebuild_derived_inputs
        or not is_default_path(args.label_csv, DEFAULT_LABEL_CSV)
        or not is_default_path(args.scenario_summary_csv, DEFAULT_SCENARIO_SUMMARY_CSV)
    )


def rebuild_inputs(args: argparse.Namespace) -> tuple[str, str]:
    rebuilt_dir = os.path.join(args.output_dir, "rebuilt_inputs")
    os.makedirs(rebuilt_dir, exist_ok=True)
    rebuilt_feature_csv = os.path.join(rebuilt_dir, "mms_ml_features_rebuilt.csv")
    rebuilt_sequence_csv = os.path.join(rebuilt_dir, "mms_sequence_windows_rebuilt.csv")

    feature_command = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "feature_synthesizer.py"),
        "--input-csv",
        args.capture_csv,
        "--results-csv",
        args.results_csv,
        "--output-csv",
        rebuilt_feature_csv,
        "--label-csv",
        args.label_csv,
        "--label-key-column",
        "line_number",
        "--feature-key-column",
        "line_number",
        "--label-value-column",
        args.label_column,
        "--label-output-column",
        args.label_column,
    ]
    print("\nRebuilding feature CSV from the provided labels...")
    subprocess.run(feature_command, check=True, cwd=os.path.dirname(__file__))

    sequence_command = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "build_sequence_windows.py"),
        "--feature-csv",
        rebuilt_feature_csv,
        "--output-csv",
        rebuilt_sequence_csv,
        "--label-column",
        args.label_column,
    ]
    print("\nRebuilding sequence windows from the rebuilt feature CSV...")
    subprocess.run(sequence_command, check=True, cwd=os.path.dirname(__file__))

    return rebuilt_feature_csv, rebuilt_sequence_csv


def build_rules_only_drop_columns(feature_csv: str, label_column: str) -> str:
    columns = pd.read_csv(feature_csv, nrows=0).columns.tolist()
    keep = set(fusion_train.DEFAULT_EXPERT_COLUMNS)
    drops = [
        column
        for column in columns
        if column not in keep
        and column != label_column
        and column not in fusion_train.DEFAULT_METADATA_COLUMNS
        and column not in fusion_train.DEFAULT_SCENARIO_COLUMNS
    ]
    return ",".join(sorted(set(drops)))


def metric_or_none(metrics: dict, *keys: str) -> float | None:
    current = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return None if current is None else float(current)


def build_sequence_args(args: argparse.Namespace, output_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        sequence_csv=args.sequence_csv,
        model_path=os.path.join(output_dir, "gru_baseline_model.joblib"),
        report_path=os.path.join(output_dir, "gru_baseline_report.txt"),
        predictions_csv=os.path.join(output_dir, "gru_baseline_predictions.csv"),
        label_column=args.label_column,
        model_type=args.model_type,
        split_mode=args.split_mode,
        time_column="window_end_time_unix",
        group_column="window_group",
        scenario_column="scenario_id",
        threshold_objective=args.threshold_objective,
        test_size=0.2,
        val_size=0.2,
        random_state=args.random_state,
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        epochs=args.epochs,
        batch_size=512,
        learning_rate=1e-3,
        weight_decay=1e-4,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        patience=3,
        tcn_kernel_size=3,
        device="auto",
    )


def build_rules_args(args: argparse.Namespace, output_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        feature_csv=args.feature_csv,
        model_path=os.path.join(output_dir, "rules_only_model.joblib"),
        report_path=os.path.join(output_dir, "rules_only_report.txt"),
        label_column=args.label_column,
        split_mode=args.split_mode,
        group_columns="src_ip_cat,dst_ip_cat",
        time_column="event_time_unix",
        scenario_column="scenario_id",
        threshold_objective=args.threshold_objective,
        test_size=0.2,
        val_size=0.2,
        random_state=args.random_state,
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        include_expert_features=True,
        keep_identity_features=False,
        extra_drop_columns=build_rules_only_drop_columns(args.feature_csv, args.label_column),
    )


def _numeric_series(df: pd.DataFrame, column: str, default_value: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default_value)
    return pd.Series(default_value, index=df.index, dtype="float64")


def _strict_rule_flags(
    df: pd.DataFrame,
    protocol_threshold: float = 45.0,
    score_threshold: float = 1.0,
    enable_sequence: bool = False,
    sequence_requires_protocol: bool = False,
) -> pd.DataFrame:
    has_report_seq = (_numeric_series(df, "has_report_seq_num") > 0)
    has_ctl_num = (_numeric_series(df, "has_ctl_num") > 0)

    report_seq_regression = (_numeric_series(df, "report_seq_regression_stream") > 0) & has_report_seq
    report_seq_reused = (_numeric_series(df, "report_seq_reused_stream") > 0) & has_report_seq
    report_seq_delta_negative = (_numeric_series(df, "report_seq_delta_stream") < 0) & has_report_seq
    sequence_raw = report_seq_regression | report_seq_reused | report_seq_delta_negative

    ctl_regression = (_numeric_series(df, "ctl_num_regression_actor_object") > 0) & has_ctl_num
    ctl_reused = (_numeric_series(df, "ctl_num_reused_actor_object") > 0) & has_ctl_num
    ctl_delta_negative = (_numeric_series(df, "ctl_num_delta_actor_object") < 0) & has_ctl_num
    ctl_any = ctl_regression | ctl_reused | ctl_delta_negative

    origin_new_pair = (_numeric_series(df, "origin_seen_with_new_src") > 0)
    source_new_origin = (_numeric_series(df, "source_seen_with_new_origin") > 0)
    origin_any = origin_new_pair | source_new_origin

    protocol_high = (_numeric_series(df, "protocol_score") >= float(protocol_threshold))

    sequence_contributing = sequence_raw.copy()
    if sequence_requires_protocol:
        sequence_contributing = sequence_contributing & protocol_high
    if not enable_sequence:
        sequence_contributing = pd.Series(False, index=df.index)

    out = pd.DataFrame(
        {
            "strict_origin_mismatch": origin_any.astype("int8"),
            "strict_sequence_discrepancy_raw": sequence_raw.astype("int8"),
            "strict_sequence_discrepancy": sequence_contributing.astype("int8"),
            "strict_ctl_discrepancy": ctl_any.astype("int8"),
            "strict_protocol_high": protocol_high.astype("int8"),
        },
        index=df.index,
    )

    # Weighted deterministic score:
    # - high-confidence conditions (origin/ctl/protocol): +1 each
    # - optional sequence contribution: +0.25 (kept low due known noise on this dataset)
    out["strict_rule_score"] = (
        out["strict_origin_mismatch"].astype("float64")
        + out["strict_ctl_discrepancy"].astype("float64")
        + out["strict_protocol_high"].astype("float64")
        + 0.25 * out["strict_sequence_discrepancy"].astype("float64")
    )
    out["strict_rule_prediction"] = (out["strict_rule_score"] >= float(score_threshold)).astype("int8")
    return out


def run_state_machine_reference(args: argparse.Namespace, output_dir: str) -> tuple[SimpleNamespace, dict]:
    """Invoke the deterministic MMS protocol state-machine branch."""
    sm_ns = SimpleNamespace(
        feature_csv=args.feature_csv,
        results_csv=args.results_csv,
        label_column=args.label_column,
        split_mode=args.split_mode,
        scenario_column="scenario_id",
        time_column="event_time_unix",
        group_columns="src_ip_cat,dst_ip_cat",
        test_size=0.2,
        val_size=0.2,
        random_state=args.random_state,
        sm_include_moderate=not bool(getattr(args, "sm_no_moderate", False)),
    )
    sm_output_dir = os.path.join(output_dir, "state_machine")
    os.makedirs(sm_output_dir, exist_ok=True)
    return det_checker.run_state_machine_branch(sm_ns, sm_output_dir)


def run_strict_rules_reference(args: argparse.Namespace, output_dir: str) -> tuple[SimpleNamespace, dict]:
    strict_args = SimpleNamespace(
        feature_csv=args.feature_csv,
        report_path=os.path.join(output_dir, "strict_rules_report.txt"),
        predictions_csv=os.path.join(output_dir, "strict_rules_predictions.csv"),
        split_mode=args.split_mode,
        label_column=args.label_column,
        group_columns="src_ip_cat,dst_ip_cat",
        time_column="event_time_unix",
        scenario_column="scenario_id",
        test_size=0.2,
        val_size=0.2,
        random_state=args.random_state,
        threshold=float(args.strict_score_threshold),
        protocol_threshold=float(args.strict_protocol_threshold),
        enable_sequence=bool(args.strict_enable_sequence),
        sequence_requires_protocol=bool(args.strict_sequence_requires_protocol),
    )

    print(f"Loading strict-rules reference features from {strict_args.feature_csv}")
    df = pd.read_csv(strict_args.feature_csv, low_memory=False)
    resolved_label_column = fusion_train.resolve_label_column(df, strict_args.label_column)
    df = df.dropna(subset=[resolved_label_column]).copy()
    df[resolved_label_column] = df[resolved_label_column].astype(int)
    y = df[resolved_label_column]

    rule_flags = _strict_rule_flags(
        df,
        protocol_threshold=strict_args.protocol_threshold,
        score_threshold=strict_args.threshold,
        enable_sequence=strict_args.enable_sequence,
        sequence_requires_protocol=strict_args.sequence_requires_protocol,
    )
    scores = rule_flags["strict_rule_score"].to_numpy(dtype="float64")
    split_summary: dict[str, object]

    if strict_args.split_mode == "group":
        group_columns = fusion_train.parse_column_list(strict_args.group_columns)
        if not group_columns:
            raise ValueError("Group split requires at least one group column.")
        group_ids = fusion_train.build_group_ids(df, group_columns)
        train_full_idx, test_idx, group_df = fusion_train.stratified_group_split(
            y=y,
            group_ids=group_ids,
            test_size=strict_args.test_size,
            random_state=strict_args.random_state,
        )
        train_group_ids = group_ids.iloc[train_full_idx].reset_index(drop=True)
        y_train_full = y.iloc[train_full_idx].reset_index(drop=True)
        train_inner_idx, val_inner_idx, _ = fusion_train.stratified_group_split(
            y=y_train_full,
            group_ids=train_group_ids,
            test_size=strict_args.val_size,
            random_state=strict_args.random_state + 1,
        )
        train_idx = train_full_idx[train_inner_idx]
        val_idx = train_full_idx[val_inner_idx]
        split_summary = {
            "group_columns": group_columns,
            "unique_groups": int(len(group_df)),
            "positive_groups": int(group_df["group_label"].sum()),
        }
    elif strict_args.split_mode == "scenario":
        if strict_args.scenario_column not in df.columns:
            raise ValueError(
                f"Scenario split requested but scenario column '{strict_args.scenario_column}' was not found."
            )
        scenario_ids = fusion_train.build_scenario_group_ids(df, strict_args.scenario_column)
        train_full_idx, test_idx, scenario_df = fusion_train.stratified_group_split(
            y=y,
            group_ids=scenario_ids,
            test_size=strict_args.test_size,
            random_state=strict_args.random_state,
        )
        train_scenario_ids = scenario_ids.iloc[train_full_idx].reset_index(drop=True)
        y_train_full = y.iloc[train_full_idx].reset_index(drop=True)
        train_inner_idx, val_inner_idx, _ = fusion_train.stratified_group_split(
            y=y_train_full,
            group_ids=train_scenario_ids,
            test_size=strict_args.val_size,
            random_state=strict_args.random_state + 1,
        )
        train_idx = train_full_idx[train_inner_idx]
        val_idx = train_full_idx[val_inner_idx]
        split_summary = {
            "scenario_column": strict_args.scenario_column,
            "unique_scenarios": int(len(scenario_df)),
            "positive_scenarios": int(scenario_df["group_label"].sum()),
        }
    else:
        time_values = fusion_train.build_time_order(df, strict_args.time_column)
        train_idx, val_idx, test_idx = fusion_train.chronological_split(
            y=y,
            time_values=time_values,
            test_size=strict_args.test_size,
            val_size=strict_args.val_size,
        )
        split_summary = {
            "time_column": strict_args.time_column,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        }

    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    val_scores = scores[val_idx]
    test_scores = scores[test_idx]

    val_metrics = fusion_train.evaluate_model(y_val, val_scores, strict_args.threshold)
    test_metrics = fusion_train.evaluate_model(y_test, test_scores, strict_args.threshold)

    if strict_args.scenario_column in df.columns:
        val_group_ids = fusion_train.build_scenario_group_ids(
            df.iloc[val_idx].reset_index(drop=True),
            strict_args.scenario_column,
        )
        test_group_ids = fusion_train.build_scenario_group_ids(
            df.iloc[test_idx].reset_index(drop=True),
            strict_args.scenario_column,
        )
        val_metrics["scenario_metrics"] = fusion_train.evaluate_grouped_predictions(
            y_true=y_val,
            scores=val_scores,
            threshold=strict_args.threshold,
            group_ids=val_group_ids,
            group_label="scenario",
        )
        test_metrics["scenario_metrics"] = fusion_train.evaluate_grouped_predictions(
            y_true=y_test,
            scores=test_scores,
            threshold=strict_args.threshold,
            group_ids=test_group_ids,
            group_label="scenario",
        )

    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_flags = rule_flags.iloc[test_idx].reset_index(drop=True)
    prediction_columns = ["line_number", strict_args.scenario_column, strict_args.time_column]
    prediction_columns = [column for column in prediction_columns if column in test_df.columns]
    predictions_df = test_df[prediction_columns].copy()
    predictions_df[resolved_label_column] = y_test.to_numpy()
    predictions_df = pd.concat(
        [
            predictions_df,
            test_flags[
                [
                    "strict_origin_mismatch",
                    "strict_protocol_high",
                    "strict_sequence_discrepancy_raw",
                    "strict_sequence_discrepancy",
                    "strict_ctl_discrepancy",
                    "strict_rule_score",
                    "strict_rule_prediction",
                ]
            ],
        ],
        axis=1,
    )
    predictions_df.to_csv(strict_args.predictions_csv, index=False)

    rules_available = {
        "origin_seen_with_new_src": "origin_seen_with_new_src" in df.columns,
        "source_seen_with_new_origin": "source_seen_with_new_origin" in df.columns,
        "protocol_score": "protocol_score" in df.columns,
        "has_report_seq_num": "has_report_seq_num" in df.columns,
        "report_seq_regression_stream": "report_seq_regression_stream" in df.columns,
        "report_seq_reused_stream": "report_seq_reused_stream" in df.columns,
        "report_seq_delta_stream": "report_seq_delta_stream" in df.columns,
        "has_ctl_num": "has_ctl_num" in df.columns,
        "ctl_num_regression_actor_object": "ctl_num_regression_actor_object" in df.columns,
        "ctl_num_reused_actor_object": "ctl_num_reused_actor_object" in df.columns,
        "ctl_num_delta_actor_object": "ctl_num_delta_actor_object" in df.columns,
    }
    bundle = {
        "threshold": strict_args.threshold,
        "protocol_threshold": strict_args.protocol_threshold,
        "enable_sequence": strict_args.enable_sequence,
        "sequence_requires_protocol": strict_args.sequence_requires_protocol,
        "metrics": {"validation": val_metrics, "test": test_metrics},
        "split_mode": strict_args.split_mode,
        "split_summary": split_summary,
        "label_column": resolved_label_column,
        "rules_available": rules_available,
        "predictions_csv": strict_args.predictions_csv,
    }

    lines = [
        "Strict Deterministic Rules Reference",
        "====================================",
        "",
        f"Feature CSV: {strict_args.feature_csv}",
        f"Resolved label column: {resolved_label_column}",
        f"Rows: {len(df)}",
        f"Positive rows: {int(y.sum())} ({y.mean() * 100:.4f}%)",
        f"Split mode: {strict_args.split_mode}",
        f"Split summary: {json.dumps(split_summary)}",
        f"Protocol threshold: {strict_args.protocol_threshold}",
        f"Sequence enabled: {strict_args.enable_sequence}",
        f"Sequence requires protocol threshold: {strict_args.sequence_requires_protocol}",
        f"Threshold: {strict_args.threshold}",
        "",
        "Rule predicates (deterministic weighted score):",
        "- strict_origin_mismatch = origin_seen_with_new_src OR source_seen_with_new_origin",
        "- strict_ctl_discrepancy = ctl_num_regression_actor_object OR ctl_num_reused_actor_object OR negative ctl_num_delta_actor_object (all gated by has_ctl_num)",
        "- strict_protocol_high = protocol_score >= protocol_threshold",
        "- strict_sequence_discrepancy_raw = report_seq_regression_stream OR report_seq_reused_stream OR negative report_seq_delta_stream (gated by has_report_seq_num)",
        "- strict_sequence_discrepancy contributes only when --strict-enable-sequence is on; optionally also requires strict_protocol_high",
        "- strict_rule_score = origin + ctl + protocol + 0.25 * sequence_contributing",
        "- predict attack when strict_rule_score >= strict-score-threshold",
        "",
        f"Rule columns available: {json.dumps(rules_available)}",
        "",
        "Validation metrics:",
        json.dumps({k: v for k, v in val_metrics.items() if k != 'classification_report'}, indent=2),
        "",
        "Validation classification report:",
        val_metrics["classification_report"],
        "",
        "Validation scenario metrics:",
        json.dumps(val_metrics.get("scenario_metrics", {}), indent=2),
        "",
        "Test metrics:",
        json.dumps({k: v for k, v in test_metrics.items() if k != 'classification_report'}, indent=2),
        "",
        "Test classification report:",
        test_metrics["classification_report"],
        "",
        "Test scenario metrics:",
        json.dumps(test_metrics.get("scenario_metrics", {}), indent=2),
        "",
        f"Predictions CSV: {strict_args.predictions_csv}",
    ]
    with open(strict_args.report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Strict rules report saved to {strict_args.report_path}")
    print(f"Strict rules predictions saved to {strict_args.predictions_csv}")

    return strict_args, bundle


def run_scenario_review(args: argparse.Namespace, output_dir: str, sequence_predictions_csv: str) -> None:
    review_dir = os.path.join(output_dir, "scenario_review")
    os.makedirs(review_dir, exist_ok=True)
    command = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "export_scenario_review.py"),
        "--scenario-summary-csv",
        args.scenario_summary_csv,
        "--label-csv",
        args.label_csv,
        "--results-csv",
        args.results_csv,
        "--meta-predictions-csv",
        "",
        "--sequence-predictions-csv",
        sequence_predictions_csv,
        "--output-dir",
        review_dir,
    ]
    subprocess.run(command, check=True, cwd=os.path.dirname(__file__))


def build_summary_rows(
    sequence_bundle: dict,
    rules_bundle: dict | None,
    strict_rules_bundle: dict | None,
    state_machine_bundle: dict | None = None,
) -> list[dict]:
    rows = [
        {
            "branch": "sequence_branch",
            "experiment": f"minimal_{sequence_bundle.get('model_type', 'gru')}",
            "test_f1": float(sequence_bundle["metrics"]["test"]["f1"]),
            "test_scenario_f1": metric_or_none(
                sequence_bundle["metrics"]["test"],
                "scenario_metrics",
                "scenario_f1",
            ),
            "test_average_precision": float(sequence_bundle["metrics"]["test"]["average_precision"]),
            "threshold": float(sequence_bundle["threshold"]),
        }
    ]
    if rules_bundle is not None:
        rows.append(
            {
                "branch": "rules_branch",
                "experiment": "minimal_rules_only",
                "test_f1": float(rules_bundle["metrics"]["test"]["f1"]),
                "test_scenario_f1": metric_or_none(
                    rules_bundle["metrics"]["test"],
                    "scenario_metrics",
                    "scenario_f1",
                ),
                "test_average_precision": float(rules_bundle["metrics"]["test"]["average_precision"]),
                "threshold": float(rules_bundle["threshold"]),
            }
        )
    if strict_rules_bundle is not None:
        rows.append(
            {
                "branch": "strict_rules_branch",
                "experiment": "minimal_strict_deterministic_rules",
                "test_f1": float(strict_rules_bundle["metrics"]["test"]["f1"]),
                "test_scenario_f1": metric_or_none(
                    strict_rules_bundle["metrics"]["test"],
                    "scenario_metrics",
                    "scenario_f1",
                ),
                "test_average_precision": float(
                    strict_rules_bundle["metrics"]["test"]["average_precision"]
                ),
                "threshold": float(strict_rules_bundle["threshold"]),
            }
        )
    if state_machine_bundle is not None:
        rows.append(
            {
                "branch": "state_machine_branch",
                "experiment": "deterministic_protocol_state_machine",
                "test_f1": float(state_machine_bundle["metrics"]["test"]["f1"]),
                "test_scenario_f1": metric_or_none(
                    state_machine_bundle["metrics"]["test"],
                    "scenario_metrics",
                    "scenario_f1",
                ),
                "test_average_precision": float(
                    state_machine_bundle["metrics"]["test"]["average_precision"]
                ),
                "threshold": float(state_machine_bundle.get("threshold", 0.5)),
            }
        )
    return rows


def write_summary(
    args: argparse.Namespace,
    output_dir: str,
    sequence_args: SimpleNamespace,
    sequence_bundle: dict,
    rules_args: SimpleNamespace | None,
    rules_bundle: dict | None,
    strict_rules_args: SimpleNamespace | None,
    strict_rules_bundle: dict | None,
    state_machine_args: SimpleNamespace | None = None,
    state_machine_bundle: dict | None = None,
) -> None:
    rows = build_summary_rows(sequence_bundle, rules_bundle, strict_rules_bundle, state_machine_bundle)
    summary_df = pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)
    summary_csv = os.path.join(output_dir, "minimal_branch_summary.csv")
    summary_txt = os.path.join(output_dir, "minimal_baseline_summary.txt")
    summary_df.to_csv(summary_csv, index=False)

    sequence_test = sequence_bundle["metrics"]["test"]
    lines = [
        "Minimal MMS IDS Baseline",
        "========================",
        "",
        "This workflow intentionally keeps only the core pieces:",
        "- GRU sequence branch",
        "- optional rules-only reference baseline",
        "- scenario review export for manual label cleanup",
        "",
        f"Feature CSV: {args.feature_csv}",
        f"Sequence CSV: {args.sequence_csv}",
        f"Label CSV: {args.label_csv}",
        f"Scenario summary CSV: {args.scenario_summary_csv}",
        f"Split mode: {args.split_mode}",
        f"Threshold objective: {args.threshold_objective}",
        "",
        "GRU baseline:",
        f"Model path: {sequence_args.model_path}",
        f"Predictions CSV: {sequence_args.predictions_csv}",
        f"Test F1: {sequence_test['f1']:.4f}",
        f"Test scenario F1: {metric_or_none(sequence_test, 'scenario_metrics', 'scenario_f1')}",
        f"Test average precision: {sequence_test['average_precision']:.4f}",
        f"Threshold: {sequence_bundle['threshold']:.6f}",
    ]

    if rules_bundle is not None and rules_args is not None:
        rules_test = rules_bundle["metrics"]["test"]
        lines.extend(
            [
                "",
                "Rules-only reference:",
                f"Model path: {rules_args.model_path}",
                f"Test F1: {rules_test['f1']:.4f}",
                f"Test scenario F1: {metric_or_none(rules_test, 'scenario_metrics', 'scenario_f1')}",
                f"Test average precision: {rules_test['average_precision']:.4f}",
                f"Threshold: {rules_bundle['threshold']:.6f}",
            ]
        )

    if strict_rules_bundle is not None and strict_rules_args is not None:
        strict_test = strict_rules_bundle["metrics"]["test"]
        lines.extend(
            [
                "",
                "Strict deterministic rules reference:",
                f"Report path: {strict_rules_args.report_path}",
                f"Predictions CSV: {strict_rules_args.predictions_csv}",
                f"Test F1: {strict_test['f1']:.4f}",
                f"Test scenario F1: {metric_or_none(strict_test, 'scenario_metrics', 'scenario_f1')}",
                f"Test average precision: {strict_test['average_precision']:.4f}",
                f"Score threshold: {strict_rules_bundle['threshold']:.6f}",
                f"Protocol threshold: {strict_rules_bundle.get('protocol_threshold')}",
                f"Sequence enabled: {strict_rules_bundle.get('enable_sequence')}",
                f"Sequence requires protocol: {strict_rules_bundle.get('sequence_requires_protocol')}",
            ]
        )

    if state_machine_bundle is not None and state_machine_args is not None:
        sm_test = state_machine_bundle["metrics"]["test"]
        cov = state_machine_bundle.get("full_dataset_coverage", {})
        lines.extend(
            [
                "",
                "Deterministic protocol state-machine branch:",
                f"Report path: {state_machine_args.report_path}",
                f"Predictions CSV: {state_machine_args.predictions_csv}",
                f"Test F1: {sm_test['f1']:.4f}",
                f"Test scenario F1: {metric_or_none(sm_test, 'scenario_metrics', 'scenario_f1')}",
                f"Test average precision: {sm_test['average_precision']:.4f}",
                f"Hard-rule row hits (full dataset): {cov.get('hard_rule_row_hits', 'N/A')}",
                f"Hard-rule precision (full dataset): {cov.get('hard_rule_precision', float('nan')):.4f}",
                f"Hard-rule recall of positives (full dataset): {cov.get('hard_rule_recall_of_positives', float('nan')):.4f}",
                f"Scenario-escalated rows (full dataset): {cov.get('scenario_escalated_rows', 'N/A')}",
                f"Include moderate violations: {state_machine_bundle.get('include_moderate')}",
            ]
        )

    lines.extend(
        [
            "",
            "Scenario review:",
            f"Review CSV: {os.path.join(output_dir, 'scenario_review', 'scenario_review.csv')}",
            f"Event review CSV: {os.path.join(output_dir, 'scenario_review', 'scenario_event_review.csv')}",
            "",
            "Recommended next move:",
            "Use the scenario review CSV to trim false-positive scenarios, then rerun this same minimal baseline.",
            "",
            "Branch summary:",
            summary_df.to_string(index=False),
        ]
    )

    with open(summary_txt, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"Saved minimal branch summary to {summary_csv}")
    print(f"Saved minimal baseline summary to {summary_txt}")
    print(summary_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if should_rebuild_inputs(args):
        rebuilt_feature_csv, rebuilt_sequence_csv = rebuild_inputs(args)
        args.feature_csv = rebuilt_feature_csv
        args.sequence_csv = rebuilt_sequence_csv

    print("Training minimal GRU baseline...")
    sequence_args = build_sequence_args(args, args.output_dir)
    sequence_bundle = sequence_train.train_sequence_branch(sequence_args)

    rules_args = None
    rules_bundle = None
    strict_rules_args = None
    strict_rules_bundle = None
    state_machine_args = None
    state_machine_bundle = None
    if not args.skip_rules:
        print("\nTraining rules-only reference baseline...")
        rules_args = build_rules_args(args, args.output_dir)
        rules_bundle = fusion_train.train_fusion_model(rules_args)
    if args.run_strict_rules:
        print("\nRunning strict deterministic-rules reference baseline...")
        strict_rules_args, strict_rules_bundle = run_strict_rules_reference(args, args.output_dir)
    if getattr(args, "run_state_machine", False):
        print("\nRunning deterministic protocol state-machine branch...")
        state_machine_args, state_machine_bundle = run_state_machine_reference(args, args.output_dir)

    print("\nExporting scenario review CSVs...")
    run_scenario_review(args, args.output_dir, sequence_args.predictions_csv)

    print("\nWriting compact baseline summary...")
    write_summary(
        args,
        args.output_dir,
        sequence_args,
        sequence_bundle,
        rules_args,
        rules_bundle,
        strict_rules_args,
        strict_rules_bundle,
        state_machine_args,
        state_machine_bundle,
    )


if __name__ == "__main__":
    main()
