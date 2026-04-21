import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_FEATURES = [
    "service_cat",
    "pdu_type_cat",
    "origin_identifier_cat",
    "control_object_cat",
    "control_action_cat",
    "actor_id_cat",
    "source_origin_pair_cat",
    "prev_service_cat",
    "service_transition_cat",
    "prev_action_same_object_cat",
    "action_transition_cat",
    "is_request",
    "is_response",
    "is_control_event",
    "is_write_service",
    "is_report_event",
    "has_origin_identifier",
    "has_control_object",
    "has_control_action",
    "has_invoke_id",
    "has_ctl_num",
    "has_report_seq_num",
    "source_origin_consistent",
    "origin_seen_with_new_src",
    "object_seen_from_new_actor",
    "actor_changed_for_object",
    "invoke_id_reused_stream",
    "ctl_num_reused_actor_object",
    "ctl_num_regression_actor_object",
    "report_seq_reused_stream",
    "report_seq_regression_stream",
    "time_since_last_stream_event",
    "time_since_last_actor_object",
    "ctl_num_val",
    "ctl_num_delta_actor_object",
    "report_seq_num_val",
    "report_seq_delta_stream",
    "actor_object_legitimacy_ratio",
    "actor_action_legitimacy_ratio",
    "origin_object_legitimacy_ratio",
    "actor_object_action_ratio",
]
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build flattened per-stream sequence windows from the semantic MMS feature CSV."
    )
    parser.add_argument("--feature-csv", default=str(PROCESSED_DATA_DIR / "mms_ml_features_100k.csv"))
    parser.add_argument("--output-csv", default=str(PROCESSED_DATA_DIR / "mms_sequence_windows.csv"))
    parser.add_argument("--group-column", default="stream_id_cat")
    parser.add_argument(
        "--label-column",
        default="supervised_is_anomaly",
        help="Preferred label column. Falls back to is_anomaly when needed.",
    )
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--min-events-per-group", type=int, default=8)
    parser.add_argument("--label-mode", choices=("end", "any"), default="end")
    parser.add_argument("--time-column", default="event_time_unix")
    parser.add_argument("--line-column", default="line_number")
    parser.add_argument(
        "--passthrough-columns",
        default=(
            "scenario_id,scenario_role,scenario_seed_count,scenario_member_count,scenario_group_key,"
            "scenario_window_start_line,scenario_window_end_line,scenario_core_start_line,"
            "scenario_core_end_line,scenario_start_timestamp,scenario_end_timestamp,"
            "scenario_duration_seconds"
        ),
        help="Comma-separated end-event metadata columns to copy onto each sequence window.",
    )
    parser.add_argument(
        "--sequence-features",
        default=",".join(DEFAULT_SEQUENCE_FEATURES),
        help="Comma-separated event-level features to flatten into each sequence window.",
    )
    return parser.parse_args()


def parse_column_list(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_label_column(df: pd.DataFrame, requested_label: str, feature_csv: str) -> str:
    if requested_label in df.columns:
        return requested_label
    if requested_label != "is_anomaly" and "is_anomaly" in df.columns:
        print(f"Requested label column '{requested_label}' was not found. Falling back to 'is_anomaly'.")
        return "is_anomaly"
    raise ValueError(f"Label column '{requested_label}' not found in {feature_csv}.")


def resolve_target(window_labels: pd.Series, label_mode: str) -> int | None:
    if label_mode == "end":
        end_value = window_labels.iloc[-1]
        if pd.isna(end_value):
            return None
        return int(end_value)

    if window_labels.isna().any():
        return None
    return int(window_labels.max())


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.feature_csv, low_memory=False)
    resolved_label_column = resolve_label_column(df, args.label_column, args.feature_csv)
    required_columns = {args.group_column, resolved_label_column, args.time_column}
    missing_required = [column for column in required_columns if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in {args.feature_csv}: {missing_required}")

    sequence_features = parse_column_list(args.sequence_features)
    available_sequence_features = [column for column in sequence_features if column in df.columns]
    missing_sequence_features = [column for column in sequence_features if column not in df.columns]
    passthrough_columns = [column for column in parse_column_list(args.passthrough_columns) if column in df.columns]
    if not available_sequence_features:
        raise ValueError("None of the requested sequence features are present in the feature CSV.")

    if missing_sequence_features:
        print(f"Skipping missing sequence features: {missing_sequence_features}")

    sort_columns = [args.group_column, args.time_column]
    if args.line_column in df.columns:
        sort_columns.append(args.line_column)
    df = df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    window_rows: list[dict] = []
    for group_value, group_df in df.groupby(args.group_column, sort=False):
        if len(group_df) < max(args.window_size, args.min_events_per_group):
            continue

        group_df = group_df.reset_index(drop=True)
        for end_idx in range(args.window_size - 1, len(group_df), args.step_size):
            window_df = group_df.iloc[end_idx - args.window_size + 1 : end_idx + 1]
            target = resolve_target(window_df[resolved_label_column], args.label_mode)
            if target is None:
                continue

            row = {
                "window_group": group_value,
                "window_start_index": int(window_df.index[0]),
                "window_end_index": int(window_df.index[-1]),
                "window_size": args.window_size,
                resolved_label_column: target,
                "window_positive_count": int(window_df[resolved_label_column].fillna(0).sum()),
            }
            if args.line_column in window_df.columns:
                row["window_start_line_number"] = int(window_df.iloc[0][args.line_column])
                row["window_end_line_number"] = int(window_df.iloc[-1][args.line_column])
            if "event_timestamp" in window_df.columns:
                row["window_end_timestamp"] = window_df.iloc[-1]["event_timestamp"]
            if args.time_column in window_df.columns:
                row["window_end_time_unix"] = window_df.iloc[-1][args.time_column]
            for column in passthrough_columns:
                row[column] = window_df.iloc[-1][column]

            for offset, (_, event) in enumerate(window_df.iterrows()):
                steps_back = args.window_size - 1 - offset
                suffix = f"t_minus_{steps_back}" if steps_back > 0 else "t0"
                for feature_name in available_sequence_features:
                    row[f"{feature_name}_{suffix}"] = event[feature_name]

            window_rows.append(row)

    sequence_df = pd.DataFrame(window_rows)
    if sequence_df.empty:
        raise ValueError("No sequence windows were produced. Check group sizes and label coverage.")

    sequence_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(sequence_df)} sequence windows to {args.output_csv}")
    print(
        sequence_df[
            ["window_size", resolved_label_column, "window_positive_count"]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
