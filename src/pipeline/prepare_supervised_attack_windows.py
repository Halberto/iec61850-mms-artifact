import argparse
import os

import pandas as pd


DEFAULT_POSITIVE_TAGS = {
    "failed-control",
    "likely-attack",
    "attack",
    "anomaly",
    "masquerade",
    "malicious",
}


def parse_args() -> argparse.Namespace:
    base_path = r"C:\Users\herme\OneDrive\Documents\MMS Paper\dataset\dataset"
    parser = argparse.ArgumentParser(
        description=(
            "Convert sparse event labels into cleaner supervised attack windows with scenario ids. "
            "Positive seed events are grouped into scenarios and expanded with configurable context."
        )
    )
    parser.add_argument("--input-label-csv", default=os.path.join(base_path, "mms_sample_attack_tags.csv"))
    parser.add_argument("--output-label-csv", default=os.path.join(base_path, "mms_prepared_supervised_labels.csv"))
    parser.add_argument(
        "--scenario-summary-csv",
        default=os.path.join(base_path, "mms_prepared_scenario_summary.csv"),
        help="Optional scenario-level summary output.",
    )
    parser.add_argument(
        "--positive-tags",
        default="failed-control,likely-attack,attack,anomaly,masquerade,malicious",
    )
    parser.add_argument("--group-columns", default="stream_id")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--line-column", default="line_number")
    parser.add_argument("--tag-column", default="tag")
    parser.add_argument("--gap-seconds", type=float, default=30.0)
    parser.add_argument("--pre-context-events", type=int, default=4)
    parser.add_argument("--post-context-events", type=int, default=4)
    parser.add_argument("--pre-context-seconds", type=float, default=15.0)
    parser.add_argument("--post-context-seconds", type=float, default=15.0)
    return parser.parse_args()


def parse_column_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_positive_tags(raw_value: str) -> set[str]:
    parsed = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    return parsed or DEFAULT_POSITIVE_TAGS


def build_group_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns in label CSV: {missing}")
    return df.loc[:, columns].fillna("").astype(str).agg("|".join, axis=1)


def expand_scenario_members(
    group_df: pd.DataFrame,
    seed_indices: list[int],
    pre_events: int,
    post_events: int,
    pre_seconds: float,
    post_seconds: float,
) -> pd.Index:
    if not seed_indices:
        return pd.Index([])

    dt_series = group_df["dt"]
    min_seed_pos = min(seed_indices)
    max_seed_pos = max(seed_indices)
    min_seed_time = dt_series.iloc[min_seed_pos]
    max_seed_time = dt_series.iloc[max_seed_pos]

    lower_event_bound = max(0, min_seed_pos - pre_events)
    upper_event_bound = min(len(group_df) - 1, max_seed_pos + post_events)

    lower_time_bound = min_seed_time - pd.Timedelta(seconds=pre_seconds) if pd.notna(min_seed_time) else None
    upper_time_bound = max_seed_time + pd.Timedelta(seconds=post_seconds) if pd.notna(max_seed_time) else None

    event_mask = pd.Series(False, index=group_df.index)
    event_mask.iloc[lower_event_bound : upper_event_bound + 1] = True

    if lower_time_bound is not None and upper_time_bound is not None:
        time_mask = (dt_series >= lower_time_bound) & (dt_series <= upper_time_bound)
        final_mask = event_mask | time_mask
    else:
        final_mask = event_mask

    return group_df.index[final_mask]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_label_csv, low_memory=False)
    required = {args.line_column, args.timestamp_column, args.tag_column}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input_label_csv}: {missing}")

    positive_tags = parse_positive_tags(args.positive_tags)
    group_columns = parse_column_list(args.group_columns)
    df["dt"] = pd.to_datetime(df[args.timestamp_column], errors="coerce")
    df["group_key"] = build_group_key(df, group_columns)
    df["tag_norm"] = df[args.tag_column].fillna("").astype(str).str.strip().str.lower()
    df["seed_is_attack"] = df["tag_norm"].isin(positive_tags).astype(int)
    df = df.sort_values(["group_key", "dt", args.line_column], kind="mergesort").reset_index(drop=True)

    df["supervised_is_anomaly"] = 0
    df["scenario_id"] = ""
    df["scenario_role"] = "normal"
    df["scenario_seed_count"] = 0
    df["scenario_group_key"] = ""
    df["scenario_member_count"] = 0
    df["scenario_window_start_line"] = pd.NA
    df["scenario_window_end_line"] = pd.NA
    df["scenario_core_start_line"] = pd.NA
    df["scenario_core_end_line"] = pd.NA
    df["scenario_start_timestamp"] = ""
    df["scenario_end_timestamp"] = ""
    df["scenario_duration_seconds"] = 0.0

    scenario_counter = 0
    scenario_records: list[dict] = []
    for group_key, group_df in df.groupby("group_key", sort=False):
        positive_positions = [int(pos) for pos in group_df.index[group_df["seed_is_attack"] == 1]]
        if not positive_positions:
            continue

        scenario_clusters: list[list[int]] = []
        current_cluster: list[int] = [positive_positions[0]]
        for current_idx in positive_positions[1:]:
            previous_idx = current_cluster[-1]
            prev_time = df.at[previous_idx, "dt"]
            curr_time = df.at[current_idx, "dt"]
            gap_seconds = (curr_time - prev_time).total_seconds() if pd.notna(prev_time) and pd.notna(curr_time) else None
            if gap_seconds is not None and gap_seconds <= args.gap_seconds:
                current_cluster.append(current_idx)
            else:
                scenario_clusters.append(current_cluster)
                current_cluster = [current_idx]
        scenario_clusters.append(current_cluster)

        group_df = group_df.copy()
        group_df["group_pos"] = range(len(group_df))
        for cluster in scenario_clusters:
            scenario_counter += 1
            scenario_id = f"scenario_{scenario_counter:04d}"
            cluster_positions = group_df.loc[group_df.index.isin(cluster), "group_pos"].tolist()
            member_indices = expand_scenario_members(
                group_df=group_df,
                seed_indices=cluster_positions,
                pre_events=args.pre_context_events,
                post_events=args.post_context_events,
                pre_seconds=args.pre_context_seconds,
                post_seconds=args.post_context_seconds,
            )
            if member_indices.empty:
                continue

            member_df = df.loc[member_indices].copy()
            core_df = df.loc[cluster].copy()
            scenario_start_dt = member_df["dt"].min()
            scenario_end_dt = member_df["dt"].max()
            duration_seconds = (
                float((scenario_end_dt - scenario_start_dt).total_seconds())
                if pd.notna(scenario_start_dt) and pd.notna(scenario_end_dt)
                else 0.0
            )
            scenario_window_start_line = int(member_df[args.line_column].min())
            scenario_window_end_line = int(member_df[args.line_column].max())
            scenario_core_start_line = int(core_df[args.line_column].min())
            scenario_core_end_line = int(core_df[args.line_column].max())
            scenario_start_timestamp = "" if pd.isna(scenario_start_dt) else scenario_start_dt.isoformat()
            scenario_end_timestamp = "" if pd.isna(scenario_end_dt) else scenario_end_dt.isoformat()

            df.loc[member_indices, "supervised_is_anomaly"] = 1
            df.loc[member_indices, "scenario_id"] = scenario_id
            df.loc[member_indices, "scenario_group_key"] = group_key
            df.loc[member_indices, "scenario_seed_count"] = len(cluster)
            df.loc[member_indices, "scenario_member_count"] = len(member_indices)
            df.loc[member_indices, "scenario_window_start_line"] = scenario_window_start_line
            df.loc[member_indices, "scenario_window_end_line"] = scenario_window_end_line
            df.loc[member_indices, "scenario_core_start_line"] = scenario_core_start_line
            df.loc[member_indices, "scenario_core_end_line"] = scenario_core_end_line
            df.loc[member_indices, "scenario_start_timestamp"] = scenario_start_timestamp
            df.loc[member_indices, "scenario_end_timestamp"] = scenario_end_timestamp
            df.loc[member_indices, "scenario_duration_seconds"] = duration_seconds
            df.loc[member_indices, "scenario_role"] = "context"
            df.loc[cluster, "scenario_role"] = "core"
            scenario_records.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_group_key": group_key,
                    "seed_event_count": int(len(cluster)),
                    "member_row_count": int(len(member_indices)),
                    "core_start_line": scenario_core_start_line,
                    "core_end_line": scenario_core_end_line,
                    "window_start_line": scenario_window_start_line,
                    "window_end_line": scenario_window_end_line,
                    "start_timestamp": scenario_start_timestamp,
                    "end_timestamp": scenario_end_timestamp,
                    "duration_seconds": duration_seconds,
                }
            )

    output_columns = [
        args.line_column,
        args.timestamp_column,
        args.tag_column,
        "seed_is_attack",
        "supervised_is_anomaly",
        "scenario_id",
        "scenario_role",
        "scenario_seed_count",
        "scenario_member_count",
        "scenario_group_key",
        "scenario_window_start_line",
        "scenario_window_end_line",
        "scenario_core_start_line",
        "scenario_core_end_line",
        "scenario_start_timestamp",
        "scenario_end_timestamp",
        "scenario_duration_seconds",
    ]
    passthrough_columns = [column for column in group_columns if column not in output_columns]
    output_columns.extend(passthrough_columns)

    output_df = df[output_columns].copy()
    output_df.to_csv(args.output_label_csv, index=False)
    scenario_summary_df = pd.DataFrame(scenario_records).sort_values(
        ["start_timestamp", "scenario_id"], kind="mergesort"
    )
    if args.scenario_summary_csv:
        scenario_summary_df.to_csv(args.scenario_summary_csv, index=False)

    scenario_count = int((output_df["scenario_id"] != "").sum())
    positive_rows = int(output_df["supervised_is_anomaly"].sum())
    print(f"Saved prepared labels to {args.output_label_csv}")
    if args.scenario_summary_csv:
        print(f"Saved scenario summary to {args.scenario_summary_csv}")
    print(f"Positive seed events: {int(output_df['seed_is_attack'].sum())}")
    print(f"Positive window rows: {positive_rows}")
    print(f"Scenario-labeled rows: {scenario_count}")
    if "scenario_id" in output_df.columns:
        unique_scenarios = output_df.loc[output_df["scenario_id"] != "", "scenario_id"].nunique()
        print(f"Unique scenarios: {int(unique_scenarios)}")


if __name__ == "__main__":
    main()
