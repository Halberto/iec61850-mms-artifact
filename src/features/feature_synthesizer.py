import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


RESULT_CONTEXT_COLUMNS = [
    "line_number",
    "final_tag",
    "protocol_score",
    "stat_score",
    "origin_identifier",
    "ctl_num",
    "report_seq_num",
]

DEFAULT_POSITIVE_LABELS = {"1", "true", "yes", "attack", "anomaly", "malicious", "masquerade", "failed-control", "likely-attack"}
DEFAULT_NEGATIVE_LABELS = {"0", "false", "no", "normal", "benign"}
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
SAMPLE_DATA_DIR = REPO_ROOT / "data" / "sample"
LABELS_DIR = REPO_ROOT / "data" / "labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MMS semantic features and optionally merge an independent label file "
            "instead of relying on hybrid final_tag."
        )
    )
    parser.add_argument("--input-csv", default=str(SAMPLE_DATA_DIR / "mms_sample_100k.csv"))
    parser.add_argument("--results-csv", default=str(RAW_DATA_DIR / "hybrid_ids_alerts_full_capture.csv"))
    parser.add_argument("--output-csv", default=str(SAMPLE_DATA_DIR / "mms_ml_features_100k_rebuilt.csv"))
    parser.add_argument("--label-csv", default=str(LABELS_DIR / "mms_full_capture_supervised_labels.csv"))
    parser.add_argument("--feature-key-column", default="line_number")
    parser.add_argument("--label-key-column", default="line_number")
    parser.add_argument("--label-value-column", default="tag")
    parser.add_argument("--label-output-column", default="supervised_is_anomaly")
    parser.add_argument(
        "--label-extra-columns",
        default=(
            "scenario_id,scenario_role,scenario_seed_count,scenario_member_count,"
            "scenario_group_key,scenario_window_start_line,scenario_window_end_line,"
            "scenario_core_start_line,scenario_core_end_line,scenario_start_timestamp,"
            "scenario_end_timestamp,scenario_duration_seconds,seed_is_attack"
        ),
        help="Additional columns from the label CSV to merge into the feature table.",
    )
    parser.add_argument(
        "--positive-label-values",
        default="failed-control,likely-attack,attack,anomaly,masquerade,malicious,1,true,yes",
    )
    parser.add_argument(
        "--negative-label-values",
        default="normal,benign,0,false,no",
    )
    parser.add_argument(
        "--fallback-to-heuristic-labels",
        action="store_true",
        help="Use the heuristic hybrid tag for rows that do not have a supervised label.",
    )
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def categorical_codes(series: pd.Series) -> pd.Series:
    return series.astype("category").cat.codes.astype("int32")


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_label_value_sets(raw_value: str, defaults: set[str]) -> set[str]:
    parsed = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    return parsed or defaults


def parse_column_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}


def deduplicate_results_by_line_number(df_results: pd.DataFrame) -> pd.DataFrame:
    if "line_number" not in df_results.columns:
        raise ValueError("Hybrid results CSV must contain 'line_number'.")

    if "event_source" in df_results.columns:
        source_priority = df_results["event_source"].fillna("").astype(str).map({"packet": 0, "synthetic": 1}).fillna(2)
        df_results = (
            df_results.assign(_source_priority=source_priority)
            .sort_values(["line_number", "_source_priority"], kind="mergesort")
            .drop_duplicates(subset=["line_number"], keep="first")
            .drop(columns=["_source_priority"])
        )
    else:
        df_results = df_results.drop_duplicates(subset=["line_number"], keep="first")
    return df_results
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def encode_supervised_labels(
    series: pd.Series,
    positive_values: set[str],
    negative_values: set[str],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.where(numeric.isin([0, 1])).astype("float64")

    normalized = normalize_text(series).str.lower()
    encoded = pd.Series(np.nan, index=series.index, dtype="float64")
    encoded.loc[normalized.isin(positive_values)] = 1.0
    encoded.loc[normalized.isin(negative_values)] = 0.0
    return encoded


def merge_supervised_labels(
    df: pd.DataFrame,
    label_csv: str,
    feature_key_column: str,
    label_key_column: str,
    label_value_column: str,
    label_output_column: str,
    label_extra_columns: list[str],
    positive_values: set[str],
    negative_values: set[str],
) -> pd.DataFrame:
    print(f"Loading supervised labels from {label_csv}")
    label_df = pd.read_csv(label_csv, low_memory=False)
    missing = [column for column in (label_key_column, label_value_column) if column not in label_df.columns]
    if missing:
        raise ValueError(f"Missing label columns in {label_csv}: {missing}")
    if feature_key_column not in df.columns:
        raise ValueError(f"Feature key column '{feature_key_column}' is not available in the feature dataframe.")

    requested_columns = [label_key_column, label_value_column] + [column for column in label_extra_columns if column]
    available_columns = [column for column in requested_columns if column in label_df.columns]
    label_df = label_df[available_columns].copy()
    label_df[label_output_column] = encode_supervised_labels(
        label_df[label_value_column],
        positive_values=positive_values,
        negative_values=negative_values,
    )
    if label_value_column != label_output_column:
        label_df = label_df.drop(columns=[label_value_column])
    label_df = label_df.drop_duplicates(subset=[label_key_column], keep="last")

    merged = df.merge(
        label_df,
        left_on=feature_key_column,
        right_on=label_key_column,
        how="left",
    )
    if label_key_column != feature_key_column:
        merged = merged.drop(columns=[label_key_column])

    labeled_rows = int(merged[label_output_column].notna().sum())
    positive_rows = int(merged[label_output_column].fillna(0).sum())
    print(f"Merged {labeled_rows} supervised labels ({positive_rows} positives).")
    return merged


def get_nested_value(payload: dict, path: Iterable[str], default: Any = np.nan) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def fill_text_from_sources(*sources: pd.Series) -> pd.Series:
    result = pd.Series("", index=sources[0].index, dtype="object")
    for source in sources:
        normalized = normalize_text(source)
        result = result.mask(result == "", normalized)
    return result


def add_group_prior_count(
    df: pd.DataFrame,
    group_cols: list[str],
    output_col: str,
    valid_mask: pd.Series | None = None,
) -> None:
    result = pd.Series(0, index=df.index, dtype="int32")
    if valid_mask is None:
        valid_mask = pd.Series(True, index=df.index)
    valid_mask = valid_mask.fillna(False)
    if valid_mask.any():
        result.loc[valid_mask] = (
            df.loc[valid_mask]
            .groupby(group_cols, sort=False)
            .cumcount()
            .astype("int32")
        )
    df[output_col] = result


def add_group_time_delta(
    df: pd.DataFrame,
    group_cols: list[str],
    output_col: str,
    valid_mask: pd.Series | None = None,
) -> None:
    result = pd.Series(0.0, index=df.index, dtype="float64")
    if valid_mask is None:
        valid_mask = pd.Series(True, index=df.index)
    valid_mask = valid_mask.fillna(False)
    if valid_mask.any():
        result.loc[valid_mask] = (
            df.loc[valid_mask]
            .groupby(group_cols, sort=False)["dt"]
            .diff()
            .dt.total_seconds()
            .fillna(0.0)
        )
    df[output_col] = result


def extract_features(args: argparse.Namespace) -> None:
    input_csv = args.input_csv
    results_csv = args.results_csv
    output_csv = args.output_csv

    print(f"Loading input data: {input_csv}")
    df_raw = pd.read_csv(input_csv, low_memory=False)
    df_raw["line_number"] = range(1, len(df_raw) + 1)

    print(f"Loading hybrid results: {results_csv}")
    df_results = pd.read_csv(results_csv, low_memory=False)
    df_results = deduplicate_results_by_line_number(df_results)
    result_columns = [column for column in RESULT_CONTEXT_COLUMNS if column in df_results.columns]

    print(f"Merging raw events with hybrid context columns: {result_columns}")
    df = df_raw.merge(df_results[result_columns], on="line_number", how="inner")
    print(f"Processing {len(df)} aligned rows...")

    if args.label_csv:
        positive_values = parse_label_value_sets(args.positive_label_values, DEFAULT_POSITIVE_LABELS)
        negative_values = parse_label_value_sets(args.negative_label_values, DEFAULT_NEGATIVE_LABELS)
        df = merge_supervised_labels(
            df=df,
            label_csv=args.label_csv,
            feature_key_column=args.feature_key_column,
            label_key_column=args.label_key_column,
            label_value_column=args.label_value_column,
            label_output_column=args.label_output_column,
            label_extra_columns=parse_column_list(args.label_extra_columns),
            positive_values=positive_values,
            negative_values=negative_values,
        )

    df["dt"] = pd.to_datetime(df["wtimestamp"], errors="coerce")
    df = df.sort_values(["dt", "line_number"], kind="mergesort").reset_index(drop=True)

    control_payloads = df["control_parameters"].apply(parse_json_object)
    parsed_origin_identifier = control_payloads.apply(
        lambda payload: get_nested_value(payload, ("originIdentifier", "value"), "")
    )
    parsed_ctl_num = control_payloads.apply(lambda payload: get_nested_value(payload, ("ctlNum", "value")))
    parsed_origin_category = control_payloads.apply(
        lambda payload: get_nested_value(payload, ("originCategory", "value"))
    )
    parsed_test_flag = control_payloads.apply(lambda payload: get_nested_value(payload, ("test", "value")))
    parsed_check_bits = control_payloads.apply(lambda payload: get_nested_value(payload, ("check", "value"), ""))

    df["origin_identifier_value"] = fill_text_from_sources(
        df.get("origin_identifier", pd.Series("", index=df.index)),
        parsed_origin_identifier,
    )
    df["ctl_num_value"] = numeric_series(
        df.get("ctl_num", pd.Series(np.nan, index=df.index))
    ).fillna(numeric_series(parsed_ctl_num))
    df["report_seq_num_value"] = numeric_series(
        df.get("report_seq_num", pd.Series(np.nan, index=df.index))
    )
    df["origin_category_value"] = numeric_series(parsed_origin_category).fillna(-1)
    df["control_test_flag"] = parsed_test_flag.astype("boolean").fillna(False).astype("int8")
    df["check_bits_length"] = normalize_text(parsed_check_bits).str.len().astype("int32")

    df["event_timestamp"] = df["wtimestamp"]
    df["event_time_unix"] = (df["dt"].astype("int64") // 10**9).where(df["dt"].notna(), np.nan)
    df["hour_of_day"] = df["dt"].dt.hour.fillna(-1).astype("int32")
    df["day_of_week"] = df["dt"].dt.dayofweek.fillna(-1).astype("int32")
    df["time_bucket_15m"] = df["dt"].dt.floor("15min").astype(str)
    df["is_off_hours"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 20)).astype("int8")

    df["service_norm"] = normalize_text(df["service"])
    df["pdu_type_norm"] = normalize_text(df["pdu_type"])
    df["src_ip_norm"] = normalize_text(df["src_ip"])
    df["dst_ip_norm"] = normalize_text(df["dst_ip"])
    df["stream_id_norm"] = normalize_text(df["stream_id"])
    df["control_object_norm"] = normalize_text(df["control_object"])
    df["control_action_norm"] = normalize_text(df["control_action"])
    df["invoke_id_norm"] = normalize_text(df["invoke_id"])
    df["origin_identifier_norm"] = normalize_text(df["origin_identifier_value"])

    df["has_origin_identifier"] = (df["origin_identifier_norm"] != "").astype("int8")
    df["has_control_object"] = (df["control_object_norm"] != "").astype("int8")
    df["has_control_action"] = (df["control_action_norm"] != "").astype("int8")
    df["has_invoke_id"] = (df["invoke_id_norm"] != "").astype("int8")
    df["has_ctl_num"] = df["ctl_num_value"].notna().astype("int8")
    df["has_report_seq_num"] = df["report_seq_num_value"].notna().astype("int8")
    df["is_request"] = (df["direction"] == "REQUEST").astype("int8")
    df["is_response"] = (df["direction"] == "RESPONSE").astype("int8")
    df["is_control_event"] = (
        numeric_series(df["is_control"]).fillna(0).astype(int)
        | df["has_control_object"]
        | df["has_control_action"]
    ).astype("int8")
    df["is_write_service"] = (df["service_norm"] == "WRITE").astype("int8")
    df["is_report_event"] = (
        df["service_norm"].eq("UNCONFIRMED") | df["summary"].fillna("").astype(str).str.contains("<RPT>", regex=False)
    ).astype("int8")

    df["actor_id"] = np.where(
        df["origin_identifier_norm"] != "",
        df["src_ip_norm"] + "|" + df["origin_identifier_norm"],
        df["src_ip_norm"] + "|<NO_ORIGIN>",
    )
    df["source_origin_pair"] = np.where(
        df["origin_identifier_norm"] != "",
        df["src_ip_norm"] + "|" + df["origin_identifier_norm"],
        df["src_ip_norm"] + "|<NO_ORIGIN>",
    )
    df["actor_object_pair"] = np.where(
        df["has_control_object"] == 1,
        df["actor_id"] + "|" + df["control_object_norm"],
        "",
    )
    df["actor_action_pair"] = np.where(
        df["has_control_action"] == 1,
        df["actor_id"] + "|" + df["control_action_norm"],
        "",
    )
    df["object_action_pair"] = np.where(
        (df["has_control_object"] == 1) & (df["has_control_action"] == 1),
        df["control_object_norm"] + "|" + df["control_action_norm"],
        "",
    )
    df["origin_object_pair"] = np.where(
        (df["has_origin_identifier"] == 1) & (df["has_control_object"] == 1),
        df["origin_identifier_norm"] + "|" + df["control_object_norm"],
        "",
    )
    df["actor_object_action_triplet"] = np.where(
        (df["has_control_object"] == 1) & (df["has_control_action"] == 1),
        df["actor_id"] + "|" + df["control_object_norm"] + "|" + df["control_action_norm"],
        "",
    )

    df["service_cat"] = categorical_codes(df["service_norm"])
    df["pdu_type_cat"] = categorical_codes(df["pdu_type_norm"])
    df["src_ip_cat"] = categorical_codes(df["src_ip_norm"])
    df["dst_ip_cat"] = categorical_codes(df["dst_ip_norm"])
    df["stream_id_cat"] = categorical_codes(df["stream_id_norm"])
    df["origin_identifier_cat"] = categorical_codes(df["origin_identifier_norm"])
    df["control_object_cat"] = categorical_codes(df["control_object_norm"])
    df["control_action_cat"] = categorical_codes(df["control_action_norm"])
    df["invoke_id_cat"] = categorical_codes(df["invoke_id_norm"])
    df["actor_id_cat"] = categorical_codes(pd.Series(df["actor_id"]))
    df["source_origin_pair_cat"] = categorical_codes(pd.Series(df["source_origin_pair"]))
    df["actor_object_pair_cat"] = categorical_codes(pd.Series(df["actor_object_pair"]))
    df["actor_action_pair_cat"] = categorical_codes(pd.Series(df["actor_action_pair"]))
    df["object_action_pair_cat"] = categorical_codes(pd.Series(df["object_action_pair"]))

    print("Calculating temporal, semantic, and relational history features...")
    stream_groups = df.groupby("stream_id_norm", sort=False)
    df["stream_event_index"] = stream_groups.cumcount().astype("int32")
    df["prev_service_norm"] = stream_groups["service_norm"].shift(1).fillna("<START>")
    df["service_transition_norm"] = df["prev_service_norm"] + "->" + df["service_norm"]
    df["prev_invoke_id_norm"] = stream_groups["invoke_id_norm"].shift(1).fillna("")
    df["time_since_last_stream_event"] = stream_groups["dt"].diff().dt.total_seconds().fillna(0.0)

    actor_mask = df["actor_id"] != ""
    add_group_prior_count(df, ["actor_id"], "actor_prior_count", actor_mask)
    add_group_prior_count(df, ["src_ip_norm"], "src_ip_prior_count")
    add_group_prior_count(df, ["origin_identifier_norm"], "origin_prior_count", df["has_origin_identifier"] == 1)
    add_group_prior_count(df, ["source_origin_pair"], "source_origin_prior_count", df["has_origin_identifier"] == 1)
    add_group_prior_count(df, ["control_object_norm"], "object_prior_count", df["has_control_object"] == 1)
    add_group_prior_count(df, ["control_action_norm"], "action_prior_count", df["has_control_action"] == 1)
    add_group_prior_count(df, ["actor_object_pair"], "actor_object_prior_count", df["actor_object_pair"] != "")
    add_group_prior_count(df, ["actor_action_pair"], "actor_action_prior_count", df["actor_action_pair"] != "")
    add_group_prior_count(df, ["object_action_pair"], "object_action_prior_count", df["object_action_pair"] != "")
    add_group_prior_count(df, ["origin_object_pair"], "origin_object_prior_count", df["origin_object_pair"] != "")
    add_group_prior_count(
        df,
        ["actor_object_action_triplet"],
        "actor_object_action_prior_count",
        df["actor_object_action_triplet"] != "",
    )

    object_actor_first_seen = (
        (df["has_control_object"] == 1)
        & (~df.loc[:, ["control_object_norm", "actor_id"]].duplicated())
    ).astype("int8")
    df["object_unique_actor_count_prior"] = 0
    object_mask = df["has_control_object"] == 1
    if object_mask.any():
        df.loc[object_mask, "object_unique_actor_count_prior"] = (
            object_actor_first_seen.loc[object_mask]
            .groupby(df.loc[object_mask, "control_object_norm"], sort=False)
            .cumsum()
            .subtract(object_actor_first_seen.loc[object_mask])
            .astype("int32")
        )

    origin_src_first_seen = (
        (df["has_origin_identifier"] == 1)
        & (~df.loc[:, ["origin_identifier_norm", "src_ip_norm"]].duplicated())
    ).astype("int8")
    df["origin_unique_src_count_prior"] = 0
    origin_mask = df["has_origin_identifier"] == 1
    if origin_mask.any():
        df.loc[origin_mask, "origin_unique_src_count_prior"] = (
            origin_src_first_seen.loc[origin_mask]
            .groupby(df.loc[origin_mask, "origin_identifier_norm"], sort=False)
            .cumsum()
            .subtract(origin_src_first_seen.loc[origin_mask])
            .astype("int32")
        )

    df["origin_seen_with_new_src"] = (
        (df["origin_prior_count"] > 0) & (df["source_origin_prior_count"] == 0)
    ).astype("int8")
    df["source_seen_with_new_origin"] = (
        (df["src_ip_prior_count"] > 0) & (df["source_origin_prior_count"] == 0) & (df["has_origin_identifier"] == 1)
    ).astype("int8")
    df["object_seen_from_new_actor"] = (
        (df["object_prior_count"] > 0) & (df["actor_object_prior_count"] == 0) & (df["has_control_object"] == 1)
    ).astype("int8")
    df["action_seen_from_new_actor"] = (
        (df["action_prior_count"] > 0) & (df["actor_action_prior_count"] == 0) & (df["has_control_action"] == 1)
    ).astype("int8")
    df["source_origin_consistent"] = (
        (df["has_origin_identifier"] == 1) & (df["origin_seen_with_new_src"] == 0)
    ).astype("int8")

    add_group_time_delta(df, ["actor_object_pair"], "time_since_last_actor_object", df["actor_object_pair"] != "")
    add_group_time_delta(df, ["control_object_norm"], "time_since_last_object_control", df["has_control_object"] == 1)
    add_group_time_delta(df, ["source_origin_pair"], "time_since_last_origin_source", df["has_origin_identifier"] == 1)

    df["prev_action_same_object_norm"] = ""
    df["prev_actor_same_object_norm"] = ""
    if object_mask.any():
        object_groups = df.loc[object_mask].groupby("control_object_norm", sort=False)
        df.loc[object_mask, "prev_action_same_object_norm"] = (
            object_groups["control_action_norm"].shift(1).fillna("<START>")
        )
        df.loc[object_mask, "prev_actor_same_object_norm"] = (
            object_groups["actor_id"].shift(1).fillna("<START>")
        )

    df["action_transition_norm"] = np.where(
        df["has_control_object"] == 1,
        df["prev_action_same_object_norm"] + "->" + df["control_action_norm"],
        "<NO_OBJECT>",
    )
    df["actor_changed_for_object"] = (
        (df["has_control_object"] == 1)
        & (df["object_prior_count"] > 0)
        & (df["prev_actor_same_object_norm"] != "")
        & (df["prev_actor_same_object_norm"] != "<START>")
        & (df["prev_actor_same_object_norm"] != df["actor_id"])
    ).astype("int8")

    df["prev_service_cat"] = categorical_codes(df["prev_service_norm"])
    df["service_transition_cat"] = categorical_codes(df["service_transition_norm"])
    df["prev_action_same_object_cat"] = categorical_codes(df["prev_action_same_object_norm"])
    df["action_transition_cat"] = categorical_codes(df["action_transition_norm"])
    df["prev_actor_same_object_cat"] = categorical_codes(df["prev_actor_same_object_norm"])

    invoke_id_numeric = numeric_series(df["invoke_id_norm"])
    invoke_id_numeric = invoke_id_numeric.where(invoke_id_numeric.abs() <= 1_000_000_000)
    df["invoke_id_value"] = invoke_id_numeric.fillna(-1)
    df["invoke_id_reused_stream"] = (
        (df["has_invoke_id"] == 1)
        & df.duplicated(subset=["stream_id_norm", "invoke_id_norm"])
    ).astype("int8")
    df["invoke_id_delta_stream"] = (
        df["invoke_id_value"] - stream_groups["invoke_id_value"].shift(1)
    ).fillna(0.0)

    df["ctl_num_val"] = df["ctl_num_value"].fillna(-1)
    df["report_seq_num_val"] = df["report_seq_num_value"].fillna(-1)
    df["prev_ctl_num_stream"] = stream_groups["ctl_num_val"].shift(1)
    df["ctl_num_delta_stream"] = (df["ctl_num_val"] - df["prev_ctl_num_stream"]).fillna(0.0)
    df["prev_report_seq_stream"] = stream_groups["report_seq_num_val"].shift(1)
    df["report_seq_delta_stream"] = (df["report_seq_num_val"] - df["prev_report_seq_stream"]).fillna(0.0)
    df["report_seq_reused_stream"] = (
        (df["has_report_seq_num"] == 1)
        & df.duplicated(subset=["stream_id_norm", "report_seq_num_val"])
    ).astype("int8")
    df["report_seq_regression_stream"] = (
        (df["has_report_seq_num"] == 1)
        & df["prev_report_seq_stream"].notna()
        & (df["report_seq_num_val"] < df["prev_report_seq_stream"])
    ).astype("int8")

    actor_object_mask = (df["actor_object_pair"] != "") & (df["has_ctl_num"] == 1)
    df["prev_ctl_num_actor_object"] = np.nan
    if actor_object_mask.any():
        df.loc[actor_object_mask, "prev_ctl_num_actor_object"] = (
            df.loc[actor_object_mask]
            .groupby("actor_object_pair", sort=False)["ctl_num_val"]
            .shift(1)
        )
    df["ctl_num_delta_actor_object"] = (
        df["ctl_num_val"] - df["prev_ctl_num_actor_object"]
    ).fillna(0.0)
    df["ctl_num_reused_actor_object"] = (
        actor_object_mask & df.duplicated(subset=["actor_object_pair", "ctl_num_val"])
    ).astype("int8")
    df["ctl_num_regression_actor_object"] = (
        actor_object_mask
        & df["prev_ctl_num_actor_object"].notna()
        & (df["ctl_num_val"] < df["prev_ctl_num_actor_object"])
    ).astype("int8")

    def get_payload_size(raw_hex: Any) -> int:
        if not isinstance(raw_hex, str):
            return 0
        cleaned = raw_hex.strip()
        if not cleaned:
            return 0
        return len(cleaned) // 2

    df["payload_size"] = df["raw_mms_hex"].apply(get_payload_size).astype("int32")
    df["payload_size_delta"] = stream_groups["payload_size"].diff().fillna(0.0)
    df["time_delta"] = stream_groups["dt"].diff().dt.total_seconds().fillna(0.0)
    df["mean_time_delta_stream"] = stream_groups["time_delta"].transform("mean")
    df["std_time_delta_stream"] = stream_groups["time_delta"].transform("std").fillna(0.0)
    df["time_z_score"] = (
        (df["time_delta"] - df["mean_time_delta_stream"]) / (df["std_time_delta_stream"] + 1e-6)
    )

    df["actor_object_legitimacy_ratio"] = (
        df["actor_object_prior_count"] / (df["object_prior_count"] + 1.0)
    )
    df["actor_action_legitimacy_ratio"] = (
        df["actor_action_prior_count"] / (df["actor_prior_count"] + 1.0)
    )
    df["origin_object_legitimacy_ratio"] = (
        df["origin_object_prior_count"] / (df["object_prior_count"] + 1.0)
    )
    df["actor_object_action_ratio"] = (
        df["actor_object_action_prior_count"] / (df["actor_object_prior_count"] + 1.0)
    )

    df["heuristic_is_anomaly"] = (df["final_tag"] != "normal").astype("int8")
    if args.label_csv:
        df["is_anomaly"] = df[args.label_output_column]
        if args.fallback_to_heuristic_labels:
            df["is_anomaly"] = df["is_anomaly"].fillna(df["heuristic_is_anomaly"])
    else:
        df["is_anomaly"] = df["heuristic_is_anomaly"]

    ml_features = [
        "line_number",
        "event_timestamp",
        "event_time_unix",
        "time_bucket_15m",
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
        "seed_is_attack",
        "hour_of_day",
        "day_of_week",
        "is_off_hours",
        "stream_id_cat",
        "service_cat",
        "pdu_type_cat",
        "src_ip_cat",
        "dst_ip_cat",
        "origin_identifier_cat",
        "control_object_cat",
        "control_action_cat",
        "invoke_id_cat",
        "actor_id_cat",
        "source_origin_pair_cat",
        "actor_object_pair_cat",
        "actor_action_pair_cat",
        "object_action_pair_cat",
        "prev_service_cat",
        "service_transition_cat",
        "prev_action_same_object_cat",
        "action_transition_cat",
        "prev_actor_same_object_cat",
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
        "control_test_flag",
        "source_origin_consistent",
        "origin_seen_with_new_src",
        "source_seen_with_new_origin",
        "object_seen_from_new_actor",
        "action_seen_from_new_actor",
        "actor_changed_for_object",
        "invoke_id_reused_stream",
        "ctl_num_reused_actor_object",
        "ctl_num_regression_actor_object",
        "report_seq_reused_stream",
        "report_seq_regression_stream",
        "stream_event_index",
        "time_since_last_stream_event",
        "time_since_last_actor_object",
        "time_since_last_object_control",
        "time_since_last_origin_source",
        "actor_prior_count",
        "src_ip_prior_count",
        "origin_prior_count",
        "source_origin_prior_count",
        "object_prior_count",
        "action_prior_count",
        "actor_object_prior_count",
        "actor_action_prior_count",
        "object_action_prior_count",
        "origin_object_prior_count",
        "actor_object_action_prior_count",
        "object_unique_actor_count_prior",
        "origin_unique_src_count_prior",
        "invoke_id_value",
        "invoke_id_delta_stream",
        "ctl_num_val",
        "ctl_num_delta_stream",
        "ctl_num_delta_actor_object",
        "report_seq_num_val",
        "report_seq_delta_stream",
        "origin_category_value",
        "check_bits_length",
        "time_delta",
        "payload_size",
        "payload_size_delta",
        "time_z_score",
        "actor_object_legitimacy_ratio",
        "actor_action_legitimacy_ratio",
        "origin_object_legitimacy_ratio",
        "actor_object_action_ratio",
        "protocol_score",
        "stat_score",
        "heuristic_is_anomaly",
        args.label_output_column if args.label_csv else "is_anomaly",
        "is_anomaly",
    ]

    ml_features = list(dict.fromkeys(ml_features))
    ml_features = [column for column in ml_features if column in df.columns]
    df_ml = df[ml_features].copy()
    non_numeric_columns = {
        "event_timestamp",
        "time_bucket_15m",
        "scenario_id",
        "scenario_role",
        "scenario_group_key",
        "scenario_start_timestamp",
        "scenario_end_timestamp",
    }
    numeric_columns = [column for column in df_ml.columns if column not in non_numeric_columns]
    df_ml[numeric_columns] = (
        df_ml[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    print(f"Saving synthesized features to {output_csv}")
    df_ml.to_csv(output_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    extract_features(parse_args())
