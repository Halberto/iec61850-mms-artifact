import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


DEFAULT_EXPERT_COLUMNS = ("protocol_score", "stat_score")
DEFAULT_IDENTITY_COLUMNS = ("src_ip_cat", "dst_ip_cat", "stream_id_cat")
DEFAULT_METADATA_COLUMNS = ("line_number", "event_timestamp", "event_time_unix", "time_bucket_15m")
DEFAULT_SCENARIO_COLUMNS = (
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
)
DEFAULT_PREFERRED_LABEL_COLUMNS = ("supervised_is_anomaly", "is_anomaly")
MAX_ABS_FEATURE_VALUE = 1_000_000_000.0
MAX_THRESHOLD_CANDIDATES = 500
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a defensible MMS classifier with group-aware evaluation. "
            "Expert score features are excluded by default because they can leak "
            "the upstream hybrid detector into the model target."
        )
    )
    parser.add_argument("--feature-csv", default=str(PROCESSED_DATA_DIR / "mms_ml_features_100k.csv"))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "mms_fusion_ml_model.joblib"))
    parser.add_argument("--report-path", default=str(RESULTS_DIR / "fusion_ml_report.txt"))
    parser.add_argument(
        "--label-column",
        default="supervised_is_anomaly",
        help="Preferred label column. Falls back to is_anomaly when the requested label is unavailable.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("group", "time", "scenario"),
        default="group",
        help="Use grouped channel splits, chronological splits, or scenario holdout splits.",
    )
    parser.add_argument(
        "--group-columns",
        default="src_ip_cat,dst_ip_cat",
        help="Comma-separated columns used to create communication groups for splitting.",
    )
    parser.add_argument(
        "--time-column",
        default="event_time_unix",
        help="Timestamp/order column used when --split-mode time.",
    )
    parser.add_argument(
        "--scenario-column",
        default="scenario_id",
        help="Scenario identifier column used when --split-mode scenario.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--threshold-objective",
        choices=("row_f1", "scenario_f1"),
        default="row_f1",
        help="Metric used to pick the decision threshold on the validation split.",
    )
    parser.add_argument(
        "--include-expert-features",
        action="store_true",
        help="Keep protocol_score/stat_score in training. Use only with labels independent from the hybrid detector.",
    )
    parser.add_argument(
        "--keep-identity-features",
        action="store_true",
        help="Keep source/destination identity features in training. Off by default to reduce memorization.",
    )
    parser.add_argument(
        "--extra-drop-columns",
        default="",
        help="Additional comma-separated feature columns to exclude from training.",
    )
    return parser.parse_args()


def parse_column_list(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def build_group_ids(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing group columns: {missing}")
    return df.loc[:, columns].astype(str).agg("|".join, axis=1)


def build_scenario_group_ids(df: pd.DataFrame, scenario_column: str) -> pd.Series:
    if scenario_column not in df.columns:
        raise ValueError(f"Scenario column '{scenario_column}' not found in dataframe.")

    scenario_ids = df[scenario_column].fillna("").astype(str).str.strip()
    normal_mask = scenario_ids.eq("")
    if not normal_mask.any():
        return scenario_ids

    fallback_parts: list[pd.Series] = []
    if "scenario_group_key" in df.columns:
        fallback_parts.append(df["scenario_group_key"].fillna("").astype(str))
    elif "stream_id_cat" in df.columns:
        fallback_parts.append(df["stream_id_cat"].astype(str))
    elif "window_group" in df.columns:
        fallback_parts.append(df["window_group"].astype(str))

    if "time_bucket_15m" in df.columns:
        fallback_parts.append(df["time_bucket_15m"].fillna("").astype(str))
    elif "event_time_unix" in df.columns:
        fallback_parts.append((pd.to_numeric(df["event_time_unix"], errors="coerce").fillna(-1) // 900).astype(int).astype(str))
    elif "window_end_time_unix" in df.columns:
        fallback_parts.append((pd.to_numeric(df["window_end_time_unix"], errors="coerce").fillna(-1) // 900).astype(int).astype(str))
    elif "line_number" in df.columns:
        fallback_parts.append((pd.to_numeric(df["line_number"], errors="coerce").fillna(-1) // 500).astype(int).astype(str))

    if fallback_parts:
        fallback_group = fallback_parts[0]
        for part in fallback_parts[1:]:
            fallback_group = fallback_group + "|" + part
    else:
        fallback_group = pd.Series("normal", index=df.index)

    built = scenario_ids.copy()
    built.loc[normal_mask] = "normal|" + fallback_group.loc[normal_mask]
    return built


def build_time_order(df: pd.DataFrame, time_column: str) -> pd.Series:
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in feature CSV.")

    raw = df[time_column]
    if pd.api.types.is_numeric_dtype(raw):
        time_values = pd.to_numeric(raw, errors="coerce")
    else:
        parsed = pd.to_datetime(raw, errors="coerce")
        time_values = (parsed.astype("int64") / 10**9).where(parsed.notna(), np.nan)

    if time_values.isna().all():
        raise ValueError(f"Time column '{time_column}' could not be parsed into usable timestamps.")
    return time_values


def resolve_label_column(df: pd.DataFrame, requested_label: str) -> str:
    if requested_label in df.columns:
        return requested_label

    fallback_order = [column for column in DEFAULT_PREFERRED_LABEL_COLUMNS if column != requested_label]
    for candidate in fallback_order:
        if candidate in df.columns:
            print(
                f"Requested label column '{requested_label}' was not found. Falling back to '{candidate}'."
            )
            return candidate

    raise ValueError(
        f"Label column '{requested_label}' not found. Available columns do not include any of "
        f"{DEFAULT_PREFERRED_LABEL_COLUMNS}."
    )


def stratified_group_split(
    y: pd.Series,
    group_ids: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    group_df = (
        pd.DataFrame({"group_id": group_ids, "label": y.astype(int)})
        .groupby("group_id", as_index=False)
        .agg(group_label=("label", "max"), row_count=("label", "size"), positive_rows=("label", "sum"))
    )

    if group_df["group_label"].nunique() < 2:
        raise ValueError("Need both positive and negative groups for defensible grouped evaluation.")

    positive_group_count = int(group_df["group_label"].sum())
    negative_group_count = int((group_df["group_label"] == 0).sum())
    if positive_group_count < 2 or negative_group_count < 2:
        raise ValueError(
            "Need at least two positive groups and two negative groups to split groups stratified by class."
        )

    total_groups = len(group_df)
    if 0 < test_size < 1:
        requested_test_groups = int(round(total_groups * test_size))
    else:
        requested_test_groups = int(test_size)
    requested_test_groups = max(requested_test_groups, 2)
    requested_test_groups = min(requested_test_groups, total_groups - 2)
    if requested_test_groups < 2:
        raise ValueError("Not enough groups to create train and test splits with both classes represented.")

    positive_groups = group_df.loc[group_df["group_label"] == 1, "group_id"].tolist()
    negative_groups = group_df.loc[group_df["group_label"] == 0, "group_id"].tolist()

    desired_positive_test = int(round(requested_test_groups * positive_group_count / total_groups))
    desired_positive_test = max(desired_positive_test, 1)
    desired_positive_test = min(desired_positive_test, positive_group_count - 1)

    desired_negative_test = requested_test_groups - desired_positive_test
    if desired_negative_test < 1:
        desired_negative_test = 1
        desired_positive_test = requested_test_groups - desired_negative_test
    if desired_negative_test > negative_group_count - 1:
        desired_negative_test = negative_group_count - 1
        desired_positive_test = requested_test_groups - desired_negative_test

    if desired_positive_test < 1 or desired_negative_test < 1:
        raise ValueError("Could not allocate positive and negative groups to the test split.")
    if desired_positive_test >= positive_group_count or desired_negative_test >= negative_group_count:
        raise ValueError("Test split would consume all groups of one class.")

    rng = np.random.RandomState(random_state)
    rng.shuffle(positive_groups)
    rng.shuffle(negative_groups)
    test_groups = positive_groups[:desired_positive_test] + negative_groups[:desired_negative_test]
    train_groups = positive_groups[desired_positive_test:] + negative_groups[desired_negative_test:]

    train_mask = group_ids.isin(set(train_groups)).to_numpy()
    test_mask = group_ids.isin(set(test_groups)).to_numpy()
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)

    if y.iloc[train_idx].sum() == 0 or y.iloc[test_idx].sum() == 0:
        raise ValueError("Grouped split produced a train or test set with no positives.")

    return train_idx, test_idx, group_df


def select_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    return select_threshold_with_objective(y_true=y_true, scores=scores, objective="row_f1")


def build_threshold_candidates(scores: np.ndarray, max_candidates: int = MAX_THRESHOLD_CANDIDATES) -> np.ndarray:
    finite_scores = np.asarray(scores, dtype="float64")
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        return np.array([0.5], dtype="float64")
    unique_scores = np.unique(finite_scores)
    if unique_scores.size > max_candidates:
        sample_idx = np.linspace(0, unique_scores.size - 1, max_candidates).astype(int)
        unique_scores = unique_scores[sample_idx]
    return np.unique(np.concatenate([np.array([0.5], dtype="float64"), unique_scores]))


def build_group_score_frame(
    y_true: pd.Series,
    scores: np.ndarray,
    group_ids: pd.Series,
) -> pd.DataFrame:
    grouped = (
        pd.DataFrame(
            {
                "group_id": group_ids.astype(str),
                "label": y_true.astype(int),
                "score": scores,
            }
        )
        .groupby("group_id", as_index=False)
        .agg(
            true_positive_rows=("label", "sum"),
            row_count=("label", "size"),
            max_score=("score", "max"),
            mean_score=("score", "mean"),
        )
    )
    grouped["true_label"] = (grouped["true_positive_rows"] > 0).astype(int)
    return grouped


def select_threshold_with_objective(
    y_true: pd.Series,
    scores: np.ndarray,
    objective: str,
    group_ids: pd.Series | None = None,
) -> float:
    if y_true.nunique() < 2:
        return 0.5

    if objective == "row_f1":
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if thresholds.size == 0:
            return 0.5
        f1_values = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best_index = int(np.nanargmax(f1_values))
        return float(thresholds[best_index])

    if objective != "scenario_f1":
        raise ValueError(f"Unsupported threshold objective '{objective}'.")
    if group_ids is None:
        raise ValueError("Scenario threshold optimization requires group_ids.")

    grouped = build_group_score_frame(y_true=y_true, scores=scores, group_ids=group_ids)
    if grouped["true_label"].nunique() < 2:
        return 0.5

    candidates = build_threshold_candidates(grouped["max_score"].to_numpy())
    best_threshold = 0.5
    best_f1 = -1.0
    best_recall = -1.0
    for threshold in candidates:
        predicted = (grouped["max_score"] >= threshold).astype(int)
        f1_value = float(f1_score(grouped["true_label"], predicted, zero_division=0))
        positives = grouped["true_label"].to_numpy(dtype=int)
        predicted_np = predicted.to_numpy(dtype=int)
        true_positive = int(((positives == 1) & (predicted_np == 1)).sum())
        false_negative = int(((positives == 1) & (predicted_np == 0)).sum())
        recall_value = true_positive / max(true_positive + false_negative, 1)
        if (
            f1_value > best_f1 + 1e-12
            or (abs(f1_value - best_f1) <= 1e-12 and recall_value > best_recall + 1e-12)
            or (
                abs(f1_value - best_f1) <= 1e-12
                and abs(recall_value - best_recall) <= 1e-12
                and threshold < best_threshold
            )
        ):
            best_threshold = float(threshold)
            best_f1 = f1_value
            best_recall = recall_value
    return best_threshold


def resolve_count(total_rows: int, requested_size: float, minimum: int, maximum: int) -> int:
    if 0 < requested_size < 1:
        count = int(round(total_rows * requested_size))
    else:
        count = int(requested_size)
    return min(max(count, minimum), maximum)


def chronological_split(
    y: pd.Series,
    time_values: pd.Series,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = time_values.notna().to_numpy()
    if not valid_mask.any():
        raise ValueError("No valid timestamps were available for chronological splitting.")

    valid_idx = np.flatnonzero(valid_mask)
    ordered_idx = valid_idx[np.argsort(time_values.iloc[valid_idx].to_numpy(), kind="mergesort")]
    total_rows = len(ordered_idx)
    if total_rows < 3:
        raise ValueError("Need at least three timestamped rows for train/validation/test time splits.")

    test_count = resolve_count(total_rows, test_size, minimum=1, maximum=total_rows - 2)
    remaining = total_rows - test_count
    val_count = resolve_count(remaining, val_size, minimum=1, maximum=remaining - 1)
    train_count = total_rows - test_count - val_count
    if train_count < 1:
        raise ValueError("Chronological split did not leave any training rows.")

    train_idx = ordered_idx[:train_count]
    val_idx = ordered_idx[train_count : train_count + val_count]
    test_idx = ordered_idx[train_count + val_count :]

    if y.iloc[train_idx].sum() == 0 or y.iloc[val_idx].sum() == 0 or y.iloc[test_idx].sum() == 0:
        raise ValueError(
            "Chronological split produced a train, validation, or test segment with no positives. "
            "Use a different time window or regenerate features with a broader sample."
        )

    return train_idx, val_idx, test_idx


def evaluate_model(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "positive_rows": int(y_true.sum()),
        "predicted_positive_rows": int(predictions.sum()),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(y_true, predictions, zero_division=0),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_grouped_predictions(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    group_ids: pd.Series,
    group_label: str = "scenario",
) -> dict:
    grouped = build_group_score_frame(y_true=y_true, scores=scores, group_ids=group_ids)
    grouped["predicted_label"] = (grouped["max_score"] >= threshold).astype(int)

    y_group = grouped["true_label"]
    pred_group = grouped["predicted_label"]
    metrics = {
        "group_label": group_label,
        f"positive_{group_label}s": int(y_group.sum()),
        f"predicted_positive_{group_label}s": int(pred_group.sum()),
        f"{group_label}_f1": float(f1_score(y_group, pred_group, zero_division=0)),
        f"{group_label}_balanced_accuracy": float(balanced_accuracy_score(y_group, pred_group)),
        f"{group_label}_confusion_matrix": confusion_matrix(y_group, pred_group).tolist(),
        f"{group_label}_classification_report": classification_report(y_group, pred_group, zero_division=0),
    }
    if y_group.nunique() > 1:
        metrics[f"{group_label}_roc_auc"] = float(roc_auc_score(y_group, grouped["max_score"]))
        metrics[f"{group_label}_average_precision"] = float(average_precision_score(y_group, grouped["max_score"]))
    else:
        metrics[f"{group_label}_roc_auc"] = None
        metrics[f"{group_label}_average_precision"] = None
    return metrics


def choose_feature_columns(
    df: pd.DataFrame,
    label_column: str,
    include_expert_features: bool,
    keep_identity_features: bool,
    extra_drop_columns: Iterable[str],
) -> Tuple[List[str], List[str]]:
    excluded_columns = {label_column}
    excluded_columns.update(column for column in DEFAULT_METADATA_COLUMNS if column in df.columns)
    excluded_columns.update(column for column in DEFAULT_SCENARIO_COLUMNS if column in df.columns)
    excluded_columns.update(
        column
        for column in df.columns
        if column != label_column and (column == "is_anomaly" or column.endswith("_is_anomaly"))
    )
    if not include_expert_features:
        excluded_columns.update(column for column in DEFAULT_EXPERT_COLUMNS if column in df.columns)
    if not keep_identity_features:
        excluded_columns.update(column for column in DEFAULT_IDENTITY_COLUMNS if column in df.columns)
    excluded_columns.update(column for column in extra_drop_columns if column in df.columns)

    feature_columns = [column for column in df.columns if column not in excluded_columns]
    if not feature_columns:
        raise ValueError("No training features remain after applying exclusion rules.")
    return feature_columns, sorted(excluded_columns)


def sanitize_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = (
        frame.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return sanitized.mask(sanitized.abs() > MAX_ABS_FEATURE_VALUE, 0.0)


def train_fusion_model(args: argparse.Namespace) -> None:
    print(f"Loading features from {args.feature_csv}")
    df = pd.read_csv(args.feature_csv, low_memory=False)
    resolved_label_column = resolve_label_column(df, args.label_column)
    df = df.dropna(subset=[resolved_label_column]).copy()
    df[resolved_label_column] = df[resolved_label_column].astype(int)
    feature_columns, excluded_columns = choose_feature_columns(
        df=df,
        label_column=resolved_label_column,
        include_expert_features=args.include_expert_features,
        keep_identity_features=args.keep_identity_features,
        extra_drop_columns=parse_column_list(args.extra_drop_columns),
    )

    X = sanitize_numeric_frame(df[feature_columns])
    y = df[resolved_label_column]

    print(f"Dataset shape: {X.shape}")
    print(f"Resolved label column: {resolved_label_column}")
    print(f"Positive rows: {int(y.sum())} ({y.mean() * 100:.4f}%)")
    print(f"Split mode: {args.split_mode}")
    print(f"Training features: {feature_columns}")
    print(f"Excluded columns: {excluded_columns}")
    split_summary = {}
    if args.split_mode == "group":
        group_columns = parse_column_list(args.group_columns)
        if not group_columns:
            raise ValueError("At least one group column is required for group-aware evaluation.")
        print(f"Group columns: {group_columns}")

        group_ids = build_group_ids(df, group_columns)
        train_full_idx, test_idx, group_df = stratified_group_split(
            y=y,
            group_ids=group_ids,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        train_group_ids = group_ids.iloc[train_full_idx].reset_index(drop=True)
        X_train_full = X.iloc[train_full_idx].reset_index(drop=True)
        y_train_full = y.iloc[train_full_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        train_inner_idx, val_inner_idx, _ = stratified_group_split(
            y=y_train_full,
            group_ids=train_group_ids,
            test_size=args.val_size,
            random_state=args.random_state + 1,
        )
        train_idx = train_full_idx[train_inner_idx]
        val_idx = train_full_idx[val_inner_idx]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        split_summary = {
            "group_columns": group_columns,
            "unique_groups": int(len(group_df)),
            "positive_groups": int(group_df["group_label"].sum()),
        }
        print("Training Random Forest with grouped train/validation/test splits...")
    elif args.split_mode == "scenario":
        if args.scenario_column not in df.columns:
            raise ValueError(
                f"Scenario split requested but scenario column '{args.scenario_column}' was not found in the feature CSV."
            )
        scenario_ids = build_scenario_group_ids(df, args.scenario_column)
        print(f"Scenario column: {args.scenario_column}")

        train_full_idx, test_idx, scenario_df = stratified_group_split(
            y=y,
            group_ids=scenario_ids,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_scenario_ids = scenario_ids.iloc[train_full_idx].reset_index(drop=True)
        X_train_full = X.iloc[train_full_idx].reset_index(drop=True)
        y_train_full = y.iloc[train_full_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        train_inner_idx, val_inner_idx, _ = stratified_group_split(
            y=y_train_full,
            group_ids=train_scenario_ids,
            test_size=args.val_size,
            random_state=args.random_state + 1,
        )
        train_idx = train_full_idx[train_inner_idx]
        val_idx = train_full_idx[val_inner_idx]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        split_summary = {
            "scenario_column": args.scenario_column,
            "unique_scenarios": int(len(scenario_df)),
            "positive_scenarios": int(scenario_df["group_label"].sum()),
        }
        print("Training Random Forest with scenario-based train/validation/test splits...")
    else:
        time_values = build_time_order(df, args.time_column)
        print(f"Time column: {args.time_column}")
        train_idx, val_idx, test_idx = chronological_split(
            y=y,
            time_values=time_values,
            test_size=args.test_size,
            val_size=args.val_size,
        )

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        X_train_full = pd.concat([X_train, X_val], ignore_index=True)
        y_train_full = pd.concat([y_train, y_val], ignore_index=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        split_summary = {
            "time_column": args.time_column,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        }
        print("Training Random Forest with chronological train/validation/test splits...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    val_scores = clf.predict_proba(X_val)[:, 1]
    val_group_ids = None
    if args.threshold_objective == "scenario_f1":
        if args.scenario_column not in df.columns:
            raise ValueError("Scenario threshold optimization requested but scenario metadata is unavailable.")
        val_group_ids = build_scenario_group_ids(df.iloc[val_idx].reset_index(drop=True), args.scenario_column)
    threshold = select_threshold_with_objective(
        y_true=y_val,
        scores=val_scores,
        objective=args.threshold_objective,
        group_ids=val_group_ids,
    )
    val_metrics = evaluate_model(y_val, val_scores, threshold)
    if args.scenario_column in df.columns:
        if val_group_ids is None:
            val_group_ids = build_scenario_group_ids(df.iloc[val_idx].reset_index(drop=True), args.scenario_column)
        val_metrics["scenario_metrics"] = evaluate_grouped_predictions(
            y_true=y_val,
            scores=val_scores,
            threshold=threshold,
            group_ids=val_group_ids,
            group_label="scenario",
        )

    print(f"Validation threshold selected by {args.threshold_objective}: {threshold:.4f}")
    print("Refitting on the full training groups with the same hyperparameters...")
    clf.fit(X_train_full, y_train_full)

    test_scores = clf.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, test_scores, threshold)
    if args.scenario_column in df.columns:
        test_group_ids = build_scenario_group_ids(df.iloc[test_idx].reset_index(drop=True), args.scenario_column)
        test_metrics["scenario_metrics"] = evaluate_grouped_predictions(
            y_true=y_test,
            scores=test_scores,
            threshold=threshold,
            group_ids=test_group_ids,
            group_label="scenario",
        )

    importances = (
        pd.DataFrame({"feature": feature_columns, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nValidation report:")
    print(val_metrics["classification_report"])
    print("Validation confusion matrix:")
    print(np.array(val_metrics["confusion_matrix"]))

    print("\nTest report:")
    print(test_metrics["classification_report"])
    print("Test confusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    print("\nTop features:")
    print(importances.head(10).to_string(index=False))

    model_bundle = {
        "model": clf,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "split_mode": args.split_mode,
        "threshold_objective": args.threshold_objective,
        "split_summary": split_summary,
        "include_expert_features": args.include_expert_features,
        "keep_identity_features": args.keep_identity_features,
        "label_column": resolved_label_column,
        "metrics": {"validation": val_metrics, "test": test_metrics},
    }
    print(f"\nSaving model bundle to {args.model_path}")
    joblib.dump(model_bundle, args.model_path)

    report_lines = [
        "MMS Fusion ML Evaluation",
        "========================",
        "",
        f"Dataset: {args.feature_csv}",
        f"Label column: {resolved_label_column}",
        f"Rows: {len(df)}",
        f"Positive rows: {int(y.sum())} ({y.mean() * 100:.4f}%)",
        f"Split mode: {args.split_mode}",
        f"Threshold objective: {args.threshold_objective}",
        f"Training features: {', '.join(feature_columns)}",
        f"Excluded columns: {', '.join(excluded_columns)}",
        f"Expert features included: {args.include_expert_features}",
        f"Identity features included: {args.keep_identity_features}",
        f"Split summary: {json.dumps(split_summary)}",
        "",
        "Leakage note:",
        (
            "protocol_score/stat_score are excluded by default because they may encode the "
            "same upstream hybrid detector that produced the target."
        ),
        "",
        "Validation metrics:",
        json.dumps({k: v for k, v in val_metrics.items() if k != "classification_report"}, indent=2),
        "",
        "Validation classification report:",
        val_metrics["classification_report"],
        "",
        "Validation scenario metrics:",
        json.dumps(val_metrics.get("scenario_metrics", {}), indent=2),
        "",
        "Test metrics:",
        json.dumps({k: v for k, v in test_metrics.items() if k != "classification_report"}, indent=2),
        "",
        "Test classification report:",
        test_metrics["classification_report"],
        "",
        "Test scenario metrics:",
        json.dumps(test_metrics.get("scenario_metrics", {}), indent=2),
        "",
        "Feature importances:",
        importances.to_string(index=False),
    ]

    with open(args.report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    print(f"Report saved to {args.report_path}")
    return model_bundle


if __name__ == "__main__":
    train_fusion_model(parse_args())
