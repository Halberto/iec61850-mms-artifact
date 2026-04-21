import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

import train_fusion_ml as fusion_train
import train_sequence_branch as sequence_train


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fusion combiner using out-of-fold branch predictions. "
            "Base branches produce OOF scores on the meta-train segment, then the combiner is fit on those "
            "scores and evaluated on a held-out test split."
        )
    )
    parser.add_argument("--feature-csv", default=str(PROCESSED_DATA_DIR / "mms_ml_features_100k.csv"))
    parser.add_argument("--sequence-csv", default=str(PROCESSED_DATA_DIR / "mms_sequence_windows.csv"))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "mms_meta_fusion_model.joblib"))
    parser.add_argument("--report-path", default=str(RESULTS_DIR / "meta_fusion_report.txt"))
    parser.add_argument("--predictions-csv", default=str(RESULTS_DIR / "meta_fusion_predictions.csv"))
    parser.add_argument("--label-column", default="supervised_is_anomaly")
    parser.add_argument("--split-mode", choices=("time", "scenario"), default="time")
    parser.add_argument("--time-column", default="window_end_time_unix")
    parser.add_argument("--scenario-column", default="scenario_id")
    parser.add_argument("--event-key-column", default="line_number")
    parser.add_argument("--sequence-end-key-column", default="window_end_line_number")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--oof-splits", type=int, default=4)
    parser.add_argument(
        "--threshold-objective",
        choices=("row_f1", "scenario_f1"),
        default="row_f1",
        help="Metric used to select the meta-fusion threshold on OOF validation scores.",
    )
    parser.add_argument("--tabular-n-estimators", type=int, default=300)
    parser.add_argument("--tabular-max-depth", type=int, default=12)
    parser.add_argument("--sequence-n-estimators", type=int, default=300)
    parser.add_argument("--sequence-max-depth", type=int, default=14)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    return parser.parse_args()


def fit_branch_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def build_meta_features(
    tabular_scores: np.ndarray,
    sequence_scores: np.ndarray,
    rule_frame: pd.DataFrame,
) -> pd.DataFrame:
    meta_df = pd.DataFrame(
        {
            "tabular_score": tabular_scores,
            "sequence_score": sequence_scores,
            "protocol_score": rule_frame["protocol_score"].to_numpy(),
            "stat_score": rule_frame["stat_score"].to_numpy(),
        }
    )
    meta_df["rule_max_score"] = meta_df[["protocol_score", "stat_score"]].max(axis=1)
    meta_df["rule_sum_score"] = meta_df["protocol_score"] + meta_df["stat_score"]
    return meta_df


def chronological_train_test_split(
    y: pd.Series,
    time_values: pd.Series,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = time_values.notna().to_numpy()
    if not valid_mask.any():
        raise ValueError("No valid timestamps available for chronological train/test splitting.")

    valid_idx = np.flatnonzero(valid_mask)
    ordered_idx = valid_idx[np.argsort(time_values.iloc[valid_idx].to_numpy(), kind="mergesort")]
    if len(ordered_idx) < 3:
        raise ValueError("Need at least three timestamped rows for chronological train/test splitting.")

    test_count = fusion_train.resolve_count(len(ordered_idx), test_size, minimum=1, maximum=len(ordered_idx) - 1)
    train_idx = ordered_idx[:-test_count]
    test_idx = ordered_idx[-test_count:]

    if y.iloc[train_idx].sum() == 0 or y.iloc[test_idx].sum() == 0:
        raise ValueError("Chronological train/test split produced a train or test segment with no positives.")
    return train_idx, test_idx


def scenario_train_test_split(
    y: pd.Series,
    group_ids: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train_idx, test_idx, scenario_df = fusion_train.stratified_group_split(
        y=y,
        group_ids=group_ids,
        test_size=test_size,
        random_state=random_state,
    )
    return train_idx, test_idx, scenario_df


def iter_time_folds(
    train_indices: np.ndarray,
    time_values: pd.Series,
    n_splits: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    ordered = train_indices[np.argsort(time_values.iloc[train_indices].to_numpy(), kind="mergesort")]
    max_splits = min(n_splits, len(ordered) - 1)
    if max_splits < 2:
        raise ValueError("Not enough rows to create time-based OOF folds.")
    splitter = TimeSeriesSplit(n_splits=max_splits)
    for fit_idx, pred_idx in splitter.split(ordered):
        yield ordered[fit_idx], ordered[pred_idx]


def iter_scenario_folds(
    train_indices: np.ndarray,
    group_ids: pd.Series,
    n_splits: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    groups = group_ids.iloc[train_indices].astype(str)
    unique_groups = groups.nunique()
    max_splits = min(n_splits, unique_groups)
    if max_splits < 2:
        raise ValueError("Not enough distinct scenarios to create OOF folds.")
    splitter = GroupKFold(n_splits=max_splits)
    local_X = np.zeros(len(train_indices))
    local_y = np.zeros(len(train_indices))
    for fit_local, pred_local in splitter.split(local_X, local_y, groups=groups):
        yield train_indices[fit_local], train_indices[pred_local]


def generate_oof_predictions(
    X_tabular: pd.DataFrame,
    X_sequence: pd.DataFrame,
    y: pd.Series,
    train_indices: np.ndarray,
    fold_iterator: Iterable[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict]:
    oof_tabular = pd.Series(np.nan, index=train_indices, dtype="float64")
    oof_sequence = pd.Series(np.nan, index=train_indices, dtype="float64")
    fold_records: list[dict] = []

    for fold_number, (fit_idx, pred_idx) in enumerate(fold_iterator, start=1):
        y_fit = y.iloc[fit_idx]
        y_pred = y.iloc[pred_idx]
        if y_fit.sum() == 0 or y_pred.sum() == 0:
            continue
        if y_fit.nunique() < 2 or y_pred.nunique() < 2:
            continue

        tabular_model = fit_branch_model(
            X_train=X_tabular.iloc[fit_idx],
            y_train=y_fit,
            n_estimators=args.tabular_n_estimators,
            max_depth=args.tabular_max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state + fold_number,
        )
        sequence_model = fit_branch_model(
            X_train=X_sequence.iloc[fit_idx],
            y_train=y_fit,
            n_estimators=args.sequence_n_estimators,
            max_depth=args.sequence_max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state + 100 + fold_number,
        )

        oof_tabular.loc[pred_idx] = tabular_model.predict_proba(X_tabular.iloc[pred_idx])[:, 1]
        oof_sequence.loc[pred_idx] = sequence_model.predict_proba(X_sequence.iloc[pred_idx])[:, 1]
        fold_records.append(
            {
                "fold": fold_number,
                "train_rows": int(len(fit_idx)),
                "validation_rows": int(len(pred_idx)),
                "validation_positives": int(y_pred.sum()),
            }
        )

    valid_mask = oof_tabular.notna() & oof_sequence.notna()
    if not valid_mask.any():
        raise ValueError("OOF generation produced no valid prediction rows.")

    valid_indices = oof_tabular.loc[valid_mask].index.to_numpy()
    meta_train_df = pd.DataFrame(
        {
            "tabular_score": oof_tabular.loc[valid_mask].to_numpy(),
            "sequence_score": oof_sequence.loc[valid_mask].to_numpy(),
        },
        index=valid_indices,
    )
    meta_train_y = y.iloc[meta_train_df.index].reset_index(drop=True)
    meta_train_df = meta_train_df.reset_index(drop=True)
    return meta_train_df, meta_train_y, valid_indices, {"folds": fold_records, "oof_rows": int(valid_mask.sum())}


def train_meta_fusion(args: argparse.Namespace) -> dict:
    print(f"Loading event features from {args.feature_csv}")
    event_df = pd.read_csv(args.feature_csv, low_memory=False)
    event_label_column = fusion_train.resolve_label_column(event_df, args.label_column)

    print(f"Loading sequence windows from {args.sequence_csv}")
    sequence_df = pd.read_csv(args.sequence_csv, low_memory=False)
    sequence_label_column = fusion_train.resolve_label_column(sequence_df, args.label_column)

    if args.event_key_column not in event_df.columns:
        raise ValueError(f"Event key column '{args.event_key_column}' not found in event feature CSV.")
    if args.sequence_end_key_column not in sequence_df.columns:
        raise ValueError(f"Sequence key column '{args.sequence_end_key_column}' not found in sequence CSV.")
    if args.time_column not in sequence_df.columns:
        raise ValueError(f"Time column '{args.time_column}' not found in sequence CSV.")
    for column in ("protocol_score", "stat_score"):
        if column not in event_df.columns:
            raise ValueError(f"Required rule feature '{column}' not found in event feature CSV.")

    event_feature_columns, _ = fusion_train.choose_feature_columns(
        df=event_df,
        label_column=event_label_column,
        include_expert_features=False,
        keep_identity_features=False,
        extra_drop_columns=[],
    )
    sequence_feature_columns, _ = sequence_train.choose_sequence_features(sequence_df, sequence_label_column)

    event_keep_columns = [
        args.event_key_column,
        event_label_column,
        "protocol_score",
        "stat_score",
        args.scenario_column,
    ] + event_feature_columns
    event_keep_columns = [column for column in dict.fromkeys(event_keep_columns) if column in event_df.columns]
    merged = sequence_df.merge(
        event_df[event_keep_columns],
        left_on=args.sequence_end_key_column,
        right_on=args.event_key_column,
        how="inner",
        suffixes=("", "_event"),
    )
    merged = merged.dropna(subset=[sequence_label_column]).copy()
    merged[sequence_label_column] = merged[sequence_label_column].astype(int)

    mismatch_count = 0
    if event_label_column in merged.columns and event_label_column != sequence_label_column:
        mismatch_mask = (
            merged[event_label_column].notna()
            & (merged[event_label_column].astype(int) != merged[sequence_label_column])
        )
        mismatch_count = int(mismatch_mask.sum())

    print(f"Merged aligned branch dataset shape: {merged.shape}")
    print(f"Sequence label column: {sequence_label_column}")
    print(f"Tabular label column: {event_label_column}")
    print(f"Label mismatches between event/sequence rows: {mismatch_count}")

    y = merged[sequence_label_column]
    time_values = fusion_train.build_time_order(merged, args.time_column)
    if args.split_mode == "scenario":
        if args.scenario_column not in merged.columns:
            raise ValueError(f"Scenario split requested but '{args.scenario_column}' is missing in aligned data.")
        scenario_group_ids = fusion_train.build_scenario_group_ids(merged, args.scenario_column)
        meta_train_idx, test_idx, scenario_df = scenario_train_test_split(
            y=y,
            group_ids=scenario_group_ids,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        fold_iterator = iter_scenario_folds(
            train_indices=meta_train_idx,
            group_ids=scenario_group_ids,
            n_splits=args.oof_splits,
        )
        split_summary = {
            "split_mode": "scenario",
            "scenario_column": args.scenario_column,
            "unique_scenarios": int(len(scenario_df)),
            "positive_scenarios": int(scenario_df["group_label"].sum()),
            "train_rows": int(len(meta_train_idx)),
            "test_rows": int(len(test_idx)),
        }
    else:
        meta_train_idx, test_idx = chronological_train_test_split(
            y=y,
            time_values=time_values,
            test_size=args.test_size,
        )
        fold_iterator = iter_time_folds(
            train_indices=meta_train_idx,
            time_values=time_values,
            n_splits=args.oof_splits,
        )
        split_summary = {
            "split_mode": "time",
            "time_column": args.time_column,
            "train_rows": int(len(meta_train_idx)),
            "test_rows": int(len(test_idx)),
        }

    X_tabular = fusion_train.sanitize_numeric_frame(merged[event_feature_columns])
    X_sequence = fusion_train.sanitize_numeric_frame(merged[sequence_feature_columns])
    X_rules = fusion_train.sanitize_numeric_frame(merged[["protocol_score", "stat_score"]])

    print("Generating out-of-fold branch predictions on the meta-train segment...")
    meta_oof_df, meta_oof_y, meta_oof_indices, oof_summary = generate_oof_predictions(
        X_tabular=X_tabular,
        X_sequence=X_sequence,
        y=y,
        train_indices=meta_train_idx,
        fold_iterator=fold_iterator,
        args=args,
    )

    rule_meta_df = build_meta_features(
        tabular_scores=meta_oof_df["tabular_score"].to_numpy(),
        sequence_scores=meta_oof_df["sequence_score"].to_numpy(),
        rule_frame=X_rules.iloc[meta_oof_indices].reset_index(drop=True),
    )
    meta_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=args.random_state,
    )
    print("Training meta-fusion combiner on OOF branch predictions...")
    meta_model.fit(rule_meta_df, meta_oof_y)
    meta_train_scores = meta_model.predict_proba(rule_meta_df)[:, 1]
    meta_train_group_ids = None
    if args.threshold_objective == "scenario_f1":
        if args.scenario_column not in merged.columns:
            raise ValueError("Scenario threshold optimization requested but scenario metadata is unavailable.")
        meta_train_group_ids = fusion_train.build_scenario_group_ids(
            merged.iloc[meta_oof_indices].reset_index(drop=True),
            args.scenario_column,
        )
    threshold = fusion_train.select_threshold_with_objective(
        y_true=meta_oof_y,
        scores=meta_train_scores,
        objective=args.threshold_objective,
        group_ids=meta_train_group_ids,
    )
    meta_train_metrics = fusion_train.evaluate_model(meta_oof_y, meta_train_scores, threshold)
    if args.scenario_column in merged.columns:
        if meta_train_group_ids is None:
            meta_train_group_ids = fusion_train.build_scenario_group_ids(
                merged.iloc[meta_oof_indices].reset_index(drop=True),
                args.scenario_column,
            )
        meta_train_metrics["scenario_metrics"] = fusion_train.evaluate_grouped_predictions(
            y_true=meta_oof_y,
            scores=meta_train_scores,
            threshold=threshold,
            group_ids=meta_train_group_ids,
            group_label="scenario",
        )

    print("Training final tabular and sequence branches on the full meta-train segment...")
    tabular_model = fit_branch_model(
        X_train=X_tabular.iloc[meta_train_idx],
        y_train=y.iloc[meta_train_idx],
        n_estimators=args.tabular_n_estimators,
        max_depth=args.tabular_max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    sequence_model = fit_branch_model(
        X_train=X_sequence.iloc[meta_train_idx],
        y_train=y.iloc[meta_train_idx],
        n_estimators=args.sequence_n_estimators,
        max_depth=args.sequence_max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state + 1,
    )

    tabular_test_scores = tabular_model.predict_proba(X_tabular.iloc[test_idx])[:, 1]
    sequence_test_scores = sequence_model.predict_proba(X_sequence.iloc[test_idx])[:, 1]
    meta_test_df = build_meta_features(
        tabular_scores=tabular_test_scores,
        sequence_scores=sequence_test_scores,
        rule_frame=X_rules.iloc[test_idx].reset_index(drop=True),
    )
    y_test = y.iloc[test_idx].reset_index(drop=True)
    test_meta_scores = meta_model.predict_proba(meta_test_df)[:, 1]

    test_metrics = fusion_train.evaluate_model(y_test, test_meta_scores, threshold)
    if args.scenario_column in merged.columns:
        test_group_ids = fusion_train.build_scenario_group_ids(
            merged.iloc[test_idx].reset_index(drop=True),
            args.scenario_column,
        )
        test_metrics["scenario_metrics"] = fusion_train.evaluate_grouped_predictions(
            y_true=y_test,
            scores=test_meta_scores,
            threshold=threshold,
            group_ids=test_group_ids,
            group_label="scenario",
        )
    tabular_test_metrics = fusion_train.evaluate_model(y_test, tabular_test_scores, 0.5)
    sequence_test_metrics = fusion_train.evaluate_model(y_test, sequence_test_scores, 0.5)
    rules_test_metrics = fusion_train.evaluate_model(y_test, meta_test_df["rule_max_score"].to_numpy(), 0.5)

    meta_coefficients = (
        pd.DataFrame({"feature": rule_meta_df.columns, "coefficient": meta_model.coef_[0]})
        .sort_values("coefficient", key=lambda col: np.abs(col), ascending=False)
        .reset_index(drop=True)
    )

    predictions_df = merged.iloc[test_idx][
        [args.sequence_end_key_column, args.time_column, sequence_label_column]
    ].reset_index(drop=True)
    if args.scenario_column in merged.columns:
        predictions_df[args.scenario_column] = merged.iloc[test_idx][args.scenario_column].reset_index(drop=True)
    predictions_df["tabular_score"] = tabular_test_scores
    predictions_df["sequence_score"] = sequence_test_scores
    predictions_df["protocol_score"] = meta_test_df["protocol_score"]
    predictions_df["stat_score"] = meta_test_df["stat_score"]
    predictions_df["meta_fusion_score"] = test_meta_scores
    predictions_df["meta_prediction"] = (test_meta_scores >= threshold).astype(int)
    predictions_df.to_csv(args.predictions_csv, index=False)

    print("\nMeta-fusion OOF-train report:")
    print(meta_train_metrics["classification_report"])
    print("Meta-fusion test report:")
    print(test_metrics["classification_report"])
    print("Meta-fusion test confusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    print("\nMeta combiner coefficients:")
    print(meta_coefficients.to_string(index=False))

    model_bundle = {
        "tabular_model": tabular_model,
        "sequence_model": sequence_model,
        "meta_model": meta_model,
        "label_column": sequence_label_column,
        "event_label_column": event_label_column,
        "event_feature_columns": event_feature_columns,
        "sequence_feature_columns": sequence_feature_columns,
        "meta_feature_columns": list(rule_meta_df.columns),
        "threshold": threshold,
        "threshold_objective": args.threshold_objective,
        "split_summary": {
            **split_summary,
            "label_mismatch_count": mismatch_count,
            **oof_summary,
        },
        "metrics": {
            "oof_train": meta_train_metrics,
            "test": test_metrics,
            "tabular_test": tabular_test_metrics,
            "sequence_test": sequence_test_metrics,
            "rules_test": rules_test_metrics,
        },
    }
    joblib.dump(model_bundle, args.model_path)

    report_lines = [
        "MMS Meta Fusion Evaluation (OOF)",
        "===============================",
        "",
        f"Event dataset: {args.feature_csv}",
        f"Sequence dataset: {args.sequence_csv}",
        f"Sequence label column: {sequence_label_column}",
        f"Tabular label column: {event_label_column}",
        f"Aligned rows: {len(merged)}",
        f"Positive rows: {int(y.sum())} ({y.mean() * 100:.4f}%)",
        f"Threshold objective: {args.threshold_objective}",
        f"Split summary: {json.dumps(model_bundle['split_summary'])}",
        "",
        "OOF meta-train metrics:",
        json.dumps({k: v for k, v in meta_train_metrics.items() if k != "classification_report"}, indent=2),
        "",
        "OOF meta-train classification report:",
        meta_train_metrics["classification_report"],
        "",
        "OOF meta-train scenario metrics:",
        json.dumps(meta_train_metrics.get("scenario_metrics", {}), indent=2),
        "",
        "Meta-fusion test metrics:",
        json.dumps({k: v for k, v in test_metrics.items() if k != "classification_report"}, indent=2),
        "",
        "Meta-fusion test classification report:",
        test_metrics["classification_report"],
        "",
        "Meta-fusion test scenario metrics:",
        json.dumps(test_metrics.get("scenario_metrics", {}), indent=2),
        "",
        "Branch test metrics:",
        json.dumps(
            {
                "tabular_test_f1": tabular_test_metrics["f1"],
                "sequence_test_f1": sequence_test_metrics["f1"],
                "rules_test_f1": rules_test_metrics["f1"],
            },
            indent=2,
        ),
        "",
        "Meta combiner coefficients:",
        meta_coefficients.to_string(index=False),
    ]
    with open(args.report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    print(f"Saved meta-fusion model bundle to {args.model_path}")
    print(f"Saved meta-fusion report to {args.report_path}")
    print(f"Saved meta-fusion predictions to {args.predictions_csv}")
    return model_bundle


if __name__ == "__main__":
    train_meta_fusion(parse_args())
