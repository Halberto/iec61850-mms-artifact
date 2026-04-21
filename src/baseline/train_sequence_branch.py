import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import train_fusion_ml as fusion_train

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


SEQUENCE_METADATA_COLUMNS = (
    "window_group",
    "window_start_index",
    "window_end_index",
    "window_size",
    "window_positive_count",
    "window_start_line_number",
    "window_end_line_number",
    "window_end_timestamp",
    "window_end_time_unix",
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
)
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


class GRUSequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        pooled = self.dropout(output[:, -1, :])
        return self.head(pooled).squeeze(1)


class TCNSequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.network(x.transpose(1, 2))
        pooled = hidden.mean(dim=2)
        return self.head(pooled).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a sequence branch over flattened MMS windows. "
            "Supports RF, GRU, and TCN sequence models."
        )
    )
    parser.add_argument("--sequence-csv", default=str(PROCESSED_DATA_DIR / "mms_sequence_windows.csv"))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "mms_sequence_branch_model.joblib"))
    parser.add_argument("--report-path", default=str(RESULTS_DIR / "sequence_branch_report.txt"))
    parser.add_argument(
        "--predictions-csv",
        default=str(RESULTS_DIR / "sequence_branch_predictions.csv"),
    )
    parser.add_argument("--label-column", default="supervised_is_anomaly")
    parser.add_argument("--model-type", choices=("gru", "tcn", "rf"), default="gru")
    parser.add_argument("--split-mode", choices=("time", "group", "scenario"), default="time")
    parser.add_argument("--time-column", default="window_end_time_unix")
    parser.add_argument("--group-column", default="window_group")
    parser.add_argument("--scenario-column", default="scenario_id")
    parser.add_argument(
        "--threshold-objective",
        choices=("row_f1", "scenario_f1"),
        default="row_f1",
        help="Metric used to choose the validation threshold and best checkpoint.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=14)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    return parser.parse_args()


def choose_sequence_features(df: pd.DataFrame, label_column: str) -> tuple[list[str], list[str]]:
    excluded_columns = {label_column}
    excluded_columns.update(column for column in SEQUENCE_METADATA_COLUMNS if column in df.columns)
    excluded_columns.update(
        column
        for column in df.columns
        if column != label_column and (column == "is_anomaly" or column.endswith("_is_anomaly"))
    )
    feature_columns = [column for column in df.columns if column not in excluded_columns]
    if not feature_columns:
        raise ValueError("No usable sequence features remain after excluding metadata and label columns.")
    return feature_columns, sorted(excluded_columns)


def split_sequence_dataset(
    df: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if args.split_mode == "time":
        time_values = fusion_train.build_time_order(df, args.time_column)
        print(f"Time column: {args.time_column}")
        train_idx, val_idx, test_idx = fusion_train.chronological_split(
            y=y,
            time_values=time_values,
            test_size=args.test_size,
            val_size=args.val_size,
        )
        split_summary = {
            "split_mode": "time",
            "time_column": args.time_column,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        }
        return train_idx, val_idx, test_idx, split_summary

    if args.split_mode == "group":
        if args.group_column not in df.columns:
            raise ValueError(f"Group column '{args.group_column}' not found in sequence CSV.")
        group_ids = df[args.group_column].astype(str)
        train_full_idx, test_idx, group_df = fusion_train.stratified_group_split(
            y=y,
            group_ids=group_ids,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        train_group_ids = group_ids.iloc[train_full_idx].reset_index(drop=True)
        y_train_full = y.iloc[train_full_idx].reset_index(drop=True)
        train_idx_inner, val_idx_inner, _ = fusion_train.stratified_group_split(
            y=y_train_full,
            group_ids=train_group_ids,
            test_size=args.val_size,
            random_state=args.random_state + 1,
        )
        train_idx = train_full_idx[train_idx_inner]
        val_idx = train_full_idx[val_idx_inner]
        split_summary = {
            "split_mode": "group",
            "group_column": args.group_column,
            "unique_groups": int(len(group_df)),
            "positive_groups": int(group_df["group_label"].sum()),
            "test_rows": int(len(test_idx)),
        }
        return train_idx, val_idx, test_idx, split_summary

    if args.scenario_column not in df.columns:
        raise ValueError(f"Scenario column '{args.scenario_column}' not found in sequence CSV.")
    scenario_ids = fusion_train.build_scenario_group_ids(df, args.scenario_column)
    train_full_idx, test_idx, scenario_df = fusion_train.stratified_group_split(
        y=y,
        group_ids=scenario_ids,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_scenario_ids = scenario_ids.iloc[train_full_idx].reset_index(drop=True)
    y_train_full = y.iloc[train_full_idx].reset_index(drop=True)
    train_idx_inner, val_idx_inner, _ = fusion_train.stratified_group_split(
        y=y_train_full,
        group_ids=train_scenario_ids,
        test_size=args.val_size,
        random_state=args.random_state + 1,
    )
    train_idx = train_full_idx[train_idx_inner]
    val_idx = train_full_idx[val_idx_inner]
    split_summary = {
        "split_mode": "scenario",
        "scenario_column": args.scenario_column,
        "unique_scenarios": int(len(scenario_df)),
        "positive_scenarios": int(scenario_df["group_label"].sum()),
        "test_rows": int(len(test_idx)),
    }
    return train_idx, val_idx, test_idx, split_summary


def parse_temporal_column(column_name: str) -> tuple[str, int]:
    if column_name.endswith("_t0"):
        return column_name[:-3], 0
    if "_t_minus_" not in column_name:
        raise ValueError(f"Unexpected sequence feature name '{column_name}'.")
    base_name, step_text = column_name.rsplit("_t_minus_", 1)
    return base_name, int(step_text)


def build_feature_layout(feature_columns: Sequence[str]) -> tuple[list[str], list[int]]:
    base_names: list[str] = []
    step_values: set[int] = set()
    for column in feature_columns:
        base_name, step_value = parse_temporal_column(column)
        if base_name not in base_names:
            base_names.append(base_name)
        step_values.add(step_value)
    ordered_steps = sorted(step_values, reverse=True)
    return base_names, ordered_steps


def build_sequence_tensor(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    base_names: Sequence[str],
    ordered_steps: Sequence[int],
) -> np.ndarray:
    base_index = {name: idx for idx, name in enumerate(base_names)}
    step_index = {step: idx for idx, step in enumerate(ordered_steps)}
    tensor = np.zeros((len(frame), len(ordered_steps), len(base_names)), dtype=np.float32)

    for column in feature_columns:
        base_name, step_value = parse_temporal_column(column)
        tensor[:, step_index[step_value], base_index[base_name]] = (
            pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        )
    return tensor


def compute_tensor_scaler(train_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_tensor.mean(axis=(0, 1), keepdims=True)
    std = train_tensor.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def scale_tensor(tensor: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((tensor - mean) / std).astype(np.float32)


def resolve_device(device_name: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch is required for GRU/TCN sequence training but is not installed.")
    if device_name == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_neural_model(args: argparse.Namespace, input_size: int) -> nn.Module:
    if args.model_type == "gru":
        return GRUSequenceClassifier(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if args.model_type == "tcn":
        return TCNSequenceClassifier(
            input_size=input_size,
            hidden_size=args.hidden_size,
            kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported neural model type '{args.model_type}'.")


def build_loader(
    X_tensor: np.ndarray,
    y_array: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X_tensor),
        torch.from_numpy(y_array.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        batch_size = int(batch_y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
    return total_loss / max(total_rows, 1)


def predict_neural_scores(
    model: nn.Module,
    X_tensor: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            batch = torch.from_numpy(X_tensor[start : start + batch_size]).to(device)
            logits = model(batch)
            scores = torch.sigmoid(logits).cpu().numpy()
            rows.append(scores)
    return np.concatenate(rows, axis=0) if rows else np.array([], dtype=np.float32)


def fit_neural_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_group_ids: pd.Series | None,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict, list[dict]]:
    device = resolve_device(args.device)
    model = build_neural_model(args, input_size=X_train.shape[2]).to(device)
    pos_count = max(float(y_train.sum()), 1.0)
    neg_count = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = build_loader(X_train, y_train, args.batch_size, shuffle=True)

    best_state = None
    best_record = None
    epochs_without_improvement = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_scores = predict_neural_scores(model, X_val, device=device, batch_size=args.batch_size)
        threshold = fusion_train.select_threshold_with_objective(
            y_true=pd.Series(y_val),
            scores=val_scores,
            objective=args.threshold_objective,
            group_ids=val_group_ids,
        )
        val_metrics = fusion_train.evaluate_model(pd.Series(y_val), val_scores, threshold)
        if val_group_ids is not None:
            val_metrics["scenario_metrics"] = fusion_train.evaluate_grouped_predictions(
                y_true=pd.Series(y_val),
                scores=val_scores,
                threshold=threshold,
                group_ids=val_group_ids,
                group_label="scenario",
            )
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_f1": float(val_metrics["f1"]),
            "validation_average_precision": float(val_metrics["average_precision"]),
            "validation_scenario_f1": (
                None
                if "scenario_metrics" not in val_metrics
                else float(val_metrics["scenario_metrics"]["scenario_f1"])
            ),
            "threshold": float(threshold),
        }
        history.append(epoch_record)

        primary_metric = epoch_record["validation_f1"]
        best_primary_metric = None if best_record is None else best_record.get("primary_metric")
        if args.threshold_objective == "scenario_f1" and epoch_record["validation_scenario_f1"] is not None:
            primary_metric = float(epoch_record["validation_scenario_f1"])
        improved = (
            best_record is None
            or primary_metric > best_primary_metric + 1e-6
            or (
                abs(primary_metric - best_primary_metric) <= 1e-6
                and epoch_record["validation_average_precision"] > best_record["validation_average_precision"] + 1e-6
            )
        )
        if improved:
            best_record = {
                "epoch": epoch,
                "threshold": threshold,
                "primary_metric": primary_metric,
                "validation_f1": epoch_record["validation_f1"],
                "validation_average_precision": epoch_record["validation_average_precision"],
                "validation_scenario_f1": epoch_record["validation_scenario_f1"],
                "validation_metrics": val_metrics,
            }
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None or best_record is None:
        raise RuntimeError("Neural sequence training did not produce a valid checkpoint.")

    best_model = build_neural_model(args, input_size=X_train.shape[2])
    best_model.load_state_dict(best_state)
    best_model = best_model.to(device)
    return best_model, best_record, history


def fit_neural_full_train(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    args: argparse.Namespace,
    epochs: int,
) -> nn.Module:
    device = resolve_device(args.device)
    model = build_neural_model(args, input_size=X_train_full.shape[2]).to(device)
    pos_count = max(float(y_train_full.sum()), 1.0)
    neg_count = max(float(len(y_train_full) - y_train_full.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = build_loader(X_train_full, y_train_full, args.batch_size, shuffle=True)
    for _ in range(max(int(epochs), 1)):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    return model


def build_prediction_frame(
    df_subset: pd.DataFrame,
    label_column: str,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    prediction_columns = [
        column
        for column in (
            "window_end_line_number",
            "window_end_timestamp",
            "window_end_time_unix",
            "window_group",
            "scenario_id",
            "scenario_role",
            label_column,
        )
        if column in df_subset.columns
    ]
    predictions_df = df_subset[prediction_columns].reset_index(drop=True).copy()
    predictions_df["sequence_score"] = scores
    predictions_df["sequence_prediction"] = (scores >= threshold).astype(int)
    return predictions_df


def add_scenario_metrics(
    metrics: dict,
    df_subset: pd.DataFrame,
    label_column: str,
    scores: np.ndarray,
    threshold: float,
    scenario_column: str,
) -> None:
    if scenario_column not in df_subset.columns:
        return
    group_ids = fusion_train.build_scenario_group_ids(df_subset.reset_index(drop=True), scenario_column)
    metrics["scenario_metrics"] = fusion_train.evaluate_grouped_predictions(
        y_true=df_subset[label_column].reset_index(drop=True),
        scores=scores,
        threshold=threshold,
        group_ids=group_ids,
        group_label="scenario",
    )


def train_random_forest_branch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    val_group_ids: pd.Series | None,
    args: argparse.Namespace,
) -> tuple[RandomForestClassifier, float, dict, dict, pd.DataFrame, list[dict], np.ndarray, np.ndarray]:
    print("Training sequence branch Random Forest...")
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
    threshold = fusion_train.select_threshold_with_objective(
        y_true=y_val,
        scores=val_scores,
        objective=args.threshold_objective,
        group_ids=val_group_ids,
    )
    val_metrics = fusion_train.evaluate_model(y_val, val_scores, threshold)
    if val_group_ids is not None:
        val_metrics["scenario_metrics"] = fusion_train.evaluate_grouped_predictions(
            y_true=y_val,
            scores=val_scores,
            threshold=threshold,
            group_ids=val_group_ids,
            group_label="scenario",
        )

    print(f"Validation threshold selected by {args.threshold_objective}: {threshold:.4f}")
    print("Refitting on train + validation windows...")
    clf.fit(X_train_full, y_train_full)
    test_scores = clf.predict_proba(X_test)[:, 1]
    test_metrics = fusion_train.evaluate_model(y_test, test_scores, threshold)
    importances = (
        pd.DataFrame({"feature": list(X_train.columns), "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return clf, threshold, val_metrics, test_metrics, importances, [], val_scores, test_scores


def train_neural_branch(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    val_group_ids: pd.Series | None,
    args: argparse.Namespace,
) -> tuple[dict, float, dict, dict, pd.DataFrame | None, list[dict], np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError(
            f"Model type '{args.model_type}' requires PyTorch. Install torch or use --model-type rf."
        )

    base_names, ordered_steps = build_feature_layout(feature_columns)
    all_tensor = build_sequence_tensor(df, feature_columns, base_names, ordered_steps)
    y_all = df[label_column].astype(int).to_numpy()

    train_tensor = all_tensor[train_idx]
    val_tensor = all_tensor[val_idx]
    test_tensor = all_tensor[test_idx]
    train_full_tensor = all_tensor[np.concatenate([train_idx, val_idx])]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]
    y_train_full = y_all[np.concatenate([train_idx, val_idx])]

    train_mean, train_std = compute_tensor_scaler(train_tensor)
    train_tensor_scaled = scale_tensor(train_tensor, train_mean, train_std)
    val_tensor_scaled = scale_tensor(val_tensor, train_mean, train_std)

    print(f"Training sequence branch {args.model_type.upper()} on device {resolve_device(args.device)}...")
    best_model, best_record, history = fit_neural_model(
        X_train=train_tensor_scaled,
        y_train=y_train,
        X_val=val_tensor_scaled,
        y_val=y_val,
        val_group_ids=val_group_ids,
        args=args,
    )

    threshold = float(best_record["threshold"])
    val_metrics = dict(best_record["validation_metrics"])
    val_scores = predict_neural_scores(
        best_model,
        val_tensor_scaled,
        device=resolve_device(args.device),
        batch_size=args.batch_size,
    )

    full_mean, full_std = compute_tensor_scaler(train_full_tensor)
    train_full_scaled = scale_tensor(train_full_tensor, full_mean, full_std)
    test_tensor_scaled = scale_tensor(test_tensor, full_mean, full_std)
    final_model = fit_neural_full_train(
        X_train_full=train_full_scaled,
        y_train_full=y_train_full,
        args=args,
        epochs=int(best_record["epoch"]),
    )
    test_scores = predict_neural_scores(
        final_model,
        test_tensor_scaled,
        device=resolve_device(args.device),
        batch_size=args.batch_size,
    )
    test_metrics = fusion_train.evaluate_model(pd.Series(y_test), test_scores, threshold)
    model_bundle = {
        "state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
        "scaler_mean": full_mean,
        "scaler_std": full_std,
        "base_feature_names": list(base_names),
        "ordered_steps": list(ordered_steps),
        "best_epoch": int(best_record["epoch"]),
    }
    return model_bundle, threshold, val_metrics, test_metrics, None, history, val_scores, test_scores


def train_sequence_branch(args: argparse.Namespace) -> dict:
    print(f"Loading sequence windows from {args.sequence_csv}")
    df = pd.read_csv(args.sequence_csv, low_memory=False)
    resolved_label_column = fusion_train.resolve_label_column(df, args.label_column)
    df = df.dropna(subset=[resolved_label_column]).copy()
    df[resolved_label_column] = df[resolved_label_column].astype(int)

    feature_columns, excluded_columns = choose_sequence_features(df, resolved_label_column)
    X = fusion_train.sanitize_numeric_frame(df[feature_columns])
    y = df[resolved_label_column]

    print(f"Sequence dataset shape: {X.shape}")
    print(f"Resolved label column: {resolved_label_column}")
    print(f"Positive windows: {int(y.sum())} ({y.mean() * 100:.4f}%)")

    train_idx, val_idx, test_idx, split_summary = split_sequence_dataset(df, y, args)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    val_group_ids = None
    if args.threshold_objective == "scenario_f1":
        if args.scenario_column not in df.columns:
            raise ValueError("Scenario threshold optimization requested but scenario metadata is unavailable.")
        val_group_ids = fusion_train.build_scenario_group_ids(df.iloc[val_idx].reset_index(drop=True), args.scenario_column)

    if args.model_type == "rf":
        model_object, threshold, val_metrics, test_metrics, importances, history, val_scores, test_scores = train_random_forest_branch(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
            val_group_ids=val_group_ids,
            args=args,
        )
        model_payload = {"model": model_object}
    else:
        model_payload, threshold, val_metrics, test_metrics, importances, history, val_scores, test_scores = train_neural_branch(
            df=df,
            feature_columns=feature_columns,
            label_column=resolved_label_column,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            val_group_ids=val_group_ids,
            args=args,
        )

    add_scenario_metrics(
        metrics=val_metrics,
        df_subset=df.iloc[val_idx].reset_index(drop=True),
        label_column=resolved_label_column,
        scores=val_scores,
        threshold=threshold,
        scenario_column=args.scenario_column,
    )
    add_scenario_metrics(
        metrics=test_metrics,
        df_subset=df.iloc[test_idx].reset_index(drop=True),
        label_column=resolved_label_column,
        scores=test_scores,
        threshold=threshold,
        scenario_column=args.scenario_column,
    )

    print("\nValidation report:")
    print(val_metrics["classification_report"])
    print("Validation confusion matrix:")
    print(np.array(val_metrics["confusion_matrix"]))
    if "scenario_metrics" in val_metrics:
        print("Validation scenario confusion matrix:")
        print(np.array(val_metrics["scenario_metrics"]["scenario_confusion_matrix"]))

    print("\nSequence test report:")
    print(test_metrics["classification_report"])
    print("Sequence test confusion matrix:")
    print(np.array(test_metrics["confusion_matrix"]))
    if "scenario_metrics" in test_metrics:
        print("Sequence test scenario confusion matrix:")
        print(np.array(test_metrics["scenario_metrics"]["scenario_confusion_matrix"]))

    if importances is not None:
        print("\nTop sequence features:")
        print(importances.head(12).to_string(index=False))

    predictions_df = build_prediction_frame(
        df_subset=df.iloc[test_idx].reset_index(drop=True),
        label_column=resolved_label_column,
        scores=test_scores,
        threshold=threshold,
    )
    predictions_df.to_csv(args.predictions_csv, index=False)

    model_bundle = {
        **model_payload,
        "model_type": args.model_type,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "label_column": resolved_label_column,
        "threshold_objective": args.threshold_objective,
        "split_summary": split_summary,
        "metrics": {"validation": val_metrics, "test": test_metrics},
        "training_history": history,
    }
    print(f"\nSaving sequence branch model bundle to {args.model_path}")
    joblib.dump(model_bundle, args.model_path)

    report_lines = [
        "MMS Sequence Branch Evaluation",
        "==============================",
        "",
        f"Dataset: {args.sequence_csv}",
        f"Label column: {resolved_label_column}",
        f"Model type: {args.model_type}",
        f"Threshold objective: {args.threshold_objective}",
        f"Rows: {len(df)}",
        f"Positive rows: {int(y.sum())} ({y.mean() * 100:.4f}%)",
        f"Split summary: {json.dumps(split_summary)}",
        f"Feature count: {len(feature_columns)}",
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
        "Training history:",
        json.dumps(history, indent=2),
    ]
    if importances is not None:
        report_lines.extend(["", "Feature importances:", importances.to_string(index=False)])

    with open(args.report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    print(f"Sequence branch report saved to {args.report_path}")
    print(f"Sequence branch predictions saved to {args.predictions_csv}")
    return model_bundle


if __name__ == "__main__":
    train_sequence_branch(parse_args())
