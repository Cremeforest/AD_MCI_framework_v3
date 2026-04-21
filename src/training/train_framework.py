from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import ClinicalFrameworkDataset, DatasetPaths
from src.models.framework_model import ClinicalFrameworkModel, FrameworkModelConfig
from src.training.losses import combined_framework_loss
from src.training.masking import apply_random_module_mask


@dataclass
class TrainFrameworkConfig:
    project_root: Path

    train_split_file: str = "data_processed/split/adni_train.csv"
    val_split_file: str = "data_processed/split/adni_val.csv"
    test_split_file: str = "data_processed/split/adni_test.csv"

    best_model_output: str = "results/models/best_framework_model.pt"
    training_log_output: str = "results/models/training_log.csv"
    summary_output: str = "results/models/training_summary.json"
    preprocessing_output: str = "results/models/preprocessing_stats.json"

    epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    patience: int = 40

    lambda_survival: float = 1.0
    lambda_event3y: float = 0.5
    lambda_highrisk: float = 0.5

    use_module_masking: bool = True
    module_drop_prob: float = 0.15

    scheduler_factor: float = 0.5
    scheduler_patience: int = 12

    device: str = "auto"
    random_seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def make_dataset_paths(root: Path, split_file: str) -> DatasetPaths:
    return DatasetPaths(
        project_root=root,
        split_file=split_file,
    )


def compute_pos_weight(binary_targets: torch.Tensor) -> torch.Tensor | None:
    """
    pos_weight = neg / pos for BCEWithLogitsLoss
    """
    positives = float(binary_targets.sum().item())
    negatives = float(binary_targets.numel() - binary_targets.sum().item())

    if positives <= 0:
        return None

    value = negatives / max(positives, 1.0)
    return torch.tensor(value, dtype=torch.float32, device=binary_targets.device)


def harrell_c_index(
    times: torch.Tensor,
    events: torch.Tensor,
    risk_scores: torch.Tensor,
) -> float:
    """
    Simple Harrell's C-index.
    Higher risk score = higher predicted risk.
    """
    t = times.detach().cpu().numpy().astype(float)
    e = events.detach().cpu().numpy().astype(int)
    r = risk_scores.detach().cpu().numpy().astype(float)

    comparable = 0.0
    concordant = 0.0

    n = len(t)
    for i in range(n):
        for j in range(i + 1, n):
            if e[i] == 1 and t[i] < t[j]:
                comparable += 1.0
                if r[i] > r[j]:
                    concordant += 1.0
                elif r[i] == r[j]:
                    concordant += 0.5
            elif e[j] == 1 and t[j] < t[i]:
                comparable += 1.0
                if r[j] > r[i]:
                    concordant += 1.0
                elif r[i] == r[j]:
                    concordant += 0.5

    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


def evaluate_split(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    config: TrainFrameworkConfig,
    event3y_pos_weight: torch.Tensor | None,
    highrisk_pos_weight: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        outputs = model(
            baseline_x=batch["baseline_x"],
            structure_x=batch["structure_x"],
            state_x=batch["state_x"],
            dynamics_x=batch["dynamics_x"],
            availability_mask=batch["availability_mask"],
        )

        _, loss_dict = combined_framework_loss(
            outputs=outputs,
            batch=batch,
            lambda_survival=config.lambda_survival,
            lambda_event3y=config.lambda_event3y,
            lambda_highrisk=config.lambda_highrisk,
            event3y_pos_weight=event3y_pos_weight,
            highrisk_pos_weight=highrisk_pos_weight,
        )

        c_index = harrell_c_index(
            times=batch["time"],
            events=batch["event"],
            risk_scores=outputs["risk_score"],
        )

        event3y_prob = torch.sigmoid(outputs["event3y_logit"])
        highrisk_prob = torch.sigmoid(outputs["highrisk_logit"])

        event3y_acc = ((event3y_prob >= 0.5).float() == batch["event_3y"]).float().mean().item()
        highrisk_acc = ((highrisk_prob >= 0.5).float() == batch["highrisk"]).float().mean().item()

        metrics = {
            **loss_dict,
            "c_index": float(c_index),
            "event3y_acc": float(event3y_acc),
            "highrisk_acc": float(highrisk_acc),
        }

    return metrics


def train_framework(config: TrainFrameworkConfig) -> dict[str, Any]:
    set_seed(config.random_seed)
    device = get_device(config.device)
    root = config.project_root

    # ------------------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------------------
    train_paths = make_dataset_paths(root, config.train_split_file)
    val_paths = make_dataset_paths(root, config.val_split_file)
    test_paths = make_dataset_paths(root, config.test_split_file)

    train_dataset = ClinicalFrameworkDataset(
        paths=train_paths,
        fit_preprocessing=True,
        preprocessing_stats=None,
    )

    preprocessing_stats = train_dataset.get_serializable_preprocessing_stats()

    val_dataset = ClinicalFrameworkDataset(
        paths=val_paths,
        fit_preprocessing=False,
        preprocessing_stats=preprocessing_stats,
    )

    test_dataset = ClinicalFrameworkDataset(
        paths=test_paths,
        fit_preprocessing=False,
        preprocessing_stats=preprocessing_stats,
    )

    train_batch = train_dataset.get_all_tensors(device=device)
    val_batch = val_dataset.get_all_tensors(device=device)
    test_batch = test_dataset.get_all_tensors(device=device)

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    feature_dims = train_dataset.feature_dims

    model_cfg = FrameworkModelConfig(
        baseline_input_dim=feature_dims["baseline"],
        structure_input_dim=feature_dims["structure"],
        state_input_dim=feature_dims["state"],
        dynamics_input_dim=feature_dims["dynamics"],
    )

    model = ClinicalFrameworkModel(model_cfg).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    event3y_pos_weight = compute_pos_weight(train_batch["event_3y"])
    highrisk_pos_weight = compute_pos_weight(train_batch["highrisk"])

    # ------------------------------------------------------------------
    # OUTPUT PATHS
    # ------------------------------------------------------------------
    best_model_path = root / config.best_model_output
    training_log_path = root / config.training_log_output
    summary_path = root / config.summary_output
    preprocessing_path = root / config.preprocessing_output

    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(preprocessing_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_stats, f, indent=2)

    # ------------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------------
    history: list[dict[str, Any]] = []

    best_val_cindex = float("-inf")
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        train_mask = train_batch["availability_mask"]
        if config.use_module_masking:
            train_mask = apply_random_module_mask(
                availability_mask=train_mask,
                drop_prob=config.module_drop_prob,
                ensure_at_least_one=True,
            )

        outputs = model(
            baseline_x=train_batch["baseline_x"],
            structure_x=train_batch["structure_x"],
            state_x=train_batch["state_x"],
            dynamics_x=train_batch["dynamics_x"],
            availability_mask=train_mask,
        )

        total_loss, train_loss_dict = combined_framework_loss(
            outputs=outputs,
            batch=train_batch,
            lambda_survival=config.lambda_survival,
            lambda_event3y=config.lambda_event3y,
            lambda_highrisk=config.lambda_highrisk,
            event3y_pos_weight=event3y_pos_weight,
            highrisk_pos_weight=highrisk_pos_weight,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
        optimizer.step()

        # Clean evaluation on full splits using original availability mask
        train_metrics = evaluate_split(
            model=model,
            batch=train_batch,
            config=config,
            event3y_pos_weight=event3y_pos_weight,
            highrisk_pos_weight=highrisk_pos_weight,
        )

        val_metrics = evaluate_split(
            model=model,
            batch=val_batch,
            config=config,
            event3y_pos_weight=event3y_pos_weight,
            highrisk_pos_weight=highrisk_pos_weight,
        )

        scheduler.step(val_metrics["total_loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_total_loss": train_metrics["total_loss"],
            "train_survival_loss": train_metrics["survival_loss"],
            "train_event3y_loss": train_metrics["event3y_loss"],
            "train_highrisk_loss": train_metrics["highrisk_loss"],
            "train_c_index": train_metrics["c_index"],
            "train_event3y_acc": train_metrics["event3y_acc"],
            "train_highrisk_acc": train_metrics["highrisk_acc"],
            "val_total_loss": val_metrics["total_loss"],
            "val_survival_loss": val_metrics["survival_loss"],
            "val_event3y_loss": val_metrics["event3y_loss"],
            "val_highrisk_loss": val_metrics["highrisk_loss"],
            "val_c_index": val_metrics["c_index"],
            "val_event3y_acc": val_metrics["event3y_acc"],
            "val_highrisk_acc": val_metrics["highrisk_acc"],
        }
        history.append(row)

        improved = False
        if val_metrics["c_index"] > best_val_cindex + 1e-6:
            improved = True
        elif abs(val_metrics["c_index"] - best_val_cindex) <= 1e-6 and val_metrics["total_loss"] < best_val_loss:
            improved = True

        if improved:
            best_val_cindex = val_metrics["c_index"]
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_cfg),
                "train_config": {
                    **asdict(config),
                    "project_root": str(config.project_root),
                },
                "feature_dims": feature_dims,
                "best_val_c_index": best_val_cindex,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, best_model_path)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_metrics['total_loss']:.4f} | "
                f"val_loss={val_metrics['total_loss']:.4f} | "
                f"val_cindex={val_metrics['c_index']:.4f} | "
                f"lr={current_lr:.6f}"
            )

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    # ------------------------------------------------------------------
    # SAVE TRAIN LOG
    # ------------------------------------------------------------------
    history_df = pd.DataFrame(history)
    history_df.to_csv(training_log_path, index=False)

    # ------------------------------------------------------------------
    # FINAL TEST EVALUATION USING BEST MODEL
    # ------------------------------------------------------------------
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_split(
        model=model,
        batch=test_batch,
        config=config,
        event3y_pos_weight=event3y_pos_weight,
        highrisk_pos_weight=highrisk_pos_weight,
    )

    summary = {
        "device": str(device),
        "n_train": int(len(train_dataset)),
        "n_val": int(len(val_dataset)),
        "n_test": int(len(test_dataset)),
        "feature_dims": feature_dims,
        "best_epoch": int(best_epoch),
        "best_val_c_index": float(best_val_cindex),
        "best_val_loss": float(best_val_loss),
        "test_metrics": test_metrics,
        "positive_rates": {
            "train_event": train_dataset.positive_rate("event"),
            "train_event_3y": train_dataset.positive_rate("event_3y"),
            "train_highrisk": train_dataset.positive_rate("highrisk"),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("Training finished")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val C-index: {best_val_cindex:.4f}")
    print(f"Test C-index: {test_metrics['c_index']:.4f}")
    print(f"Test event3y acc: {test_metrics['event3y_acc']:.4f}")
    print(f"Test highrisk acc: {test_metrics['highrisk_acc']:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {training_log_path}")
    print(f"Summary saved to: {summary_path}")

    return {
        "summary": summary,
        "history": history_df,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = TrainFrameworkConfig(project_root=root)
    train_framework(cfg)


if __name__ == "__main__":
    main()