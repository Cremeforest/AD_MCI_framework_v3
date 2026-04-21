from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetPaths:
    project_root: Path
    split_file: str

    baseline_file: str = "data_processed/modules/adni_module_baseline.csv"
    structure_file: str = "data_processed/modules/adni_module_structure.csv"
    state_file: str = "data_processed/modules/adni_module_state.csv"
    dynamics_file: str = "data_processed/modules/adni_module_dynamics.csv"

    survival_label_file: str = "data_processed/labels/adni_survival_labels.csv"
    event3y_label_file: str = "data_processed/labels/adni_aux_event3y_labels.csv"
    highrisk_label_file: str = "data_processed/labels/adni_aux_highrisk_labels.csv"


class ClinicalFrameworkDataset(Dataset):
    """
    Dataset for the modular clinical AI framework.

    It:
    1) loads split membership by RID
    2) loads 4 module tables
    3) loads 3 label tables
    4) imputes / standardizes features using train-derived preprocessing stats
    5) provides module availability mask
    """

    MODULE_NAMES = ["baseline", "structure", "state", "dynamics"]

    def __init__(
        self,
        paths: DatasetPaths,
        fit_preprocessing: bool = False,
        preprocessing_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.fit_preprocessing = fit_preprocessing

        if fit_preprocessing and preprocessing_stats is not None:
            raise ValueError("If fit_preprocessing=True, preprocessing_stats must be None.")

        self.root = paths.project_root

        split_df = self._read_csv(paths.split_file)
        if "RID" not in split_df.columns:
            raise ValueError(f"Split file must contain RID column: {paths.split_file}")

        rid_df = split_df[["RID"]].copy()
        rid_df["RID"] = pd.to_numeric(rid_df["RID"], errors="coerce").astype("Int64")
        rid_df = rid_df.dropna(subset=["RID"]).drop_duplicates(subset=["RID"]).reset_index(drop=True)

        baseline_df = self._read_csv(paths.baseline_file)
        structure_df = self._read_csv(paths.structure_file)
        state_df = self._read_csv(paths.state_file)
        dynamics_df = self._read_csv(paths.dynamics_file)

        survival_df = self._read_csv(paths.survival_label_file)
        event3y_df = self._read_csv(paths.event3y_label_file)
        highrisk_df = self._read_csv(paths.highrisk_label_file)

        module_tables = {
            "baseline": baseline_df,
            "structure": structure_df,
            "state": state_df,
            "dynamics": dynamics_df,
        }

        self.feature_columns: dict[str, list[str]] = {}
        self.module_arrays: dict[str, np.ndarray] = {}
        self.availability_by_module: dict[str, np.ndarray] = {}

        # Prepare / fit preprocessing
        if preprocessing_stats is None:
            self.preprocessing_stats: dict[str, Any] = {}
        else:
            self.preprocessing_stats = preprocessing_stats

        for module_name, module_df in module_tables.items():
            if "RID" not in module_df.columns:
                raise ValueError(f"{module_name} module file must contain RID column.")

            tmp = module_df.copy()
            tmp["RID"] = pd.to_numeric(tmp["RID"], errors="coerce").astype("Int64")

            feature_cols = [c for c in tmp.columns if c != "RID"]
            self.feature_columns[module_name] = feature_cols

            merged = rid_df.merge(tmp, on="RID", how="left")
            raw_features = merged[feature_cols].apply(pd.to_numeric, errors="coerce")

            # Module availability: at least one observed feature in this module
            availability = (~raw_features.isna()).any(axis=1).astype(np.float32).to_numpy()
            self.availability_by_module[module_name] = availability

            if fit_preprocessing:
                stats = self._fit_module_preprocessing(raw_features, feature_cols)
                self.preprocessing_stats[module_name] = stats
            else:
                if module_name not in self.preprocessing_stats:
                    raise ValueError(
                        f"Missing preprocessing stats for module '{module_name}'."
                    )
                stats = self.preprocessing_stats[module_name]

            processed = self._transform_module(raw_features, feature_cols, stats)
            self.module_arrays[module_name] = processed.astype(np.float32)

        # Labels
        self.survival_df = rid_df.merge(
            survival_df[["RID", "time", "event"]].copy(),
            on="RID",
            how="left",
        )
        self.event3y_df = rid_df.merge(
            event3y_df[["RID", "event_3y"]].copy(),
            on="RID",
            how="left",
        )
        self.highrisk_df = rid_df.merge(
            highrisk_df[["RID", "highrisk"]].copy(),
            on="RID",
            how="left",
        )

        self.survival_df["time"] = pd.to_numeric(self.survival_df["time"], errors="coerce")
        self.survival_df["event"] = pd.to_numeric(self.survival_df["event"], errors="coerce")
        self.event3y_df["event_3y"] = pd.to_numeric(self.event3y_df["event_3y"], errors="coerce")
        self.highrisk_df["highrisk"] = pd.to_numeric(self.highrisk_df["highrisk"], errors="coerce")

        missing_label_counts = {
            "time": int(self.survival_df["time"].isna().sum()),
            "event": int(self.survival_df["event"].isna().sum()),
            "event_3y": int(self.event3y_df["event_3y"].isna().sum()),
            "highrisk": int(self.highrisk_df["highrisk"].isna().sum()),
        }
        if any(v > 0 for v in missing_label_counts.values()):
            raise ValueError(f"Missing labels detected after merge: {missing_label_counts}")

        self.rids = rid_df["RID"].astype(int).to_numpy()

        self.time = self.survival_df["time"].to_numpy(dtype=np.float32)
        self.event = self.survival_df["event"].to_numpy(dtype=np.float32)
        self.event3y = self.event3y_df["event_3y"].to_numpy(dtype=np.float32)
        self.highrisk = self.highrisk_df["highrisk"].to_numpy(dtype=np.float32)

        self.availability_mask = np.stack(
            [
                self.availability_by_module["baseline"],
                self.availability_by_module["structure"],
                self.availability_by_module["state"],
                self.availability_by_module["dynamics"],
            ],
            axis=1,
        ).astype(np.float32)

    def _read_csv(self, relative_path: str) -> pd.DataFrame:
        path = self.root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path)

    def _fit_module_preprocessing(
        self,
        raw_features: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        medians = raw_features.median(numeric_only=True).reindex(feature_cols).fillna(0.0)
        imputed = raw_features.fillna(medians)

        means = imputed.mean(numeric_only=True).reindex(feature_cols).fillna(0.0)
        stds = imputed.std(ddof=0, numeric_only=True).reindex(feature_cols).fillna(1.0)
        stds = stds.replace(0.0, 1.0)

        return {
            "feature_names": feature_cols,
            "median": medians.to_dict(),
            "mean": means.to_dict(),
            "std": stds.to_dict(),
        }

    def _transform_module(
        self,
        raw_features: pd.DataFrame,
        feature_cols: list[str],
        stats: dict[str, Any],
    ) -> np.ndarray:
        expected = stats["feature_names"]
        if list(expected) != list(feature_cols):
            raise ValueError(
                f"Feature mismatch.\nExpected: {expected}\nGot: {feature_cols}"
            )

        medians = pd.Series(stats["median"], index=feature_cols, dtype=float)
        means = pd.Series(stats["mean"], index=feature_cols, dtype=float)
        stds = pd.Series(stats["std"], index=feature_cols, dtype=float)

        imputed = raw_features.fillna(medians)
        standardized = (imputed - means) / stds
        standardized = standardized.fillna(0.0)

        return standardized.to_numpy(dtype=np.float32)

    @property
    def feature_dims(self) -> dict[str, int]:
        return {
            module_name: len(cols)
            for module_name, cols in self.feature_columns.items()
        }

    def get_serializable_preprocessing_stats(self) -> dict[str, Any]:
        return self.preprocessing_stats

    def positive_rate(self, label_name: str) -> float:
        if label_name == "event":
            return float(self.event.mean())
        if label_name == "event_3y":
            return float(self.event3y.mean())
        if label_name == "highrisk":
            return float(self.highrisk.mean())
        raise ValueError(f"Unknown label_name: {label_name}")

    def __len__(self) -> int:
        return len(self.rids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "RID": torch.tensor(self.rids[idx], dtype=torch.long),
            "baseline_x": torch.tensor(self.module_arrays["baseline"][idx], dtype=torch.float32),
            "structure_x": torch.tensor(self.module_arrays["structure"][idx], dtype=torch.float32),
            "state_x": torch.tensor(self.module_arrays["state"][idx], dtype=torch.float32),
            "dynamics_x": torch.tensor(self.module_arrays["dynamics"][idx], dtype=torch.float32),
            "availability_mask": torch.tensor(self.availability_mask[idx], dtype=torch.float32),
            "time": torch.tensor(self.time[idx], dtype=torch.float32),
            "event": torch.tensor(self.event[idx], dtype=torch.float32),
            "event_3y": torch.tensor(self.event3y[idx], dtype=torch.float32),
            "highrisk": torch.tensor(self.highrisk[idx], dtype=torch.float32),
        }

    def get_all_tensors(self, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
        batch = {
            "RID": torch.tensor(self.rids, dtype=torch.long),
            "baseline_x": torch.tensor(self.module_arrays["baseline"], dtype=torch.float32),
            "structure_x": torch.tensor(self.module_arrays["structure"], dtype=torch.float32),
            "state_x": torch.tensor(self.module_arrays["state"], dtype=torch.float32),
            "dynamics_x": torch.tensor(self.module_arrays["dynamics"], dtype=torch.float32),
            "availability_mask": torch.tensor(self.availability_mask, dtype=torch.float32),
            "time": torch.tensor(self.time, dtype=torch.float32),
            "event": torch.tensor(self.event, dtype=torch.float32),
            "event_3y": torch.tensor(self.event3y, dtype=torch.float32),
            "highrisk": torch.tensor(self.highrisk, dtype=torch.float32),
        }

        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch