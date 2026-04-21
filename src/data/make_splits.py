from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitConfig:
    project_root: Path
    cohort_file: str = "data_processed/cohort/adni_landmark_cohort.csv"
    train_output: str = "data_processed/split/adni_train.csv"
    val_output: str = "data_processed/split/adni_val.csv"
    test_output: str = "data_processed/split/adni_test.csv"
    metadata_output: str = "data_processed/split/split_metadata.json"

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    stratify_col: str = "event"


def build_adni_splits(config: SplitConfig) -> dict[str, pd.DataFrame]:
    root = config.project_root
    cohort_path = root / config.cohort_file

    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")

    df = pd.read_csv(cohort_path)

    required_cols = ["RID", config.stratify_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cohort file: {missing}")

    if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    df = df.copy()
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["RID", config.stratify_col]).copy()

    # --- First split: train vs temp (val+test)
    temp_ratio = config.val_ratio + config.test_ratio

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=config.random_state,
        stratify=df[config.stratify_col],
    )

    # --- Second split: val vs test from temp
    # Within temp, val:test = val_ratio:test_ratio
    val_within_temp = config.val_ratio / temp_ratio

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_within_temp),
        random_state=config.random_state,
        stratify=temp_df[config.stratify_col],
    )

    # Sort by RID for neatness
    train_df = train_df.sort_values("RID").reset_index(drop=True)
    val_df = val_df.sort_values("RID").reset_index(drop=True)
    test_df = test_df.sort_values("RID").reset_index(drop=True)

    # Save files
    train_path = root / config.train_output
    val_path = root / config.val_output
    test_path = root / config.test_output
    metadata_path = root / config.metadata_output

    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "source_cohort_file": config.cohort_file,
        "train_output": config.train_output,
        "val_output": config.val_output,
        "test_output": config.test_output,
        "random_state": config.random_state,
        "stratify_col": config.stratify_col,
        "ratios": {
            "train": config.train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
        },
        "counts": {
            "total": int(len(df)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "event_rate": {
            "total": float(df[config.stratify_col].mean()),
            "train": float(train_df[config.stratify_col].mean()),
            "val": float(val_df[config.stratify_col].mean()),
            "test": float(test_df[config.stratify_col].mean()),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cfg = SplitConfig(project_root=root)
    outputs = build_adni_splits(cfg)

    print("\n" + "=" * 80)
    print("ADNI splits built successfully")
    print("=" * 80)

    for name, df in outputs.items():
        print(f"\n{name.upper()}")
        print("Shape:", df.shape)
        print("Event rate:", round(df["event"].mean(), 4))
        print(df[["RID", "time", "event"]].head(5))