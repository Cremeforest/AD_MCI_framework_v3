from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class LabelConfig:
    project_root: Path
    cohort_file: str = "data_processed/cohort/adni_landmark_cohort.csv"
    survival_output: str = "data_processed/labels/adni_survival_labels.csv"
    event3y_output: str = "data_processed/labels/adni_aux_event3y_labels.csv"
    highrisk_output: str = "data_processed/labels/adni_aux_highrisk_labels.csv"
    event3y_months: float = 36.0
    highrisk_months: float = 24.0


def build_adni_labels(config: LabelConfig) -> dict[str, pd.DataFrame]:
    root = config.project_root
    cohort_path = root / config.cohort_file

    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")

    df = pd.read_csv(cohort_path)

    required_cols = ["RID", "time", "event"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cohort file: {missing}")

    # --- Main survival labels
    survival_df = df[["RID", "time", "event"]].copy()
    survival_df["time"] = pd.to_numeric(survival_df["time"], errors="coerce")
    survival_df["event"] = pd.to_numeric(survival_df["event"], errors="coerce").astype("Int64")

    # --- Auxiliary label 1: event within 3 years after landmark
    event3y_df = df[["RID", "time", "event"]].copy()
    event3y_df["time"] = pd.to_numeric(event3y_df["time"], errors="coerce")
    event3y_df["event"] = pd.to_numeric(event3y_df["event"], errors="coerce")

    event3y_df["event_3y"] = (
        (event3y_df["event"] == 1) &
        (event3y_df["time"] <= config.event3y_months)
    ).astype(int)

    event3y_df = event3y_df[["RID", "event_3y"]].copy()

    # --- Auxiliary label 2: high-risk rapid progression
    highrisk_df = df[["RID", "time", "event"]].copy()
    highrisk_df["time"] = pd.to_numeric(highrisk_df["time"], errors="coerce")
    highrisk_df["event"] = pd.to_numeric(highrisk_df["event"], errors="coerce")

    highrisk_df["highrisk"] = (
        (highrisk_df["event"] == 1) &
        (highrisk_df["time"] <= config.highrisk_months)
    ).astype(int)

    highrisk_df = highrisk_df[["RID", "highrisk"]].copy()

    # --- Save outputs
    survival_path = root / config.survival_output
    event3y_path = root / config.event3y_output
    highrisk_path = root / config.highrisk_output

    survival_path.parent.mkdir(parents=True, exist_ok=True)

    survival_df.to_csv(survival_path, index=False)
    event3y_df.to_csv(event3y_path, index=False)
    highrisk_df.to_csv(highrisk_path, index=False)

    return {
        "survival": survival_df,
        "event3y": event3y_df,
        "highrisk": highrisk_df,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cfg = LabelConfig(project_root=root)
    outputs = build_adni_labels(cfg)

    print("\n" + "=" * 80)
    print("ADNI labels built successfully")
    print("=" * 80)

    for name, df in outputs.items():
        print(f"\n{name.upper()}")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(5))

        if name == "survival":
            print("Event rate:", round(df["event"].mean(), 4))
        elif name == "event3y":
            print("Positive rate:", round(df["event_3y"].mean(), 4))
        elif name == "highrisk":
            print("Positive rate:", round(df["highrisk"].mean(), 4))