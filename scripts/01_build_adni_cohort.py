from pathlib import Path

from src.data.build_adni_cohort import CohortConfig, build_adni_landmark_cohort


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = CohortConfig(project_root=root)
    df = build_adni_landmark_cohort(cfg)

    print("\nDone.")
    print(f"Saved to: {root / cfg.output_file}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()