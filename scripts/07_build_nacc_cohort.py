from pathlib import Path

from src.data.build_nacc_cohort import NACCBuildConfig, build_nacc_cohort


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_csv = project_root / "data_raw" / "NACC_investigator.csv"
    output_csv = project_root / "data_processed" / "cohort" / "nacc_landmark_cohort_v4.csv"

    config = NACCBuildConfig(
        input_csv=input_csv,
        output_csv=output_csv,
        min_followup_days=30,
        require_mci_at_baseline=True,
        event_definition="uds_dementia",
        verbose=True,
    )

    build_nacc_cohort(config)


if __name__ == "__main__":
    main()