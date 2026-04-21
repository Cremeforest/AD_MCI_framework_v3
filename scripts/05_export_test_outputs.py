from pathlib import Path

from src.evaluation.export_test_outputs import ExportTestOutputsConfig, export_test_outputs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ExportTestOutputsConfig(project_root=root)
    results = export_test_outputs(cfg)

    df = results["dataframe"]
    print("\nDone.")
    print(f"rows={len(df)}, cols={len(df.columns)}")


if __name__ == "__main__":
    main()