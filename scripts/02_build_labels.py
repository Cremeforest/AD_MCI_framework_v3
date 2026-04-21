from pathlib import Path

from src.data.build_labels import LabelConfig, build_adni_labels


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = LabelConfig(project_root=root)
    outputs = build_adni_labels(cfg)

    print("\nDone.")
    for name, df in outputs.items():
        print(f"{name}: rows={len(df)}, cols={len(df.columns)}")


if __name__ == "__main__":
    main()