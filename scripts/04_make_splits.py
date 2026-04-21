from pathlib import Path

from src.data.make_splits import SplitConfig, build_adni_splits


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = SplitConfig(project_root=root)
    outputs = build_adni_splits(cfg)

    print("\nDone.")
    for name, df in outputs.items():
        print(f"{name}: rows={len(df)}, cols={len(df.columns)}, event_rate={df['event'].mean():.4f}")


if __name__ == "__main__":
    main()