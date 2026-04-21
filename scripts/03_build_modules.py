from pathlib import Path

from src.data.build_modules import ModuleConfig, build_adni_modules


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ModuleConfig(project_root=root)
    outputs = build_adni_modules(cfg)

    print("\nDone.")
    for name, df in outputs.items():
        print(f"{name}: rows={len(df)}, cols={len(df.columns)}")


if __name__ == "__main__":
    main()