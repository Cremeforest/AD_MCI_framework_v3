from pathlib import Path

from src.evaluation.cluster_and_km import ClusterAndKMConfig, run_cluster_and_km


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = ClusterAndKMConfig(project_root=root)
    results = run_cluster_and_km(cfg)

    cluster_table = results["cluster_table"]
    print("\nDone.")
    print(f"cluster_table rows={len(cluster_table)}, cols={len(cluster_table.columns)}")


if __name__ == "__main__":
    main()