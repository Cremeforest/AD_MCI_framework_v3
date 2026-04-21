from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CohortConfig:
    project_root: Path
    dxsum_file: str = "data_raw/DXSUM.csv"
    ptdemog_file: str = "data_raw/PTDEMOG.csv"
    output_file: str = "data_processed/cohort/adni_landmark_cohort.csv"
    landmark_days: int = 365
    min_followup_days_after_landmark: int = 1


def _find_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _require_column(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    found = _find_first_existing(columns, candidates)
    if found is None:
        raise ValueError(
            f"Could not find a column for {label}. "
            f"Tried candidates: {list(candidates)}. "
            f"Available columns include: {list(columns)[:50]}"
        )
    return found


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _normalize_viscode(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    days = (end - start).dt.days
    return days / 30.4375


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_mci_row(df: pd.DataFrame, diag_col: Optional[str], dxad_col: Optional[str], dxmci_col: Optional[str]) -> pd.Series:
    """
    Conservative MCI rule:
    1) If DXMCI exists and == 1, use it.
    2) Else if DIAGNOSIS exists and == 2, use it. (common ADNI coding)
    """
    mci = pd.Series(False, index=df.index)

    if dxmci_col is not None:
        mci = mci | (_safe_numeric(df[dxmci_col]) == 1)

    if diag_col is not None:
        mci = mci | (_safe_numeric(df[diag_col]) == 2)

    return mci


def _is_ad_row(df: pd.DataFrame, diag_col: Optional[str], dxad_col: Optional[str]) -> pd.Series:
    """
    Conservative AD rule:
    1) If DXAD exists and == 1, use it.
    2) Else if DIAGNOSIS exists and == 3, use it. (common ADNI coding)
    """
    ad = pd.Series(False, index=df.index)

    if dxad_col is not None:
        ad = ad | (_safe_numeric(df[dxad_col]) == 1)

    if diag_col is not None:
        ad = ad | (_safe_numeric(df[diag_col]) == 3)

    return ad


def build_adni_landmark_cohort(config: CohortConfig) -> pd.DataFrame:
    root = config.project_root
    dxsum_path = root / config.dxsum_file
    ptdemog_path = root / config.ptdemog_file
    output_path = root / config.output_file

    if not dxsum_path.exists():
        raise FileNotFoundError(f"DXSUM file not found: {dxsum_path}")
    if not ptdemog_path.exists():
        raise FileNotFoundError(f"PTDEMOG file not found: {ptdemog_path}")

    dx = pd.read_csv(dxsum_path)
    demo = pd.read_csv(ptdemog_path)

    rid_col = _require_column(dx.columns, ["RID"], "RID")
    viscode_col = _require_column(dx.columns, ["VISCODE", "VISCODE2"], "VISCODE")
    visit_date_col = _require_column(dx.columns, ["EXAMDATE", "VISDATE", "USERDATE"], "visit date")

    diag_col = _find_first_existing(dx.columns, ["DIAGNOSIS"])
    dxmci_col = _find_first_existing(dx.columns, ["DXMCI"])
    dxad_col = _find_first_existing(dx.columns, ["DXAD"])

    if diag_col is None and dxmci_col is None and dxad_col is None:
        raise ValueError(
            "No usable diagnosis columns found in DXSUM. "
            "Need at least one of: DIAGNOSIS, DXMCI, DXAD."
        )

    dx = dx.copy()
    dx[rid_col] = _safe_numeric(dx[rid_col]).astype("Int64")
    dx["visit_date"] = _to_datetime(dx[visit_date_col])
    dx["viscode_norm"] = _normalize_viscode(dx[viscode_col])

    dx = dx.dropna(subset=[rid_col, "visit_date"]).copy()
    dx = dx.sort_values([rid_col, "visit_date"]).reset_index(drop=True)

    dx["is_mci"] = _is_mci_row(dx, diag_col=diag_col, dxad_col=dxad_col, dxmci_col=dxmci_col)
    dx["is_ad"] = _is_ad_row(dx, diag_col=diag_col, dxad_col=dxad_col)

    # --- Step 1: define baseline as first MCI visit, preferring VISCODE == 'bl'
    mci_rows = dx[dx["is_mci"]].copy()
    if mci_rows.empty:
        raise ValueError("No MCI rows found. Please inspect DXSUM diagnosis columns/codings.")

    mci_bl = mci_rows[mci_rows["viscode_norm"] == "bl"].copy()
    baseline_from_bl = (
        mci_bl.sort_values([rid_col, "visit_date"])
        .groupby(rid_col, as_index=False)
        .first()[[rid_col, "visit_date", viscode_col]]
        .rename(columns={"visit_date": "baseline_date", viscode_col: "baseline_viscode"})
    )

    baseline_any = (
        mci_rows.sort_values([rid_col, "visit_date"])
        .groupby(rid_col, as_index=False)
        .first()[[rid_col, "visit_date", viscode_col]]
        .rename(columns={"visit_date": "baseline_date_any", viscode_col: "baseline_viscode_any"})
    )

    baseline = baseline_any.merge(baseline_from_bl, on=rid_col, how="left")

    baseline["baseline_date"] = baseline["baseline_date"].fillna(baseline["baseline_date_any"])
    baseline["baseline_viscode"] = baseline["baseline_viscode"].fillna(baseline["baseline_viscode_any"])

    baseline = baseline[[rid_col, "baseline_date", "baseline_viscode"]].copy()
    baseline["landmark_date"] = baseline["baseline_date"] + pd.to_timedelta(config.landmark_days, unit="D")

    # --- Step 2: last observed visit after baseline (for censoring)
    last_followup = (
        dx.sort_values([rid_col, "visit_date"])
        .groupby(rid_col, as_index=False)
        .last()[[rid_col, "visit_date"]]
        .rename(columns={"visit_date": "last_visit_date"})
    )

    cohort = baseline.merge(last_followup, on=rid_col, how="left")

    # --- Step 3: first AD event after landmark
    ad_rows = dx[dx["is_ad"]].copy()
    ad_after_baseline = ad_rows.merge(
        cohort[[rid_col, "baseline_date", "landmark_date"]],
        on=rid_col,
        how="inner",
    )
    ad_after_baseline = ad_after_baseline[
        ad_after_baseline["visit_date"] > ad_after_baseline["landmark_date"]
    ].copy()

    first_ad = (
        ad_after_baseline.sort_values([rid_col, "visit_date"])
        .groupby(rid_col, as_index=False)
        .first()[[rid_col, "visit_date"]]
        .rename(columns={"visit_date": "ad_event_date"})
    )

    cohort = cohort.merge(first_ad, on=rid_col, how="left")

    # --- Step 4: define end_date and event
    cohort["event"] = cohort["ad_event_date"].notna().astype(int)
    cohort["end_date"] = cohort["ad_event_date"].where(cohort["event"] == 1, cohort["last_visit_date"])

    # --- Step 5: keep only subjects observed through landmark and with follow-up after landmark
    cohort = cohort[cohort["last_visit_date"] >= cohort["landmark_date"]].copy()

    followup_days = (cohort["end_date"] - cohort["landmark_date"]).dt.days
    cohort = cohort[followup_days >= config.min_followup_days_after_landmark].copy()

    # --- Step 6: time from landmark to event/censor
    cohort["time"] = _months_between(cohort["landmark_date"], cohort["end_date"])

    # --- Step 7: attach a few demographic columns if available
    demo = demo.copy()
    demo_rid_col = _require_column(demo.columns, ["RID"], "PTDEMOG RID")
    demo[demo_rid_col] = _safe_numeric(demo[demo_rid_col]).astype("Int64")

    sex_col = _find_first_existing(demo.columns, ["PTGENDER", "SEX"])
    edu_col = _find_first_existing(demo.columns, ["PTEDUCAT", "EDUCAT"])
    yob_col = _find_first_existing(demo.columns, ["PTDOBYY"])
    dob_col = _find_first_existing(demo.columns, ["PTDOB"])

    demo_keep = [demo_rid_col]
    rename_map = {demo_rid_col: rid_col}

    if sex_col is not None:
        demo_keep.append(sex_col)
        rename_map[sex_col] = "sex"

    if edu_col is not None:
        demo_keep.append(edu_col)
        rename_map[edu_col] = "education"

    if yob_col is not None:
        demo_keep.append(yob_col)
        rename_map[yob_col] = "birth_year"

    if dob_col is not None:
        demo_keep.append(dob_col)
        rename_map[dob_col] = "birth_date_raw"

    demo_small = demo[demo_keep].drop_duplicates(subset=[demo_rid_col]).rename(columns=rename_map)

    cohort = cohort.merge(demo_small, on=rid_col, how="left")

    # --- Step 8: derive age at baseline if possible
    cohort["age"] = np.nan

    if "birth_date_raw" in cohort.columns:
        birth_date = _to_datetime(cohort["birth_date_raw"])
        age_years = (cohort["baseline_date"] - birth_date).dt.days / 365.25
        cohort["age"] = age_years

    if "birth_year" in cohort.columns:
        cohort["age"] = cohort["age"].fillna(
            cohort["baseline_date"].dt.year - _safe_numeric(cohort["birth_year"])
        )

    # --- Step 9: final cleanup
    cohort = cohort.rename(columns={rid_col: "RID"})
    cohort = cohort.sort_values(["RID", "baseline_date"]).reset_index(drop=True)

    final_cols = [
        "RID",
        "baseline_date",
        "baseline_viscode",
        "landmark_date",
        "last_visit_date",
        "ad_event_date",
        "end_date",
        "time",
        "event",
        "age",
        "sex",
        "education",
    ]
    final_cols = [c for c in final_cols if c in cohort.columns]
    cohort = cohort[final_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_path, index=False)

    return cohort


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cfg = CohortConfig(project_root=root)
    df = build_adni_landmark_cohort(cfg)

    print("\n" + "=" * 80)
    print("ADNI landmark cohort built successfully")
    print("=" * 80)
    print("Output:", root / cfg.output_file)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(5))
    print("\nEvent rate:", round(df["event"].mean(), 4) if "event" in df.columns else "NA")