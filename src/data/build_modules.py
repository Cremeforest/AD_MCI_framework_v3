from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ModuleConfig:
    project_root: Path
    cohort_file: str = "data_processed/cohort/adni_landmark_cohort.csv"

    ptdemog_file: str = "data_raw/PTDEMOG.csv"
    apoeres_file: str = "data_raw/APOERES.csv"
    mmse_file: str = "data_raw/MMSE.csv"
    cdr_file: str = "data_raw/CDR.csv"
    adas_file: str = "data_raw/ADAS.csv"
    faq_file: str = "data_raw/FAQ.csv"

    baseline_output: str = "data_processed/modules/adni_module_baseline.csv"
    structure_output: str = "data_processed/modules/adni_module_structure.csv"
    state_output: str = "data_processed/modules/adni_module_state.csv"
    dynamics_output: str = "data_processed/modules/adni_module_dynamics.csv"

    state_window_days: int = 180
    dynamic_window_6m_days: int = 183
    dynamic_window_12m_days: int = 365


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
            f"Available columns include: {list(columns)[:80]}"
        )
    return found


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    # ADNI often uses 1/2 or Male/Female
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s.isin(["MALE", "M", "1", "1.0"])] = 1.0
    out[s.isin(["FEMALE", "F", "2", "2.0"])] = 2.0
    return out


def _prepare_generic_table(df: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, str, str]:
    rid_col = _require_column(df.columns, ["RID"], f"{table_name} RID")
    date_col = _require_column(df.columns, ["EXAMDATE", "VISDATE", "USERDATE"], f"{table_name} date")

    out = df.copy()
    out[rid_col] = _to_numeric(out[rid_col]).astype("Int64")
    out["visit_date"] = _to_datetime(out[date_col])
    out = out.dropna(subset=[rid_col, "visit_date"]).copy()
    out = out.sort_values([rid_col, "visit_date"]).reset_index(drop=True)
    return out, rid_col, "visit_date"


def _nearest_before_or_on(
    source_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    value_cols: list[str],
    target_date_col: str,
    window_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each RID in cohort_df, find the nearest row in source_df with visit_date <= target_date,
    optionally requiring target_date - visit_date <= window_days.
    """
    src = source_df.copy()
    cohort = cohort_df[["RID", target_date_col]].copy()

    merged = src.merge(cohort, on="RID", how="inner")
    merged["days_diff"] = (merged[target_date_col] - merged["visit_date"]).dt.days

    merged = merged[merged["days_diff"] >= 0].copy()
    if window_days is not None:
        merged = merged[merged["days_diff"] <= window_days].copy()

    if merged.empty:
        result = cohort_df[["RID"]].copy()
        for c in value_cols:
            result[c] = np.nan
        return result

    merged = merged.sort_values(["RID", "days_diff", "visit_date"])
    nearest = merged.groupby("RID", as_index=False).first()

    keep_cols = ["RID"] + value_cols
    return nearest[keep_cols].copy()


def _nearest_to_target(
    source_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    value_cols: list[str],
    target_date_col: str,
    max_abs_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    For each RID in cohort_df, find the row in source_df closest to target_date.
    """
    src = source_df.copy()
    cohort = cohort_df[["RID", target_date_col]].copy()

    merged = src.merge(cohort, on="RID", how="inner")
    merged["abs_days_diff"] = (merged["visit_date"] - merged[target_date_col]).abs().dt.days

    if max_abs_days is not None:
        merged = merged[merged["abs_days_diff"] <= max_abs_days].copy()

    if merged.empty:
        result = cohort_df[["RID"]].copy()
        for c in value_cols:
            result[c] = np.nan
        return result

    merged = merged.sort_values(["RID", "abs_days_diff", "visit_date"])
    nearest = merged.groupby("RID", as_index=False).first()

    keep_cols = ["RID"] + value_cols
    return nearest[keep_cols].copy()


def _compute_delta(
    state_df: pd.DataFrame,
    earlier_df: pd.DataFrame,
    rid_col: str,
    state_cols: dict[str, str],
    prefix: str,
) -> pd.DataFrame:
    out = state_df[[rid_col]].copy()

    for final_name, state_col in state_cols.items():
        earlier_col = f"{state_col}_{prefix}_ref"

        left = state_df[[rid_col, state_col]].copy()
        right = earlier_df[[rid_col, earlier_col]].copy()

        merged = left.merge(right, on=rid_col, how="left")
        out[final_name] = merged[state_col] - merged[earlier_col]

    return out


def build_adni_modules(config: ModuleConfig) -> dict[str, pd.DataFrame]:
    root = config.project_root

    cohort_path = root / config.cohort_file
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")

    cohort = pd.read_csv(cohort_path)
    cohort["RID"] = _to_numeric(cohort["RID"]).astype("Int64")
    cohort["baseline_date"] = _to_datetime(cohort["baseline_date"])
    cohort["landmark_date"] = _to_datetime(cohort["landmark_date"])

    # ------------------------------------------------------------------
    # BASELINE MODULE
    # ------------------------------------------------------------------
    demo_path = root / config.ptdemog_file
    apoe_path = root / config.apoeres_file

    if not demo_path.exists():
        raise FileNotFoundError(f"PTDEMOG file not found: {demo_path}")
    if not apoe_path.exists():
        raise FileNotFoundError(f"APOERES file not found: {apoe_path}")

    demo = pd.read_csv(demo_path)
    apoe = pd.read_csv(apoe_path)

    demo_rid = _require_column(demo.columns, ["RID"], "PTDEMOG RID")
    demo[demo_rid] = _to_numeric(demo[demo_rid]).astype("Int64")

    sex_col = _find_first_existing(demo.columns, ["PTGENDER", "SEX"])
    edu_col = _find_first_existing(demo.columns, ["PTEDUCAT", "EDUCAT"])

    baseline_df = cohort[["RID", "age"]].copy()

    if sex_col is not None:
        sex_small = demo[[demo_rid, sex_col]].drop_duplicates(subset=[demo_rid]).copy()
        sex_small = sex_small.rename(columns={demo_rid: "RID", sex_col: "sex"})
        sex_small["sex"] = _normalize_sex(sex_small["sex"])
        baseline_df = baseline_df.merge(sex_small, on="RID", how="left")
    else:
        baseline_df["sex"] = np.nan

    if edu_col is not None:
        edu_small = demo[[demo_rid, edu_col]].drop_duplicates(subset=[demo_rid]).copy()
        edu_small = edu_small.rename(columns={demo_rid: "RID", edu_col: "education"})
        edu_small["education"] = _to_numeric(edu_small["education"])
        baseline_df = baseline_df.merge(edu_small, on="RID", how="left")
    else:
        baseline_df["education"] = np.nan

    apoe_rid = _require_column(apoe.columns, ["RID"], "APOERES RID")
    apoe[apoe_rid] = _to_numeric(apoe[apoe_rid]).astype("Int64")

    apoe_col = _find_first_existing(apoe.columns, ["APOE4", "APOE4NUM", "NACCAPOE"])
    apgen1_col = _find_first_existing(apoe.columns, ["APGEN1"])
    apgen2_col = _find_first_existing(apoe.columns, ["APGEN2"])

    apoe_small = apoe[[apoe_rid] + [c for c in [apoe_col, apgen1_col, apgen2_col] if c is not None]].copy()
    apoe_small = apoe_small.drop_duplicates(subset=[apoe_rid]).rename(columns={apoe_rid: "RID"})

    if apoe_col is not None:
        apoe_small["APOE4"] = _to_numeric(apoe_small[apoe_col])
    elif apgen1_col is not None and apgen2_col is not None:
        g1 = _to_numeric(apoe_small[apgen1_col]).fillna(0)
        g2 = _to_numeric(apoe_small[apgen2_col]).fillna(0)
        apoe_small["APOE4"] = (g1.eq(4).astype(int) + g2.eq(4).astype(int)).astype(float)
    else:
        apoe_small["APOE4"] = np.nan

    apoe_small = apoe_small[["RID", "APOE4"]].copy()
    baseline_df = baseline_df.merge(apoe_small, on="RID", how="left")

    # Baseline MMSE
    mmse_path = root / config.mmse_file
    if not mmse_path.exists():
        raise FileNotFoundError(f"MMSE file not found: {mmse_path}")

    mmse = pd.read_csv(mmse_path)
    mmse, mmse_rid, mmse_date = _prepare_generic_table(mmse, "MMSE")

    mmse_score_col = _require_column(mmse.columns, ["MMSCORE", "MMSE"], "MMSE score")
    mmse["MMSE"] = _to_numeric(mmse[mmse_score_col])

    baseline_mmse = _nearest_to_target(
        source_df=mmse[[mmse_rid, mmse_date, "MMSE"]].rename(columns={mmse_rid: "RID", mmse_date: "visit_date"}),
        cohort_df=cohort,
        value_cols=["MMSE"],
        target_date_col="baseline_date",
        max_abs_days=config.state_window_days,
    ).rename(columns={"MMSE": "baseline_MMSE"})

    baseline_df = baseline_df.merge(baseline_mmse, on="RID", how="left")

    # Baseline CDR global
    cdr_path = root / config.cdr_file
    if not cdr_path.exists():
        raise FileNotFoundError(f"CDR file not found: {cdr_path}")

    cdr = pd.read_csv(cdr_path)
    cdr, cdr_rid, cdr_date = _prepare_generic_table(cdr, "CDR")

    cdrglob_col = _find_first_existing(cdr.columns, ["CDGLOBAL", "CDRGLOB"])
    cdrsb_col = _find_first_existing(cdr.columns, ["CDRSB", "CDRSUM"])

    if cdrglob_col is None:
        raise ValueError("Could not find CDR global column in CDR.csv.")

    cdr["CDR_global"] = _to_numeric(cdr[cdrglob_col])
    if cdrsb_col is not None:
        cdr["CDRSB"] = _to_numeric(cdr[cdrsb_col])

    baseline_cdr = _nearest_to_target(
        source_df=cdr[[cdr_rid, cdr_date, "CDR_global"]].rename(columns={cdr_rid: "RID", cdr_date: "visit_date"}),
        cohort_df=cohort,
        value_cols=["CDR_global"],
        target_date_col="baseline_date",
        max_abs_days=config.state_window_days,
    ).rename(columns={"CDR_global": "baseline_CDR_global"})

    baseline_df = baseline_df.merge(baseline_cdr, on="RID", how="left")

    # ------------------------------------------------------------------
    # STRUCTURE MODULE
    # ------------------------------------------------------------------
    visit_frames = []

    for src_df, rid_name, date_name in [
        (mmse[[mmse_rid, mmse_date]].copy(), mmse_rid, mmse_date),
        (cdr[[cdr_rid, cdr_date]].copy(), cdr_rid, cdr_date),
    ]:
        tmp = src_df.rename(columns={rid_name: "RID", date_name: "visit_date"})
        visit_frames.append(tmp)

    visits_all = pd.concat(visit_frames, axis=0, ignore_index=True)
    visits_all["RID"] = _to_numeric(visits_all["RID"]).astype("Int64")
    visits_all["visit_date"] = _to_datetime(visits_all["visit_date"])
    visits_all = visits_all.dropna(subset=["RID", "visit_date"]).drop_duplicates()

    structure_rows = []
    for _, row in cohort[["RID", "baseline_date", "landmark_date"]].iterrows():
        rid = row["RID"]
        bdate = row["baseline_date"]
        ldate = row["landmark_date"]

        sub = visits_all[(visits_all["RID"] == rid) &
                         (visits_all["visit_date"] >= bdate) &
                         (visits_all["visit_date"] <= ldate)].copy()

        sub = sub.sort_values("visit_date")
        dates = sub["visit_date"].drop_duplicates().tolist()
        n_visits = len(dates)

        if n_visits >= 1:
            followup_span_days = (dates[-1] - dates[0]).days
        else:
            followup_span_days = np.nan

        if n_visits >= 2:
            gaps = np.diff(pd.Series(dates).values.astype("datetime64[D]")).astype(int)
            median_gap = float(np.median(gaps))
            sd_gap = float(np.std(gaps, ddof=1)) if len(gaps) >= 2 else 0.0
        else:
            median_gap = np.nan
            sd_gap = np.nan

        duration_years = max((ldate - bdate).days / 365.25, 1e-6)
        visits_per_year = n_visits / duration_years

        structure_rows.append(
            {
                "RID": rid,
                "n_visits_total": n_visits,
                "followup_span_days": followup_span_days,
                "visits_per_year": visits_per_year,
                "median_visit_gap_days": median_gap,
                "sd_visit_gap_days": sd_gap,
            }
        )

    structure_df = pd.DataFrame(structure_rows)

    # ------------------------------------------------------------------
    # STATE MODULE
    # ------------------------------------------------------------------
    adas_path = root / config.adas_file
    faq_path = root / config.faq_file

    if not adas_path.exists():
        raise FileNotFoundError(f"ADAS file not found: {adas_path}")
    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ file not found: {faq_path}")

    adas = pd.read_csv(adas_path)
    faq = pd.read_csv(faq_path)

    adas, adas_rid, adas_date = _prepare_generic_table(adas, "ADAS")
    faq, faq_rid, faq_date = _prepare_generic_table(faq, "FAQ")

    adas_col = _find_first_existing(adas.columns, ["TOTAL13", "ADAS13", "TOTSCORE", "TOTAL11"])
    if adas_col is None:
        raise ValueError("Could not find ADAS total score column in ADAS.csv.")
    adas["ADAS"] = _to_numeric(adas[adas_col])

    faq_col = _find_first_existing(faq.columns, ["FAQTOTAL", "FAQ"])
    if faq_col is None:
        raise ValueError("Could not find FAQ total score column in FAQ.csv.")
    faq["FAQ"] = _to_numeric(faq[faq_col])

    state_df = cohort[["RID"]].copy()

    state_mmse = _nearest_before_or_on(
        source_df=mmse[[mmse_rid, mmse_date, "MMSE"]].rename(columns={mmse_rid: "RID", mmse_date: "visit_date"}),
        cohort_df=cohort,
        value_cols=["MMSE"],
        target_date_col="landmark_date",
        window_days=config.state_window_days,
    ).rename(columns={"MMSE": "state_MMSE"})

    state_cdr = _nearest_before_or_on(
        source_df=cdr[[cdr_rid, cdr_date, "CDR_global", "CDRSB"]].rename(columns={cdr_rid: "RID", cdr_date: "visit_date"}),
        cohort_df=cohort,
        value_cols=["CDR_global", "CDRSB"],
        target_date_col="landmark_date",
        window_days=config.state_window_days,
    ).rename(columns={"CDR_global": "state_CDR_global", "CDRSB": "state_CDRSB"})

    state_adas = _nearest_before_or_on(
        source_df=adas[[adas_rid, adas_date, "ADAS"]].rename(columns={adas_rid: "RID", adas_date: "visit_date"}),
        cohort_df=cohort,
        value_cols=["ADAS"],
        target_date_col="landmark_date",
        window_days=config.state_window_days,
    ).rename(columns={"ADAS": "state_ADAS"})

    state_faq = _nearest_before_or_on(
        source_df=faq[[faq_rid, faq_date, "FAQ"]].rename(columns={faq_rid: "RID", faq_date: "visit_date"}),
        cohort_df=cohort,
        value_cols=["FAQ"],
        target_date_col="landmark_date",
        window_days=config.state_window_days,
    ).rename(columns={"FAQ": "state_FAQ"})

    state_df = state_df.merge(state_mmse, on="RID", how="left")
    state_df = state_df.merge(state_cdr, on="RID", how="left")
    state_df = state_df.merge(state_adas, on="RID", how="left")
    state_df = state_df.merge(state_faq, on="RID", how="left")

    # ------------------------------------------------------------------
    # DYNAMICS MODULE
    # ------------------------------------------------------------------
    cohort_6m = cohort[["RID", "landmark_date"]].copy()
    cohort_6m["target_6m_date"] = cohort_6m["landmark_date"] - pd.to_timedelta(config.dynamic_window_6m_days, unit="D")

    cohort_12m = cohort[["RID", "landmark_date"]].copy()
    cohort_12m["target_12m_date"] = cohort_12m["landmark_date"] - pd.to_timedelta(config.dynamic_window_12m_days, unit="D")

    mmse_6m = _nearest_to_target(
        source_df=mmse[[mmse_rid, mmse_date, "MMSE"]].rename(columns={mmse_rid: "RID", mmse_date: "visit_date"}),
        cohort_df=cohort_6m.rename(columns={"target_6m_date": "target_date"}),
        value_cols=["MMSE"],
        target_date_col="target_date",
        max_abs_days=120,
    ).rename(columns={"MMSE": "MMSE_6m_ref"})

    mmse_12m = _nearest_to_target(
        source_df=mmse[[mmse_rid, mmse_date, "MMSE"]].rename(columns={mmse_rid: "RID", mmse_date: "visit_date"}),
        cohort_df=cohort_12m.rename(columns={"target_12m_date": "target_date"}),
        value_cols=["MMSE"],
        target_date_col="target_date",
        max_abs_days=180,
    ).rename(columns={"MMSE": "MMSE_12m_ref"})

    cdr_6m = _nearest_to_target(
        source_df=cdr[[cdr_rid, cdr_date, "CDRSB"]].rename(columns={cdr_rid: "RID", cdr_date: "visit_date"}),
        cohort_df=cohort_6m.rename(columns={"target_6m_date": "target_date"}),
        value_cols=["CDRSB"],
        target_date_col="target_date",
        max_abs_days=120,
    ).rename(columns={"CDRSB": "CDRSB_6m_ref"})

    cdr_12m = _nearest_to_target(
        source_df=cdr[[cdr_rid, cdr_date, "CDRSB"]].rename(columns={cdr_rid: "RID", cdr_date: "visit_date"}),
        cohort_df=cohort_12m.rename(columns={"target_12m_date": "target_date"}),
        value_cols=["CDRSB"],
        target_date_col="target_date",
        max_abs_days=180,
    ).rename(columns={"CDRSB": "CDRSB_12m_ref"})

    adas_6m = _nearest_to_target(
        source_df=adas[[adas_rid, adas_date, "ADAS"]].rename(columns={adas_rid: "RID", adas_date: "visit_date"}),
        cohort_df=cohort_6m.rename(columns={"target_6m_date": "target_date"}),
        value_cols=["ADAS"],
        target_date_col="target_date",
        max_abs_days=120,
    ).rename(columns={"ADAS": "ADAS_6m_ref"})

    adas_12m = _nearest_to_target(
        source_df=adas[[adas_rid, adas_date, "ADAS"]].rename(columns={adas_rid: "RID", adas_date: "visit_date"}),
        cohort_df=cohort_12m.rename(columns={"target_12m_date": "target_date"}),
        value_cols=["ADAS"],
        target_date_col="target_date",
        max_abs_days=180,
    ).rename(columns={"ADAS": "ADAS_12m_ref"})

    faq_6m = _nearest_to_target(
        source_df=faq[[faq_rid, faq_date, "FAQ"]].rename(columns={faq_rid: "RID", faq_date: "visit_date"}),
        cohort_df=cohort_6m.rename(columns={"target_6m_date": "target_date"}),
        value_cols=["FAQ"],
        target_date_col="target_date",
        max_abs_days=120,
    ).rename(columns={"FAQ": "FAQ_6m_ref"})

    faq_12m = _nearest_to_target(
        source_df=faq[[faq_rid, faq_date, "FAQ"]].rename(columns={faq_rid: "RID", faq_date: "visit_date"}),
        cohort_df=cohort_12m.rename(columns={"target_12m_date": "target_date"}),
        value_cols=["FAQ"],
        target_date_col="target_date",
        max_abs_days=180,
    ).rename(columns={"FAQ": "FAQ_12m_ref"})

    dynamics_df = state_df[["RID", "state_MMSE", "state_CDRSB", "state_ADAS", "state_FAQ"]].copy()

    dynamics_df = dynamics_df.merge(mmse_6m, on="RID", how="left")
    dynamics_df = dynamics_df.merge(mmse_12m, on="RID", how="left")
    dynamics_df = dynamics_df.merge(cdr_6m, on="RID", how="left")
    dynamics_df = dynamics_df.merge(cdr_12m, on="RID", how="left")
    dynamics_df = dynamics_df.merge(adas_6m, on="RID", how="left")
    dynamics_df = dynamics_df.merge(adas_12m, on="RID", how="left")
    dynamics_df = dynamics_df.merge(faq_6m, on="RID", how="left")
    dynamics_df = dynamics_df.merge(faq_12m, on="RID", how="left")

    dynamics_df["MMSE_delta_6m"] = dynamics_df["state_MMSE"] - dynamics_df["MMSE_6m_ref"]
    dynamics_df["MMSE_delta_12m"] = dynamics_df["state_MMSE"] - dynamics_df["MMSE_12m_ref"]

    dynamics_df["CDRSB_delta_6m"] = dynamics_df["state_CDRSB"] - dynamics_df["CDRSB_6m_ref"]
    dynamics_df["CDRSB_delta_12m"] = dynamics_df["state_CDRSB"] - dynamics_df["CDRSB_12m_ref"]

    dynamics_df["ADAS_delta_6m"] = dynamics_df["state_ADAS"] - dynamics_df["ADAS_6m_ref"]
    dynamics_df["ADAS_delta_12m"] = dynamics_df["state_ADAS"] - dynamics_df["ADAS_12m_ref"]

    dynamics_df["FAQ_delta_6m"] = dynamics_df["state_FAQ"] - dynamics_df["FAQ_6m_ref"]
    dynamics_df["FAQ_delta_12m"] = dynamics_df["state_FAQ"] - dynamics_df["FAQ_12m_ref"]

    dynamics_df = dynamics_df[
        [
            "RID",
            "MMSE_delta_6m",
            "MMSE_delta_12m",
            "CDRSB_delta_6m",
            "CDRSB_delta_12m",
            "ADAS_delta_6m",
            "ADAS_delta_12m",
            "FAQ_delta_6m",
            "FAQ_delta_12m",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # FINAL CLEANUP + SAVE
    # ------------------------------------------------------------------
    baseline_df = baseline_df[
        ["RID", "age", "sex", "education", "APOE4", "baseline_MMSE", "baseline_CDR_global"]
    ].copy()

    structure_df = structure_df[
        ["RID", "n_visits_total", "followup_span_days", "visits_per_year", "median_visit_gap_days", "sd_visit_gap_days"]
    ].copy()

    state_df = state_df[
        ["RID", "state_MMSE", "state_CDR_global", "state_CDRSB", "state_ADAS", "state_FAQ"]
    ].copy()

    baseline_path = root / config.baseline_output
    structure_path = root / config.structure_output
    state_path = root / config.state_output
    dynamics_path = root / config.dynamics_output

    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_df.to_csv(baseline_path, index=False)
    structure_df.to_csv(structure_path, index=False)
    state_df.to_csv(state_path, index=False)
    dynamics_df.to_csv(dynamics_path, index=False)

    return {
        "baseline": baseline_df,
        "structure": structure_df,
        "state": state_df,
        "dynamics": dynamics_df,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cfg = ModuleConfig(project_root=root)
    outputs = build_adni_modules(cfg)

    print("\n" + "=" * 80)
    print("ADNI modules built successfully")
    print("=" * 80)

    for name, df in outputs.items():
        print(f"\n{name.upper()}")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(5))
        print("\nMissing values:")
        print(df.isna().sum())