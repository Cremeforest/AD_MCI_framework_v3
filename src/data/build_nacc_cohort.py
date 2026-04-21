from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =========================================================
# Configuration
# =========================================================

PSEUDO_MISSING_NUMERIC = {-4, 88, 95, 96, 97, 98, 99, 888, 8888, 999, 9999}


@dataclass
class NACCBuildConfig:
    input_csv: Path
    output_csv: Path
    min_followup_days: int = 30
    require_mci_at_baseline: bool = True
    event_definition: str = "dementia_or_ad"
    verbose: bool = True


# =========================================================
# Utilities
# =========================================================

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def replace_pseudo_missing(
    df: pd.DataFrame,
    columns: Iterable[str],
    pseudo_values: Optional[set] = None,
) -> pd.DataFrame:
    """
    Replace pseudo-missing codes in selected columns with NaN.
    Only apply this to columns where these codes truly represent missing values.
    """
    pseudo_values = pseudo_values or PSEUDO_MISSING_NUMERIC
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(list(pseudo_values), np.nan)
    return df


def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end - start).dt.days / 30.4375


def map_sex_binary(series: pd.Series) -> pd.Series:
    """
    NACCSEX commonly: 1=Male, 2=Female
    map to:
      male = 1
      female = 0
    """
    out = series.copy()
    out = out.replace({1: 1, 2: 0})
    out = pd.to_numeric(out, errors="coerce")
    out = out.where(out.isin([0, 1]), np.nan)
    return out


def map_apoe4_count(series: pd.Series) -> pd.Series:
    """
    Placeholder mapping for NACCAPOE.
    IMPORTANT:
    You should verify exact NACC coding in your data dictionary.
    
    Common practical use:
      convert to integer if already encoded as 0/1/2 risk allele count
      otherwise keep as-is for inspection.
    """
    out = pd.to_numeric(series, errors="coerce")
    # If values are only in 1/2/3/4 style categories, do not force mapping blindly.
    # Just keep raw for now unless confirmed.
    return out


# =========================================================
# Diagnosis helpers
# =========================================================

def build_baseline_mci_flag(df: pd.DataFrame) -> pd.Series:
    """
    Baseline MCI flag for v3.

    Preferred rule:
    - NACCUDSD == 3  --> MCI

    Additional exclusions kept for safety:
    - DEMENTED != 1
    - NORMCOG != 1

    NOTE:
    This is much more reliable than using MCI == 1 in this dataset,
    because the MCI column is mostly unavailable (-4).
    """
    if "NACCUDSD" not in df.columns:
        return pd.Series(False, index=df.index)

    uds = pd.to_numeric(df["NACCUDSD"], errors="coerce")

    mci_flag = (uds == 3)

    if "DEMENTED" in df.columns:
        mci_flag = mci_flag & ~(pd.to_numeric(df["DEMENTED"], errors="coerce") == 1)

    if "NORMCOG" in df.columns:
        mci_flag = mci_flag & ~(pd.to_numeric(df["NORMCOG"], errors="coerce") == 1)

    return mci_flag


def build_event_flag(df: pd.DataFrame, event_definition: str = "uds_dementia") -> pd.Series:
    """
    Event definition for follow-up progression.

    v4 preferred rule:
    uds_dementia:
      event if NACCUDSD == 4

    Interpretation based on observed code distribution:
      NACCUDSD == 1 -> normal cognition
      NACCUDSD == 3 -> MCI
      NACCUDSD == 4 -> dementia

    This makes the event definition internally consistent with
    baseline MCI defined by NACCUDSD == 3.
    """
    event_flag = pd.Series(False, index=df.index)

    if event_definition == "uds_dementia":
        if "NACCUDSD" in df.columns:
            event_flag = pd.to_numeric(df["NACCUDSD"], errors="coerce") == 4

    return event_flag


# =========================================================
# Core pipeline
# =========================================================

def load_nacc_csv(input_csv: Path, verbose: bool = True) -> pd.DataFrame:
    log("=" * 80, verbose)
    log(f"Loading NACC data from: {input_csv}", verbose)
    log("=" * 80, verbose)

    df = pd.read_csv(input_csv, low_memory=False)
    log(f"Loaded shape: {df.shape}", verbose)
    return df


def select_core_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    candidate_cols = [
        "NACCID",
        "VISITDATE",
        "NACCAGE",
        "NACCSEX",
        "EDUC",
        "NACCMMSE",
        "CDRGLOB",
        "CDRSUM",
        "NACCAPOE",
        "MCI",
        "NACCTMCI",
        "DEMENTED",
        "NORMCOG",
        "NACCALZD",
        "NACCUDSD",
    ]

    keep_cols = [c for c in candidate_cols if c in df.columns]
    out = df[keep_cols].copy()

    log(f"Selected core columns ({len(keep_cols)}): {keep_cols}", verbose)
    log(f"Core table shape: {out.shape}", verbose)
    return out


def clean_core_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log("=" * 80, verbose)
    log("Cleaning core dataframe...", verbose)
    log("=" * 80, verbose)

    # Replace pseudo-missing only in columns where this is appropriate
    pseudo_missing_cols = [
        "NACCAGE",
        "NACCSEX",
        "EDUC",
        "NACCMMSE",
        "CDRGLOB",
        "CDRSUM",
        "NACCAPOE",
        "MCI",
        "NACCTMCI",
        "DEMENTED",
        "NORMCOG",
        "NACCALZD",
        "NACCUDSD",
    ]
    df = replace_pseudo_missing(df, pseudo_missing_cols)

    # Date
    if "VISITDATE" in df.columns:
        df["VISITDATE"] = to_datetime_safe(df["VISITDATE"])

    # Numeric conversions
    numeric_cols = [
        "NACCAGE",
        "EDUC",
        "NACCMMSE",
        "CDRGLOB",
        "CDRSUM",
        "NACCAPOE",
        "MCI",
        "NACCTMCI",
        "DEMENTED",
        "NORMCOG",
        "NACCALZD",
        "NACCUDSD",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical recoding
    if "NACCSEX" in df.columns:
        df["sex_binary"] = map_sex_binary(df["NACCSEX"])

    if "NACCAPOE" in df.columns:
        df["apoe_raw"] = map_apoe4_count(df["NACCAPOE"])

    # Sort
    if "NACCID" in df.columns and "VISITDATE" in df.columns:
        df = df.sort_values(["NACCID", "VISITDATE"]).reset_index(drop=True)

    log("Finished cleaning.", verbose)
    return df


def add_diagnosis_flags(df: pd.DataFrame, event_definition: str, verbose: bool = True) -> pd.DataFrame:
    log("=" * 80, verbose)
    log("Building diagnosis flags...", verbose)
    log("=" * 80, verbose)

    df = df.copy()
    df["is_mci_like"] = build_baseline_mci_flag(df)
    df["is_event"] = build_event_flag(df, event_definition=event_definition)

    log(f"Strict baseline MCI rows: {int(df['is_mci_like'].sum())}", verbose)
    log(f"Event rows: {int(df['is_event'].sum())}", verbose)
    return df


def build_subject_level_cohort(df: pd.DataFrame, config: NACCBuildConfig) -> pd.DataFrame:
    """
    Build one row per subject:
      - baseline visit = first MCI-like visit
      - event date = first later visit meeting event definition
      - censor date = last available visit if no event
    """
    verbose = config.verbose
    log("=" * 80, verbose)
    log("Building subject-level NACC cohort...", verbose)
    log("=" * 80, verbose)

    required_cols = ["NACCID", "VISITDATE"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    rows = []

    grouped = df.groupby("NACCID", sort=False)

    for naccid, g in grouped:
        g = g.sort_values("VISITDATE").copy()

        # remove rows without date
        g = g[g["VISITDATE"].notna()].copy()
        if g.empty:
            continue

        # find baseline candidate
        if config.require_mci_at_baseline:
            baseline_candidates = g[g["is_mci_like"]].copy()
        else:
            baseline_candidates = g.copy()

        if baseline_candidates.empty:
            continue

        baseline = baseline_candidates.iloc[0]
        baseline_date = baseline["VISITDATE"]

        # follow-up after baseline
        future = g[g["VISITDATE"] > baseline_date].copy()

        if future.empty:
            continue

        # first event after baseline
        future_events = future[future["is_event"]].copy()

        if not future_events.empty:
            first_event = future_events.iloc[0]
            end_date = first_event["VISITDATE"]
            event = 1
        else:
            end_date = future["VISITDATE"].max()
            event = 0

        followup_days = (end_date - baseline_date).days
        if pd.isna(followup_days) or followup_days < config.min_followup_days:
            continue

        row = {
            "NACCID": naccid,
            "baseline_date": baseline_date,
            "end_date": end_date,
            "time_from_baseline_months": followup_days / 30.4375,
            "event": event,

            # baseline features
            "age_bl": baseline.get("NACCAGE", np.nan),
            "sex_bl": baseline.get("sex_binary", np.nan),
            "educ_bl": baseline.get("EDUC", np.nan),
            "apoe_bl_raw": baseline.get("apoe_raw", np.nan),
            "mmse_bl": baseline.get("NACCMMSE", np.nan),
            "cdrglob_bl": baseline.get("CDRGLOB", np.nan),
            "cdrsum_bl": baseline.get("CDRSUM", np.nan),

            # state features at last observed point / event point
            "mmse_state": first_event["NACCMMSE"] if event == 1 else future.iloc[-1].get("NACCMMSE", np.nan),
            "cdrglob_state": first_event["CDRGLOB"] if event == 1 else future.iloc[-1].get("CDRGLOB", np.nan),
            "cdrsum_state": first_event["CDRSUM"] if event == 1 else future.iloc[-1].get("CDRSUM", np.nan),

            # bookkeeping
            "n_total_visits": len(g),
            "n_future_visits": len(future),
        }

        rows.append(row)

    cohort = pd.DataFrame(rows)

    if not cohort.empty:
        cohort = cohort.sort_values(["baseline_date", "NACCID"]).reset_index(drop=True)

    log(f"Built subject-level cohort shape: {cohort.shape}", verbose)
    return cohort


def add_module_availability_flags(cohort: pd.DataFrame) -> pd.DataFrame:
    cohort = cohort.copy()

    if cohort.empty:
        return cohort

    baseline_cols = ["age_bl", "sex_bl", "educ_bl", "apoe_bl_raw", "mmse_bl", "cdrglob_bl"]
    cohort["avail_baseline"] = cohort[baseline_cols].notna().any(axis=1).astype(int)

    state_cols = ["mmse_state", "cdrglob_state", "cdrsum_state"]
    cohort["avail_state"] = cohort[state_cols].notna().any(axis=1).astype(int)

    cohort["avail_structure"] = 0
    cohort["avail_dynamics"] = 0

    return cohort


def summarize_cohort(cohort: pd.DataFrame, verbose: bool = True) -> None:
    log("=" * 80, verbose)
    log("Cohort summary", verbose)
    log("=" * 80, verbose)

    if cohort.empty:
        log("Cohort is empty.", verbose)
        return

    print(cohort.head(10))
    print()

    print("N subjects:", len(cohort))
    print("Events:", int(cohort["event"].sum()))
    print("Event rate:", round(cohort["event"].mean(), 4))
    print("Median follow-up months:", round(cohort["time_from_baseline_months"].median(), 2))
    print()

    missing_summary = cohort.isna().mean().sort_values(ascending=False)
    print("Top missingness:")
    print(missing_summary.head(15))


def build_nacc_cohort(config: NACCBuildConfig) -> pd.DataFrame:
    df = load_nacc_csv(config.input_csv, verbose=config.verbose)
    df = select_core_columns(df, verbose=config.verbose)
    df = clean_core_dataframe(df, verbose=config.verbose)
    df = add_diagnosis_flags(df, event_definition=config.event_definition, verbose=config.verbose)
    cohort = build_subject_level_cohort(df, config)
    cohort = add_module_availability_flags(cohort)
    summarize_cohort(cohort, verbose=config.verbose)

    ensure_parent_dir(config.output_csv)
    cohort.to_csv(config.output_csv, index=False)

    log("=" * 80, config.verbose)
    log(f"Saved cohort to: {config.output_csv}", config.verbose)
    log("=" * 80, config.verbose)

    return cohort