"""Data ingestion and validation for load and solar time series."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _parse_load_ts(s: str) -> pd.Timestamp:
    s = s.replace(" CST", "").replace(" CDT", "")
    return pd.Timestamp(s)


def load_load(csv_path: str | Path, time_col: str, kw_cols: list[str], tz: str = "America/Chicago") -> pd.DataFrame:
    """Load the site load CSV. Returns hourly DataFrame with MDP1, MDP2, combined_kw columns,
    tz-aware DatetimeIndex in America/Chicago.
    """
    raw = pd.read_csv(csv_path)
    raw[time_col] = raw[time_col].map(_parse_load_ts)
    # localize; handle DST fall-back duplicates
    ts = raw[time_col]
    try:
        ts_tz = ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        ts_tz = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    df = pd.DataFrame(index=ts_tz)
    for i, c in enumerate(kw_cols):
        df[f"mdp{i+1}"] = raw[c].values
    df["combined_kw"] = df[[f"mdp{i+1}" for i in range(len(kw_cols))]].sum(axis=1, min_count=1)
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    full = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz=tz)
    df = df.reindex(full)

    for c in df.columns:
        df[c] = df[c].interpolate(method="time", limit=2)
        df[c] = df[c].ffill().bfill()
    df.index.name = "timestamp"
    return df


def load_solar(csv_path: str | Path, time_col: str, prod_cols: list[str], tz: str = "America/Chicago") -> pd.DataFrame:
    """Load solar CSV. Returns tz-aware hourly DataFrame with per-meter and combined_kw columns.
    Solar values are hourly kWh (= average kW over the hour)."""
    raw = pd.read_csv(csv_path)
    raw[time_col] = pd.to_datetime(raw[time_col], format="%m-%d-%Y %H:%M:%S")
    ts_tz = raw[time_col].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    df = pd.DataFrame(index=ts_tz)
    for i, c in enumerate(prod_cols):
        df[f"meter{i+1}_kw"] = raw[c].clip(lower=0).values  # clip tiny negatives
    df["combined_kw"] = df[[f"meter{i+1}_kw" for i in range(len(prod_cols))]].sum(axis=1)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "timestamp"
    return df


def validate(load_df: pd.DataFrame, solar_df: pd.DataFrame | None = None) -> dict:
    """Return a data-quality report dict."""
    r = {}
    r["load_start"] = str(load_df.index.min())
    r["load_end"] = str(load_df.index.max())
    r["load_rows"] = len(load_df)
    r["load_missing_hours"] = int(load_df["combined_kw"].isna().sum())
    r["annual_kwh"] = float(load_df["combined_kw"].sum())  # hourly kW = kWh
    r["peak_kw"] = float(load_df["combined_kw"].max())
    r["load_factor"] = r["annual_kwh"] / (r["peak_kw"] * len(load_df)) if r["peak_kw"] > 0 else 0.0
    r["mdp1_peak_kw"] = float(load_df["mdp1"].max())
    r["mdp2_peak_kw"] = float(load_df["mdp2"].max())
    if solar_df is not None:
        r["solar_start"] = str(solar_df.index.min())
        r["solar_end"] = str(solar_df.index.max())
        r["solar_rows"] = len(solar_df)
        r["solar_measured_kwh"] = float(solar_df["combined_kw"].sum())
        r["solar_peak_ac_kw"] = float(solar_df["combined_kw"].max())
    return r
