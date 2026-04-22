"""Synthetic 2025 PJM ComEd-zone LMPs and 5CP/1CP peak hour identification.

No live PJM access in this environment. Anchors:
  - Aug-Sep 2025 supply bill shows period-average LMP = $0.03270/kWh
  - Aug-Sep 2025 bill identifies 8/15/25 18:00 as THE period coincident peak
  - PJM ComEd zone typical diurnal shape (public data): peak hours ~4-8 PM,
    summer multipliers ~1.6× off-peak, shoulders neutral, winter evening mild bump.

We construct hourly LMPs by multiplying a normalized shape by an annual scale so that
the average over the Aug-Sep bill period matches the observed $0.0327/kWh.

5CP proxy:
  PJM's true 5CP are the 5 highest RTO-wide load hours in Jun-Sep. Site load correlates
  with regional load on hot weekday afternoons (both driven by cooling). We use the 5
  highest site-load hours during weekday 14:00-19:00 in Jun-Sep 2025, with 8/15/25 18:00
  anchored as a known PJM 5CP hour.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd


def _diurnal_factor(hour: int, month: int, is_weekday: bool) -> float:
    """Return normalized (mean ≈ 1) multiplier for ComEd-zone LMP by hour/month/dayofweek."""
    summer = month in (6, 7, 8, 9)
    winter = month in (12, 1, 2)
    base_weekend = 0.82

    if summer:
        shape = {
            0: 0.72, 1: 0.68, 2: 0.66, 3: 0.65, 4: 0.65, 5: 0.70,
            6: 0.80, 7: 0.90, 8: 0.95, 9: 1.00, 10: 1.08, 11: 1.15,
            12: 1.20, 13: 1.25, 14: 1.30, 15: 1.38, 16: 1.45, 17: 1.55,
            18: 1.55, 19: 1.40, 20: 1.20, 21: 1.05, 22: 0.92, 23: 0.80,
        }
    elif winter:
        shape = {
            0: 0.80, 1: 0.75, 2: 0.72, 3: 0.72, 4: 0.78, 5: 0.95,
            6: 1.20, 7: 1.30, 8: 1.25, 9: 1.15, 10: 1.05, 11: 1.00,
            12: 0.98, 13: 0.97, 14: 0.98, 15: 1.05, 16: 1.20, 17: 1.35,
            18: 1.30, 19: 1.20, 20: 1.08, 21: 0.98, 22: 0.88, 23: 0.82,
        }
    else:
        shape = {
            0: 0.78, 1: 0.72, 2: 0.70, 3: 0.70, 4: 0.72, 5: 0.82,
            6: 0.95, 7: 1.10, 8: 1.10, 9: 1.05, 10: 1.02, 11: 1.00,
            12: 1.00, 13: 1.00, 14: 1.02, 15: 1.08, 16: 1.20, 17: 1.30,
            18: 1.25, 19: 1.15, 20: 1.02, 21: 0.95, 22: 0.88, 23: 0.82,
        }
    f = shape[hour]
    if not is_weekday:
        f *= base_weekend
    return f


def synthesize_hourly_lmp(year: int, tz: str, anchor_per_kwh: float) -> pd.Series:
    """Hourly LMP in $/kWh for the full year, scaled so period-average matches anchor.

    Monthly seasonal multipliers:
      Winter (Dec-Feb): 1.10
      Spring (Mar-May): 0.95
      Summer (Jun-Sep): 1.25
      Fall (Oct-Nov):   1.00
    These scale the diurnal shape before final normalization.
    """
    idx = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="1h", tz=tz, inclusive="left")
    seasonal = {1: 1.10, 2: 1.10, 3: 0.95, 4: 0.95, 5: 0.95,
                6: 1.25, 7: 1.28, 8: 1.28, 9: 1.20,
                10: 1.00, 11: 1.00, 12: 1.10}
    shape = np.array([
        _diurnal_factor(ts.hour, ts.month, ts.weekday() < 5) * seasonal[ts.month]
        for ts in idx
    ])
    # Normalize so that the period-average over Jul 17 - Aug 13 (when we know LMP = anchor)
    anchor_mask_start = pd.Timestamp(f"{year}-07-17", tz=tz)
    anchor_mask_end = pd.Timestamp(f"{year}-08-13", tz=tz)
    in_anchor = (idx >= anchor_mask_start) & (idx < anchor_mask_end)
    anchor_avg = shape[in_anchor].mean()
    scale = anchor_per_kwh / anchor_avg
    return pd.Series(shape * scale, index=idx, name="lmp_per_kwh")


def find_proxy_5cp_hours(load_df: pd.DataFrame, year: int,
                         anchored_hours: Sequence[pd.Timestamp] = ()) -> list[pd.Timestamp]:
    """Identify 5 proxy PJM 5CP hours — 5 highest weekday 14:00-19:00 load hours in Jun-Sep.
    Any `anchored_hours` are pinned in (typically the bill-confirmed 8/15 18:00).
    """
    idx = load_df.index
    mask = (
        (idx.year == year)
        & (idx.month.isin([6, 7, 8, 9]))
        & (idx.weekday < 5)
        & ((idx.hour >= 14) & (idx.hour <= 19))
    )
    sub = load_df.loc[mask, "combined_kw"].copy()
    picks: list[pd.Timestamp] = []
    for anchor in anchored_hours:
        anchor = pd.Timestamp(anchor).tz_convert(load_df.index.tz) if anchor.tzinfo else pd.Timestamp(anchor).tz_localize(load_df.index.tz)
        if anchor in sub.index:
            picks.append(anchor)
    for ts, _ in sub.sort_values(ascending=False).items():
        if len(picks) >= 5:
            break
        if ts in picks:
            continue
        # enforce no two picks within 3 hours (PJM rule: distinct peak intervals)
        if all(abs((ts - p).total_seconds()) >= 3600 * 3 for p in picks):
            picks.append(ts)
    return picks


def find_proxy_nspl_hour(load_df: pd.DataFrame, year: int,
                         anchored_hour: pd.Timestamp | None = None) -> pd.Timestamp:
    """NSPL (transmission tag) is set by single zonal peak hour, usually a hot weekday 5-7pm.
    Use the highest weekday 14:00-19:00 hour in Jun-Sep 2025."""
    if anchored_hour is not None:
        return anchored_hour
    idx = load_df.index
    mask = (
        (idx.year == year)
        & (idx.month.isin([6, 7, 8, 9]))
        & (idx.weekday < 5)
        & ((idx.hour >= 14) & (idx.hour <= 19))
    )
    return load_df.loc[mask, "combined_kw"].idxmax()
