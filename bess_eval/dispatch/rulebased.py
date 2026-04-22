"""Rule-based (reactive) dispatch. No forecasts — floor case."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..battery import BatterySpec
from ..tariff.comed_delivery import on_peak_mask


def rule_based_dispatch(
    year_df: pd.DataFrame,
    battery: BatterySpec,
    demand_threshold_ratio: float = 0.80,
    demand_on_peak_only: bool = True,
) -> pd.DataFrame:
    """Simple heuristic dispatch:
      - If solar > load: charge battery with excess (to max P_chg)
      - Else, if on-peak AND load > threshold × running_month_peak: discharge (up to P_dis)
      - Else: hold
    Monthly peak threshold ratchets up during the month.
    """
    net_load = (year_df["load_kw"] - year_df["solar_kw"]).values
    if demand_on_peak_only:
        on_peak = on_peak_mask(year_df.index).values
    else:
        on_peak = np.ones(len(year_df), dtype=bool)
    n = len(year_df)

    soc = np.zeros(n)
    p_chg = np.zeros(n)
    p_dis = np.zeros(n)
    grid_imp = np.zeros(n)

    current_soc = battery.soc_init_kwh
    month_peak = 0.0
    last_month = year_df.index[0].month
    threshold = 0.0

    for t in range(n):
        m = year_df.index[t].month
        if m != last_month:
            last_month = m
            month_peak = 0.0
            threshold = 0.0

        nl = net_load[t]
        # Ratchet threshold up
        if on_peak[t] and nl > month_peak:
            month_peak = max(month_peak, nl)
        threshold = demand_threshold_ratio * month_peak if month_peak > 0 else 0.0

        chg = 0.0
        dis = 0.0
        if nl < 0:
            # Solar excess — charge up to battery capacity/headroom
            headroom = battery.soc_max_kwh - current_soc
            want = min(-nl, battery.power_kw, headroom / battery.eta_chg)
            chg = max(0.0, want)
        elif on_peak[t] and nl > threshold and threshold > 0:
            avail = current_soc - battery.soc_min_kwh
            want = min(nl - threshold, battery.power_kw, avail * battery.eta_dis)
            dis = max(0.0, want)

        aux = battery.aux_kw
        current_soc = current_soc + battery.eta_chg * chg - dis / battery.eta_dis - aux
        current_soc = max(battery.soc_min_kwh, min(battery.soc_max_kwh, current_soc))

        p_chg[t] = chg
        p_dis[t] = dis
        soc[t] = current_soc
        grid_imp[t] = max(0.0, nl + chg - dis)

    out = pd.DataFrame(index=year_df.index)
    out["load_kw"] = year_df["load_kw"].values
    out["solar_kw"] = year_df["solar_kw"].values
    out["lmp"] = year_df["lmp"].values
    out["p_chg"] = p_chg
    out["p_dis"] = p_dis
    out["soc_kwh"] = soc
    out["grid_import"] = grid_imp
    out["grid_export"] = np.maximum(0.0, -(net_load + p_chg - p_dis))
    return out
