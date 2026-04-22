"""Perfect-foresight monthly LP for battery dispatch.

Co-optimizes:
  - DFC demand reduction (on-peak 30-min peak per meter, summed) - MONTHLY
  - Energy arbitrage on hourly LMP
  - Penalty on grid import at the PJM 5CP hours (PLC tag reduction)
  - Penalty on grid import at the PJM NSPL hour (transmission tag reduction)
  - Solar self-consumption (by minimizing clipped export)

We solve a monthly LP (not yearly) because:
  (a) DFC demand is monthly, natural segment for the peak constraint
  (b) SOC chains across months via a terminal-SOC constraint
  (c) 8760-hour MILP is borderline; 12 x 744-hour LPs are fast in HiGHS.

Output: hourly battery charge/discharge, grid import, and SOC trajectory.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from ..battery import BatterySpec
from ..tariff.comed_delivery import on_peak_mask


@dataclass
class DispatchResult:
    df: pd.DataFrame          # hourly trajectory
    monthly_peaks: pd.DataFrame


def _solve_month(
    month_df: pd.DataFrame,           # index + columns: load_kw, solar_kw, lmp (imported)
    battery: BatterySpec,
    soc_init_kwh: float,
    dfc_per_kw: float,
    plc_hours: set[pd.Timestamp],
    nspl_hours: set[pd.Timestamp],
    plc_penalty_per_kw: float,
    nspl_penalty_per_kw: float,
    export_allowed: bool,
    export_rate_per_kwh: float,
    terminal_soc_kwh: float | None = None,
    demand_on_peak_only: bool = True,  # ComEd VLL: True; PECO HT: False (max anytime)
) -> pd.DataFrame:
    """Monthly LP. Returns hourly trajectory DataFrame."""
    m = pyo.ConcreteModel()
    T = list(range(len(month_df)))
    m.T = pyo.Set(initialize=T, ordered=True)

    load = month_df["load_kw"].values
    solar = month_df["solar_kw"].values
    lmp = month_df["lmp"].values
    dt = 1.0  # hour

    on_peak = on_peak_mask(month_df.index).values

    # Variables
    m.p_chg = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, battery.power_kw))
    m.p_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, battery.power_kw))
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals,
                    bounds=(battery.soc_min_kwh, battery.soc_max_kwh))
    m.grid_import = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.grid_export = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    # Curtailment: any solar we can't absorb when export is disallowed
    def _curtail_bounds(m, t):
        return (0, max(0.0, float(solar[t])))
    m.solar_curtail = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=_curtail_bounds)
    # Monthly peak kW (billed during on-peak hours only)
    m.peak_kw = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 1e6))

    eta_c = battery.eta_chg
    eta_d = battery.eta_dis

    # Energy balance at POI: grid_import + (solar - curtail) + p_dis = load + p_chg + grid_export
    def balance(m, t):
        return m.grid_import[t] + (solar[t] - m.solar_curtail[t]) + m.p_dis[t] == load[t] + m.p_chg[t] + m.grid_export[t]
    m.balance = pyo.Constraint(m.T, rule=balance)

    # SOC dynamics
    def soc_rule(m, t):
        if t == 0:
            soc_prev = soc_init_kwh
        else:
            soc_prev = m.soc[t - 1]
        aux = battery.aux_kw * dt
        return m.soc[t] == soc_prev + (eta_c * m.p_chg[t] - m.p_dis[t] / eta_d) * dt - aux
    m.soc_dyn = pyo.Constraint(m.T, rule=soc_rule)

    # Optional terminal SOC
    if terminal_soc_kwh is not None:
        m.terminal = pyo.Constraint(expr=m.soc[T[-1]] >= terminal_soc_kwh)

    # No-export if disallowed
    if not export_allowed:
        def noexport(m, t):
            return m.grid_export[t] == 0
        m.noexport = pyo.Constraint(m.T, rule=noexport)

    # Peak tracking. ComEd (VLL) bills DFC on on-peak max only. PECO (HT) bills on
    # overall max regardless of hour. Flag controls which.
    def peak_rule(m, t):
        if demand_on_peak_only and not on_peak[t]:
            return pyo.Constraint.Skip
        return m.peak_kw >= m.grid_import[t]
    m.peak_c = pyo.Constraint(m.T, rule=peak_rule)

    # PLC / NSPL penalty: encourage grid_import ~ 0 at these hours
    plc_idx = [i for i, ts in enumerate(month_df.index) if ts in plc_hours]
    nspl_idx = [i for i, ts in enumerate(month_df.index) if ts in nspl_hours]

    # Objective: minimize total cost
    def obj_rule(m):
        energy_cost = sum(m.grid_import[t] * lmp[t] for t in T) * dt
        export_rev = sum(m.grid_export[t] * export_rate_per_kwh for t in T) * dt
        demand_cost = m.peak_kw * dfc_per_kw
        plc_cost = sum(m.grid_import[t] * plc_penalty_per_kw for t in plc_idx)
        nspl_cost = sum(m.grid_import[t] * nspl_penalty_per_kw for t in nspl_idx)
        return energy_cost + demand_cost + plc_cost + nspl_cost - export_rev
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory("appsi_highs")
    solver.solve(m, tee=False)

    out = pd.DataFrame(index=month_df.index)
    out["load_kw"] = load
    out["solar_kw_available"] = solar
    out["lmp"] = lmp
    out["p_chg"] = [pyo.value(m.p_chg[t]) for t in T]
    out["p_dis"] = [pyo.value(m.p_dis[t]) for t in T]
    out["soc_kwh"] = [pyo.value(m.soc[t]) for t in T]
    out["grid_import"] = [pyo.value(m.grid_import[t]) for t in T]
    out["grid_export"] = [pyo.value(m.grid_export[t]) for t in T]
    out["solar_curtail"] = [pyo.value(m.solar_curtail[t]) for t in T]
    out["solar_kw"] = out["solar_kw_available"] - out["solar_curtail"]
    out["peak_kw_month"] = pyo.value(m.peak_kw)
    return out


def perfect_foresight_dispatch(
    year_df: pd.DataFrame,                # columns: load_kw, solar_kw, lmp; hourly index
    battery: BatterySpec,
    dfc_per_kw_monthly: dict,             # {month: $/kW}
    plc_hours: Sequence[pd.Timestamp],
    nspl_hours: Sequence[pd.Timestamp],
    mpc_cfg: dict,
    export_allowed: bool = False,
    export_rate_per_kwh: float = 0.0,
    demand_on_peak_only: bool = True,     # ComEd True, PECO False
) -> DispatchResult:
    """Solve the full-year perfect-foresight dispatch month by month.

    Returns hourly trajectory and monthly peak summary.
    """
    plc_set = set(pd.Timestamp(ts) for ts in plc_hours)
    nspl_set = set(pd.Timestamp(ts) for ts in nspl_hours)

    results = []
    soc = battery.soc_init_kwh
    by_month = list(year_df.groupby(pd.Grouper(freq="MS")))
    for i, (month_start, month_df) in enumerate(by_month):
        if len(month_df) == 0:
            continue
        m_idx = month_start.month
        dfc = dfc_per_kw_monthly[m_idx]
        # Terminal SOC: mid-range at end of month so next month can start fresh.
        # Skip on the last month.
        terminal = None if i == len(by_month) - 1 else battery.energy_kwh * 0.5
        out = _solve_month(
            month_df, battery,
            soc_init_kwh=soc,
            dfc_per_kw=dfc,
            plc_hours=plc_set,
            nspl_hours=nspl_set,
            plc_penalty_per_kw=mpc_cfg.get("plc_penalty_per_kw_above_target", 10000.0),
            nspl_penalty_per_kw=mpc_cfg.get("nspl_penalty_per_kw_above_target", 5000.0),
            export_allowed=export_allowed,
            export_rate_per_kwh=export_rate_per_kwh,
            terminal_soc_kwh=terminal,
            demand_on_peak_only=demand_on_peak_only,
        )
        results.append(out)
        soc = float(out["soc_kwh"].iloc[-1])

    full = pd.concat(results)
    monthly_peaks = full.groupby(pd.Grouper(freq="MS"))["peak_kw_month"].first()
    return DispatchResult(df=full, monthly_peaks=monthly_peaks.to_frame("peak_kw"))
