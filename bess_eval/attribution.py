"""Per-stream value attribution from a dispatch trajectory.

Computes annual $ savings attributable to the battery by comparing:
  - Baseline: load + solar, no battery
  - With battery: same inputs, battery dispatched per the chosen controller

All costs flow through the same tariff engines (delivery + supply), so the delta
is a clean battery-only value.
"""
from __future__ import annotations
import calendar
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .battery import BatterySpec
from .tariff.comed_delivery import ComEdVLLDelivery, on_peak_mask
from .tariff.supply_index import IndexSupply


@dataclass
class AnnualResult:
    scenario: str
    delivery_cost: float
    supply_cost: float
    capacity_tag_cost: float
    transmission_tag_cost: float
    energy_cost: float
    other_supply: float
    per_stream: dict = field(default_factory=dict)
    monthly_bills: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.delivery_cost + self.supply_cost

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "total": self.total,
            "delivery": self.delivery_cost,
            "supply": self.supply_cost,
            "capacity_tag": self.capacity_tag_cost,
            "transmission_tag": self.transmission_tag_cost,
            "energy_cost": self.energy_cost,
            "other_supply": self.other_supply,
            "per_stream": self.per_stream,
        }


def allocate_battery_to_meters(df: pd.DataFrame, mdp_cols: Sequence[str]) -> pd.DataFrame:
    """Distribute battery_delta[t] = p_dis[t] - p_chg[t] across meters proportionally to
    each meter's load share at time t. Returns DataFrame with mdpX_net columns (kW seen at
    each meter after battery + solar allocation)."""
    battery_delta = df["p_dis"] - df["p_chg"]
    gross = df[list(mdp_cols)].sum(axis=1)
    solar = df["solar_kw"]
    # Allocate solar proportionally too
    out = pd.DataFrame(index=df.index)
    for c in mdp_cols:
        share = np.where(gross > 0, df[c] / gross, 0.5)
        meter_net = df[c] - share * solar - share * battery_delta
        out[f"{c}_net"] = np.maximum(0.0, meter_net)
    return out


def compute_site_peak_5cp_nspl(df: pd.DataFrame, plc_hours: Sequence[pd.Timestamp],
                                nspl_hours: Sequence[pd.Timestamp]) -> tuple[float, float]:
    """Compute the site's load (kW of grid import) at PLC hours (average of 5) and NSPL hour."""
    plc_vals = [float(df.loc[pd.Timestamp(ts), "grid_import"]) for ts in plc_hours if pd.Timestamp(ts) in df.index]
    nspl_vals = [float(df.loc[pd.Timestamp(ts), "grid_import"]) for ts in nspl_hours if pd.Timestamp(ts) in df.index]
    plc = float(np.mean(plc_vals)) if plc_vals else 0.0
    nspl = float(max(nspl_vals)) if nspl_vals else 0.0
    return plc, nspl


def annual_cost_from_dispatch(
    df: pd.DataFrame,
    mdp_cols: Sequence[str],
    delivery_tariff: ComEdVLLDelivery,
    supply_tariff: IndexSupply,
    lmp: pd.Series,
    plc_hours: Sequence[pd.Timestamp],
    nspl_hours: Sequence[pd.Timestamp],
    scenario_label: str,
) -> AnnualResult:
    """Compute full-year bill under this dispatch trajectory.

    Expected columns in df: <mdp_cols>, solar_kw, p_chg, p_dis, grid_import.
    """
    meter_net = allocate_battery_to_meters(df, mdp_cols)
    # Combined grid import after battery (from dispatch) — use as "combined_kw" for delivery calc.
    # Provide per-meter allocated flows to delivery bill.
    full_df = pd.DataFrame({
        "mdp1": meter_net[f"{mdp_cols[0]}_net"].values,
        "mdp2": meter_net[f"{mdp_cols[1]}_net"].values,
        "combined_kw": df["grid_import"].values,
    }, index=df.index)

    # Monthly billing
    delivery_total = 0.0
    monthly_bills = []
    for month_start, sub in full_df.groupby(pd.Grouper(freq="MS")):
        if len(sub) == 0:
            continue
        month_idx = month_start.month
        bill = delivery_tariff.compute_bill(
            sub, meter_cols=["mdp1", "mdp2"], combined_col="combined_kw",
            period_start=sub.index.min(), period_end=sub.index.max(),
        )
        delivery_total += bill.total
        monthly_bills.append({
            "month": month_start.strftime("%Y-%m"),
            "total": bill.total,
            "billed_demand_kw": bill.lines[2].amount / delivery_tariff.dfc_rate(month_idx),
            "kwh": float(sub["combined_kw"].sum()),
            "lines": [(l.name, l.amount) for l in bill.lines],
        })

    # Supply: compute from hourly grid_import × lmp + capacity/transmission tags using
    # the observed (post-battery) PLC/NSPL kW.
    plc_kw_post, nspl_kw_post = compute_site_peak_5cp_nspl(df, plc_hours, nspl_hours)
    total_kwh = float(df["grid_import"].sum())
    energy = float((df["grid_import"] * lmp.reindex(df.index).ffill().bfill()).sum())
    losses = total_kwh * supply_tariff.losses_factor * float(lmp.mean())
    fixed_adder = total_kwh * supply_tariff.fixed_adder
    capacity = plc_kw_post * supply_tariff.capacity_rate * 365
    transmission = nspl_kw_post * supply_tariff.transmission_rate_yr
    tec = nspl_kw_post * sum(supply_tariff.tec_rates) * 12

    supply_total = energy + losses + fixed_adder + capacity + transmission + tec

    return AnnualResult(
        scenario=scenario_label,
        delivery_cost=delivery_total,
        supply_cost=supply_total,
        capacity_tag_cost=capacity,
        transmission_tag_cost=transmission + tec,
        energy_cost=energy + losses + fixed_adder,
        other_supply=0.0,
        per_stream={
            "delivery_total": delivery_total,
            "supply_energy": energy,
            "supply_losses": losses,
            "supply_fixed_adder": fixed_adder,
            "capacity_tag": capacity,
            "transmission_tag": transmission,
            "tec_pass_through": tec,
            "plc_kw_post": plc_kw_post,
            "nspl_kw_post": nspl_kw_post,
            "total_kwh": total_kwh,
        },
        monthly_bills=monthly_bills,
    )


def compare_scenarios(baseline: AnnualResult, with_battery: AnnualResult) -> dict:
    """Decompose battery value into streams."""
    delta_total = baseline.total - with_battery.total
    # Delivery stream detail
    delivery_delta = baseline.delivery_cost - with_battery.delivery_cost
    # Extract DFC savings from monthly
    dfc_savings = 0.0
    other_delivery_savings = 0.0
    for b, w in zip(baseline.monthly_bills, with_battery.monthly_bills):
        b_dfc = next(a for n, a in b["lines"] if n == "Distribution Facility Charge")
        w_dfc = next(a for n, a in w["lines"] if n == "Distribution Facility Charge")
        dfc_savings += (b_dfc - w_dfc)
    other_delivery_savings = delivery_delta - dfc_savings

    energy_savings = baseline.energy_cost - with_battery.energy_cost
    capacity_savings = baseline.capacity_tag_cost - with_battery.capacity_tag_cost
    transmission_savings = baseline.transmission_tag_cost - with_battery.transmission_tag_cost

    return {
        "baseline_total_annual": baseline.total,
        "with_battery_total_annual": with_battery.total,
        "battery_value_annual": delta_total,
        "streams": {
            "dfc_demand_reduction": dfc_savings,
            "other_delivery_riders_taxes": other_delivery_savings,
            "energy_arbitrage": energy_savings,
            "plc_capacity_tag_reduction": capacity_savings,
            "nspl_transmission_tag_reduction": transmission_savings,
        },
        "baseline_detail": baseline.as_dict(),
        "with_battery_detail": with_battery.as_dict(),
    }
