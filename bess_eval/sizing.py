"""Battery-size sweep + knee detection.

Given a list of (power_kW, energy_kWh) block sizes, run MEMOSA-controls dispatch
for each, record annual $-saved, and identify the "knee" — the largest size where
adding the NEXT block is still yielding at least 50% of the best-block marginal
return per kWh. Beyond that size, returns plateau.

Also returns marginal $/kWh added per consecutive block so the user can eyeball
diminishing returns.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .battery import BatterySpec
from .tariff.comed_delivery import ComEdVLLDelivery
from .tariff.supply_index import IndexSupply
from .sensitivity import _compute_value


@dataclass
class SizingResult:
    label: str
    power_kw: float
    energy_kwh: float
    annual_savings: float
    marginal_savings_per_added_kwh: float   # vs the previous size; first size is vs 0
    included_in_knee: bool = False


def _make_battery(cfg_template: dict, power_kw: float, energy_kwh: float) -> BatterySpec:
    """Clone the battery config with size overridden."""
    return BatterySpec(
        power_kw=power_kw,
        energy_kwh=energy_kwh,
        rte_ac_ac=cfg_template["rte_ac_ac"],
        soc_min_frac=cfg_template["soc_min_frac"],
        soc_max_frac=cfg_template["soc_max_frac"],
        soc_init_frac=cfg_template["soc_init_frac"],
        aux_load_frac_per_day=cfg_template["aux_load_frac_per_day"],
        coupling=cfg_template.get("coupling", "AC"),
    )


def run_sweep(
    sizes: Sequence[tuple[float, float, str]],   # (power_kw, energy_kwh, label)
    year_df: pd.DataFrame,
    lmp: pd.Series,
    battery_cfg_template: dict,
    delivery: ComEdVLLDelivery,
    supply: IndexSupply,
    dfc_monthly: dict,
    plc_hours: list,
    nspl_hour: pd.Timestamp,
    mpc_cfg: dict,
    export_allowed: bool = False,
    export_rate: float = 0.0,
) -> list[SizingResult]:
    """Run MEMOSA-controls dispatch once per size and return annual savings."""
    sorted_sizes = sorted(sizes, key=lambda s: s[1])  # by energy
    results: list[SizingResult] = []
    for power_kw, energy_kwh, label in sorted_sizes:
        battery = _make_battery(battery_cfg_template, power_kw, energy_kwh)
        savings = _compute_value(
            year_df, lmp, battery, delivery, supply, dfc_monthly,
            plc_hours, nspl_hour, mpc_cfg,
            export_allowed=export_allowed, export_rate=export_rate,
            load_mape=0.04, solar_mape=0.15, lmp_mape=0.10, seed=42,
        )
        results.append(SizingResult(
            label=label, power_kw=power_kw, energy_kwh=energy_kwh,
            annual_savings=savings, marginal_savings_per_added_kwh=0.0,
        ))

    # Compute marginal returns
    prev_kwh = 0.0
    prev_sav = 0.0
    for r in results:
        added = r.energy_kwh - prev_kwh
        gain = r.annual_savings - prev_sav
        r.marginal_savings_per_added_kwh = gain / added if added > 0 else 0.0
        prev_kwh = r.energy_kwh
        prev_sav = r.annual_savings

    return results


def detect_knee(results: list[SizingResult], threshold_ratio: float = 0.5) -> dict:
    """Largest size where marginal_per_kWh >= threshold × best marginal_per_kWh.

    Rationale: the first block is the most productive $/kWh. As sizes grow each
    new block contributes less. The knee is the last block still earning at
    least `threshold_ratio` of the best block's rate. Beyond the knee you're
    paying for kWh that return substantially less than what the smaller blocks did.
    """
    if not results:
        return {"knee": None, "rationale": "no results"}

    best_marginal = max(r.marginal_savings_per_added_kwh for r in results)
    knee = None
    for r in results:
        if r.marginal_savings_per_added_kwh >= threshold_ratio * best_marginal:
            knee = r
            r.included_in_knee = True
    if knee is None:
        knee = results[0]

    return {
        "knee_label": knee.label,
        "knee_power_kw": knee.power_kw,
        "knee_energy_kwh": knee.energy_kwh,
        "knee_annual_savings": knee.annual_savings,
        "best_marginal_per_kwh": best_marginal,
        "knee_marginal_per_kwh": knee.marginal_savings_per_added_kwh,
        "threshold_ratio": threshold_ratio,
        "rationale": (
            f"Largest size whose added-kWh returned at least {threshold_ratio*100:.0f}% "
            f"of the best block's ${best_marginal:.2f}/added-kWh rate. Beyond this size, "
            f"each additional kWh earns less than ${threshold_ratio*best_marginal:.2f}/yr, "
            f"indicating diminishing returns."
        ),
    }
