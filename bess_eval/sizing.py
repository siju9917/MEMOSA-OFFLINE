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
    marginal_annual_savings: float = 0.0    # $ added when stepping up from previous block
    marginal_added_kwh: float = 0.0         # kWh added when stepping up from previous block
    marginal_capex: float = 0.0             # $ added capex (central assumption)
    marginal_payback_years: float = 0.0     # marginal_capex / marginal_annual_savings
    marginal_payback_years_lo: float = 0.0  # @ low capex
    marginal_payback_years_hi: float = 0.0  # @ high capex
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
        r.marginal_added_kwh = added
        r.marginal_annual_savings = gain
        r.marginal_savings_per_added_kwh = gain / added if added > 0 else 0.0
        prev_kwh = r.energy_kwh
        prev_sav = r.annual_savings

    return results


def attach_payback_analysis(
    results: list[SizingResult],
    capex_per_kwh_central: float,
    capex_per_kwh_low: float,
    capex_per_kwh_high: float,
) -> None:
    """Annotate each SizingResult with marginal capex + simple payback (yr) under
    central / low / high capex assumptions. Modifies `results` in place."""
    for r in results:
        r.marginal_capex = r.marginal_added_kwh * capex_per_kwh_central
        r.marginal_payback_years = (
            r.marginal_capex / r.marginal_annual_savings
            if r.marginal_annual_savings > 0 else float("inf")
        )
        lo_capex = r.marginal_added_kwh * capex_per_kwh_low
        hi_capex = r.marginal_added_kwh * capex_per_kwh_high
        r.marginal_payback_years_lo = (
            lo_capex / r.marginal_annual_savings if r.marginal_annual_savings > 0 else float("inf")
        )
        r.marginal_payback_years_hi = (
            hi_capex / r.marginal_annual_savings if r.marginal_annual_savings > 0 else float("inf")
        )


def recommend_largest_worth_it(
    results: list[SizingResult],
    target_payback_years: float,
) -> dict:
    """Among sizes whose MARGINAL block has payback ≤ target, recommend the LARGEST.
    This answers: 'within capex reality, what's the biggest block still worth adding?'
    Uses the central capex assumption."""
    sorted_r = sorted(results, key=lambda r: r.energy_kwh)
    qualifying = [r for r in sorted_r if r.marginal_payback_years <= target_payback_years]
    if qualifying:
        winner = qualifying[-1]  # largest of the qualifying
        # Find first block whose payback EXCEEDS threshold (reason for stopping)
        first_fail = None
        for r in sorted_r:
            if r.marginal_payback_years > target_payback_years:
                first_fail = r
                break
        rationale = (
            f"Largest block whose INCREMENTAL capex (${winner.marginal_capex:,.0f}) pays back "
            f"within {target_payback_years:.0f} yr on its marginal annual savings "
            f"(${winner.marginal_annual_savings:,.0f}/yr → "
            f"{winner.marginal_payback_years:.1f} yr payback on that incremental block alone)."
        )
        if first_fail and first_fail.energy_kwh > winner.energy_kwh:
            rationale += (
                f" The next block up ({first_fail.label}) would have marginal payback of "
                f"{first_fail.marginal_payback_years:.1f} yr (incremental savings only "
                f"${first_fail.marginal_annual_savings:,.0f}/yr vs ${first_fail.marginal_capex:,.0f} extra capex) "
                f"— exceeding the {target_payback_years:.0f}-year threshold."
            )
    else:
        winner = None
        rationale = (
            f"No block meets the {target_payback_years:.0f}-year marginal-payback threshold at the "
            f"assumed capex. Either the threshold is too tight or capex assumption is too high."
        )

    return {
        "recommended_label": winner.label if winner else None,
        "recommended_savings": winner.annual_savings if winner else 0.0,
        "recommended_marginal_payback_yr": winner.marginal_payback_years if winner else None,
        "target_payback_years": target_payback_years,
        "rationale": rationale,
    }


def detect_knee(results: list[SizingResult], extreme_drop_ratio: float = 0.40) -> dict:
    """Only report a 'knee' when the marginal-return curve shows an EXTREME plateau.

    A knee = a step-change where adding the next block returns less than 40% of what
    the previous block earned per added kWh (i.e. ≥60% step-down in marginal value).
    Anything less extreme is considered gentle diminishing returns, not a true plateau.

    This matches real-world BESS sizing charts where a knee is visually unambiguous —
    the curve clearly goes flat — rather than a gradual concave decline where every
    larger block still adds meaningful value.

    Three possible outcomes:
      (A) Extreme drop found → recommend the size BEFORE the drop. The block(s)
          beyond are clearly overbuilt for the site.
      (B) No extreme drop across the tested range → recommend the LARGEST tested size.
          The curve is still meaningfully sloped at the top of the tested range.
          This is often the correct finding for a site with significant tag value
          — bigger is simply better within the tested range.
      (C) Only one size tested → trivial; return it.

    In case (B) we additionally flag that an even larger block might capture more value,
    since the plateau hasn't been observed.
    """
    if not results:
        return {"knee": None, "rationale": "no results", "knee_found": False}

    sorted_r = sorted(results, key=lambda r: r.energy_kwh)
    n = len(sorted_r)
    best_marginal = max(r.marginal_savings_per_added_kwh for r in sorted_r)

    if n == 1:
        r = sorted_r[0]
        return {
            "knee_label": r.label, "knee_power_kw": r.power_kw, "knee_energy_kwh": r.energy_kwh,
            "knee_annual_savings": r.annual_savings, "best_marginal_per_kwh": best_marginal,
            "knee_marginal_per_kwh": r.marginal_savings_per_added_kwh,
            "extreme_drop_ratio": extreme_drop_ratio, "drop_location": None,
            "knee_found": False,
            "rationale": "Only one size tested.",
        }

    # Compute inter-block ratios — ALL consecutive pairs (no first-block exclusion,
    # because an extreme-threshold knee would never trigger on a natural first-block drop).
    ratios = []
    for i in range(n - 1):
        curr = sorted_r[i].marginal_savings_per_added_kwh
        nxt = sorted_r[i + 1].marginal_savings_per_added_kwh
        if curr > 0:
            ratios.append((i, nxt / curr))

    knee_idx = n - 1  # default: no knee → largest size
    drop_location = None
    knee_found = False
    if ratios:
        # Find the SINGLE deepest drop (minimum ratio)
        min_i, min_ratio = min(ratios, key=lambda x: x[1])
        if min_ratio < extreme_drop_ratio:
            knee_idx = min_i
            knee_found = True
            drop_location = {
                "from_size": sorted_r[min_i].label,
                "to_size": sorted_r[min_i + 1].label,
                "ratio": min_ratio,
                "drop_pct": (1 - min_ratio) * 100,
                "marginal_before": sorted_r[min_i].marginal_savings_per_added_kwh,
                "marginal_after": sorted_r[min_i + 1].marginal_savings_per_added_kwh,
            }

    # Also record the deepest observed drop for reporting even when no knee found
    if ratios:
        deepest_i, deepest_ratio = min(ratios, key=lambda x: x[1])
        deepest_drop_info = {
            "from_size": sorted_r[deepest_i].label,
            "to_size": sorted_r[deepest_i + 1].label,
            "ratio": deepest_ratio,
            "drop_pct": (1 - deepest_ratio) * 100,
            "marginal_before": sorted_r[deepest_i].marginal_savings_per_added_kwh,
            "marginal_after": sorted_r[deepest_i + 1].marginal_savings_per_added_kwh,
        }
    else:
        deepest_drop_info = None

    knee = sorted_r[knee_idx]
    for r in sorted_r:
        r.included_in_knee = (r.energy_kwh <= knee.energy_kwh)

    if knee_found:
        rationale = (
            f"EXTREME plateau detected: going from {drop_location['from_size']} to "
            f"{drop_location['to_size']}, marginal $/added-kWh collapses from "
            f"${drop_location['marginal_before']:.2f} to ${drop_location['marginal_after']:.2f} "
            f"({drop_location['drop_pct']:.0f}% step-down, exceeding the {(1-extreme_drop_ratio)*100:.0f}% "
            f"extreme-drop threshold). Sizes beyond {drop_location['from_size']} are clearly overbuilt — "
            f"each added kWh earns substantially less than any prior block."
        )
    else:
        if deepest_drop_info:
            deepest_str = (
                f" Deepest step-down observed is only {deepest_drop_info['drop_pct']:.0f}% "
                f"({deepest_drop_info['from_size']} → {deepest_drop_info['to_size']}: "
                f"${deepest_drop_info['marginal_before']:.2f} → ${deepest_drop_info['marginal_after']:.2f}/added-kWh), "
                f"which does not meet the {(1-extreme_drop_ratio)*100:.0f}% extreme-drop threshold."
            )
        else:
            deepest_str = ""
        rationale = (
            f"NO EXTREME KNEE DETECTED in the tested range.{deepest_str} "
            f"Marginal returns are still meaningful at the largest tested block "
            f"({knee.label}, ${knee.marginal_savings_per_added_kwh:.2f}/added-kWh). "
            f"The recommendation is the LARGEST tested block — bigger is simply better within "
            f"the range you gave, and the true plateau may lie at a size you didn't test. "
            f"Consider evaluating larger blocks if capex permits."
        )

    return {
        "knee_label": knee.label,
        "knee_power_kw": knee.power_kw,
        "knee_energy_kwh": knee.energy_kwh,
        "knee_annual_savings": knee.annual_savings,
        "best_marginal_per_kwh": best_marginal,
        "knee_marginal_per_kwh": knee.marginal_savings_per_added_kwh,
        "extreme_drop_ratio": extreme_drop_ratio,
        "drop_location": drop_location,
        "deepest_drop": deepest_drop_info,
        "knee_found": knee_found,
        "rationale": rationale,
    }
