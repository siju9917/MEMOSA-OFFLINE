"""Sensitivity analysis + Monte Carlo confidence band.

Two complementary views on uncertainty:

1. **Tornado analysis** — vary each major assumption individually by ±one standard
   unit, hold all others at baseline. Shows which assumption moves the answer most.
   Fast — only O(2N) recomputations of attribution.

2. **Monte Carlo** — sample all major uncertain inputs jointly from their distributions,
   re-run the full dispatch + attribution N times, and report the P10/P50/P90 of the
   annual battery value. Expensive (each MC draw = one MPC LP solve ≈ 15s) so we cap N.

Key uncertain inputs treated:
  - Solar annual production (fit to 112 days; ±15% band for unseen months)
  - LMP level (anchor is one billing period; ±20% multiplier on mean)
  - LMP shape intensity (peak/offpeak ratio; ±25%)
  - PLC hour identification: we know 1/5 exactly; each of the other 4 has some miss
    probability. Model as: per hour, 80% probability of being correctly predicted,
    20% miss (in which case battery provides no reduction for that hour).
  - NSPL hour: known with high confidence (anchor).
  - Demand interval understatement (hourly vs 30-min billing): +3% demand potential.
  - MPC forecast skill: load MAPE 3-6%, solar 10-20%, LMP 7-12%.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd

from .battery import BatterySpec
from .tariff.comed_delivery import ComEdVLLDelivery
from .tariff.supply_index import IndexSupply
from .dispatch.perfect_foresight import perfect_foresight_dispatch
from .dispatch.mpc import noisy_foresight_dispatch
from .attribution import annual_cost_from_dispatch, compare_scenarios


@dataclass
class SensitivityResult:
    tornado: list = field(default_factory=list)  # list of (label, lo, hi, delta)
    mc_samples: list = field(default_factory=list)  # list of scalar annual values
    p10: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    baseline_value: float = 0.0


def _compute_value(
    year_df: pd.DataFrame,
    lmp: pd.Series,
    battery: BatterySpec,
    delivery: ComEdVLLDelivery,
    supply,
    dfc_monthly: dict,
    plc_hours: list,
    nspl_hour: pd.Timestamp,
    mpc_cfg: dict,
    export_allowed: bool,
    export_rate: float,
    load_mape: float = 0.04,  # kept for signature compatibility; unused in PF path
    solar_mape: float = 0.15,
    lmp_mape: float = 0.10,
    seed: int = 42,            # unused in PF path
    mpc_gap_factor: float = 0.83,
    demand_on_peak_only: bool = True,
) -> float:
    """End-to-end: baseline vs PERFECT-FORESIGHT × gap_factor → $ annual savings.

    Why PF (not noisy-MPC) for sensitivity/MC analyses:
      - PF is deterministic → same inputs give same output (needed for honest tornado)
      - PF is monotonic in battery size (needed for sizing)
      - For flat-load sites (e.g. cold storage), noisy-foresight has pathological
        seed-variance that would DOMINATE input sensitivity. PF scaling cleanly
        separates 'how do fundamental inputs affect value' (measured here) from
        'how much does realistic dispatch lose vs optimal' (the gap_factor).

    The gap_factor is computed externally once (from multi-seed MEMOSA mean at the
    primary size / PF at the primary size) and applied uniformly here. This is a
    conservative and stable estimation strategy.
    """
    from .dispatch.perfect_foresight import perfect_foresight_dispatch

    # Baseline (deterministic)
    bl = year_df.copy()
    bl["p_chg"] = 0.0
    bl["p_dis"] = 0.0
    bl["soc_kwh"] = battery.soc_init_kwh
    bl["grid_import"] = np.maximum(0.0, bl["load_kw"] - bl["solar_kw"])
    bl["grid_export"] = 0.0
    meter_cols = [c for c in year_df.columns if c.startswith("mdp")]
    bl_res = annual_cost_from_dispatch(
        bl, meter_cols, delivery, supply, lmp,
        plc_hours, [nspl_hour], "baseline"
    )

    pf_out = perfect_foresight_dispatch(
        year_df, battery, dfc_monthly, plc_hours, [nspl_hour], mpc_cfg,
        export_allowed=export_allowed, export_rate_per_kwh=export_rate,
        demand_on_peak_only=demand_on_peak_only,
    )
    pf_df = pf_out.df.join(year_df[meter_cols])
    pf_res = annual_cost_from_dispatch(
        pf_df, meter_cols, delivery, supply, lmp,
        plc_hours, [nspl_hour], "pf"
    )
    comp = compare_scenarios(bl_res, pf_res)
    return float(comp["battery_value_annual"]) * mpc_gap_factor


def run_tornado(
    year_df: pd.DataFrame,
    lmp: pd.Series,
    battery: BatterySpec,
    delivery: ComEdVLLDelivery,
    supply,
    dfc_monthly: dict,
    plc_hours: list,
    nspl_hour: pd.Timestamp,
    mpc_cfg: dict,
    baseline_value: float,
    export_allowed: bool = False,
    export_rate: float = 0.0,
    plc_reduction_value: float = 73875.0,
    mpc_gap_factor: float = 0.83,
    demand_on_peak_only: bool = True,
) -> list[dict]:
    """Perform tornado sensitivity: vary each assumption ±1σ, compute battery value delta."""
    def run(ylbl: str, **overrides):
        df = overrides.get("year_df", year_df)
        lmp_s = overrides.get("lmp", lmp)
        dfc_m = overrides.get("dfc_monthly", dfc_monthly)
        plc = overrides.get("plc_hours", plc_hours)
        nspl = overrides.get("nspl_hour", nspl_hour)
        return _compute_value(
            df, lmp_s, battery, delivery, supply, dfc_m,
            plc, nspl, mpc_cfg, export_allowed, export_rate,
            mpc_gap_factor=mpc_gap_factor,
            demand_on_peak_only=demand_on_peak_only,
        )

    rows = []

    # 1. Solar annual (±15%)
    s_lo = year_df.copy(); s_lo["solar_kw"] = year_df["solar_kw"] * 0.85
    s_hi = year_df.copy(); s_hi["solar_kw"] = year_df["solar_kw"] * 1.15
    lo = run("solar -15%", year_df=s_lo)
    hi = run("solar +15%", year_df=s_hi)
    rows.append({"label": "Solar annual total (±15%)", "lo": lo, "hi": hi})

    # 2. LMP level (±20%) — scale both the year_df column and the separate lmp series
    lo_df = year_df.copy(); lo_df["lmp"] = year_df["lmp"] * 0.80
    hi_df = year_df.copy(); hi_df["lmp"] = year_df["lmp"] * 1.20
    lo = run("", year_df=lo_df, lmp=lmp * 0.80)
    hi = run("", year_df=hi_df, lmp=lmp * 1.20)
    rows.append({"label": "LMP level (±20%)", "lo": lo, "hi": hi})

    # 3. LMP shape (peak-offpeak intensity ±25%)
    mean_lmp = float(lmp.mean())
    # Compress / stretch the deviations from mean
    dev = lmp - mean_lmp
    lo_lmp = mean_lmp + dev * 0.75
    hi_lmp = mean_lmp + dev * 1.25
    lo_df = year_df.copy(); lo_df["lmp"] = lo_lmp.values
    hi_df = year_df.copy(); hi_df["lmp"] = hi_lmp.values
    lo = run("", year_df=lo_df, lmp=lo_lmp)
    hi = run("", year_df=hi_df, lmp=hi_lmp)
    rows.append({"label": "LMP diurnal intensity (±25%)", "lo": lo, "hi": hi})

    # 4. PLC hour identification — honest "miss" model.
    # If 1 of our 5 proxy hours isn't a true PJM 5CP (and the true one is elsewhere),
    # the battery doesn't shave that true hour. Effect: PLC reduction drops by 1/5
    # because one of the 5 hours contributing to PLC average stays near baseline.
    # Battery could still incidentally shave it if it's a DFC target hour, so call it
    # 80% effectiveness on the missed hour. Net: 1/5 × 0.8 ≈ 16% reduction in PLC savings.
    # Analytic adjustment (not a rerun):
    # baseline_plc_reduction_value ≈ (807 - 59) × $98.70 = $73,875 (from main run)
    # miss-1 reduces this by ≈ 1/5 × (1 - 0.2) = 16% → $11,820 loss.
    # best-case = baseline_value (cannot know MORE than 5 true peaks)
    plc_loss_from_miss1 = plc_reduction_value * (1.0/5.0) * (1.0 - 0.2)
    lo = baseline_value - plc_loss_from_miss1
    hi = baseline_value
    rows.append({"label": "PLC identification (miss 1 of 5 proxy)", "lo": lo, "hi": hi})

    # 5. Demand interval: hourly data misses sub-hour spikes. Ratebook bills on 30-min.
    # A 30-min billing interval would understate our baseline demand by ~3%; with-battery
    # demand by about the same; net effect on SAVINGS is similar.  To model: scale load
    # up 3% to simulate the 30-min peaks. Battery can only shave ~2/3 of that (the 3%
    # higher peak is in minutes we can't predict). Net: small upside to savings.
    scaled_df = year_df.copy()
    scaled_df["load_kw"] = year_df["load_kw"] * 1.03
    for _c in [c for c in year_df.columns if c.startswith("mdp")]:
        scaled_df[_c] = year_df[_c] * 1.03
    hi = run("", year_df=scaled_df)
    lo = baseline_value  # baseline is the hourly-data case
    rows.append({"label": "15-min billing interval (hourly understates)", "lo": lo, "hi": hi})

    # 6. Controller quality (MEMOSA-vs-PF gap) — scale PF value by ±5pp gap band
    lo = baseline_value * (mpc_gap_factor - 0.05) / mpc_gap_factor if mpc_gap_factor > 0 else baseline_value
    hi = baseline_value * (mpc_gap_factor + 0.05) / mpc_gap_factor if mpc_gap_factor > 0 else baseline_value
    rows.append({"label": "Controller quality (MEMOSA vs PF gap ±5pp)", "lo": lo, "hi": hi})

    # 7. Battery RTE (±2pp) — changes physical dispatch, need real LP re-run
    def _bat_at_rte(rte):
        return BatterySpec(
            power_kw=battery.power_kw, energy_kwh=battery.energy_kwh,
            rte_ac_ac=rte,
            soc_min_frac=battery.soc_min_frac, soc_max_frac=battery.soc_max_frac,
            soc_init_frac=battery.soc_init_frac,
            aux_load_frac_per_day=battery.aux_load_frac_per_day,
            coupling=battery.coupling,
        )
    lo_val = _compute_value(year_df, lmp, _bat_at_rte(battery.rte_ac_ac - 0.02),
                             delivery, supply, dfc_monthly,
                             plc_hours, nspl_hour, mpc_cfg, export_allowed, export_rate,
                             mpc_gap_factor=mpc_gap_factor,
                             demand_on_peak_only=demand_on_peak_only)
    hi_val = _compute_value(year_df, lmp, _bat_at_rte(battery.rte_ac_ac + 0.02),
                             delivery, supply, dfc_monthly,
                             plc_hours, nspl_hour, mpc_cfg, export_allowed, export_rate,
                             mpc_gap_factor=mpc_gap_factor,
                             demand_on_peak_only=demand_on_peak_only)
    rows.append({"label": "Battery RTE (±2 pp)", "lo": lo_val, "hi": hi_val})

    for r in rows:
        r["delta_lo"] = r["lo"] - baseline_value
        r["delta_hi"] = r["hi"] - baseline_value
        r["span"] = abs(r["delta_hi"] - r["delta_lo"])
    rows.sort(key=lambda r: -r["span"])
    return rows


def run_monte_carlo(
    year_df: pd.DataFrame,
    lmp: pd.Series,
    battery: BatterySpec,
    delivery: ComEdVLLDelivery,
    supply,
    dfc_monthly: dict,
    plc_hours: list,
    nspl_hour: pd.Timestamp,
    mpc_cfg: dict,
    baseline_value: float,
    n_samples: int = 20,
    export_allowed: bool = False,
    export_rate: float = 0.0,
    rng_seed: int = 2025,
    plc_reduction_value: float = 73875.0,
    num_known_anchor_hours: int = 1,
    total_plc_hours: int = 5,
    mpc_gap_factor: float = 0.83,
    demand_on_peak_only: bool = True,
) -> dict:
    """Monte Carlo sample joint uncertainty → P10/P50/P90 band on annual battery value."""
    rng = np.random.default_rng(rng_seed)
    vals = []
    for i in range(n_samples):
        # Draw multiplicative factors
        solar_mult = float(np.clip(rng.normal(1.0, 0.10), 0.7, 1.3))
        lmp_level_mult = float(np.clip(rng.normal(1.0, 0.12), 0.6, 1.4))
        lmp_intensity = float(np.clip(rng.normal(1.0, 0.15), 0.7, 1.4))
        # PLC miss count — Binomial(non_anchor_count, p=0.20)
        non_anchor = max(0, total_plc_hours - num_known_anchor_hours)
        misses = int(rng.binomial(non_anchor, 0.20)) if non_anchor > 0 else 0
        # Also MEMOSA controller quality: sample gap_factor ± ~5pp band
        gap_sample = float(np.clip(rng.normal(mpc_gap_factor, 0.05), 0.50, 1.00))

        df_i = year_df.copy()
        df_i["solar_kw"] = year_df["solar_kw"] * solar_mult
        mean_lmp = float(lmp.mean())
        new_lmp = (mean_lmp + (lmp - mean_lmp) * lmp_intensity) * lmp_level_mult
        df_i["lmp"] = new_lmp.values
        val = _compute_value(
            df_i, new_lmp, battery, delivery, supply, dfc_monthly,
            plc_hours, nspl_hour, mpc_cfg, export_allowed, export_rate,
            mpc_gap_factor=gap_sample,
            demand_on_peak_only=demand_on_peak_only,
        )
        # PLC-miss haircut: each missed hour loses 1/total × 80% of PLC reduction value
        plc_haircut = misses * (1.0 / total_plc_hours) * 0.8 * plc_reduction_value
        val -= plc_haircut
        vals.append(val)
    arr = np.array(vals)
    return {
        "samples": vals,
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
