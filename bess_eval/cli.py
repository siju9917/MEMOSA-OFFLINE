"""End-to-end CLI: take a site config, produce a full report.

Usage:
    python -m bess_eval.cli configs/joliet.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .ingest import load_load, load_solar, validate
from .solar_synth import fit_solar_model, synthesize_year
from .pjm_data import synthesize_hourly_lmp, find_proxy_5cp_hours, find_proxy_nspl_hour
from .tariff.comed_delivery import ComEdVLLDelivery
from .tariff.supply_index import IndexSupply
from .battery import BatterySpec
from .dispatch.perfect_foresight import perfect_foresight_dispatch
from .dispatch.rulebased import rule_based_dispatch
from .dispatch.mpc import rolling_mpc_dispatch
from .attribution import annual_cost_from_dispatch, compare_scenarios
from .dr_eval import evaluate_all, recommend_stack
from .sensitivity import run_tornado, run_monte_carlo
from .report import build_report


def run(cfg_path: str):
    cfg = Config.load(cfg_path)
    root = Path(cfg_path).resolve().parent.parent

    print("=" * 70)
    print(f"  Battery Value Evaluation — {cfg.site.name}")
    print("=" * 70)

    # --- 1. Ingest data
    print("\n[1/9] Loading data ...")
    load_df = load_load(
        root / cfg.data.load_csv,
        time_col=cfg.data.load_time_col,
        kw_cols=cfg.data.load_kw_cols,
        tz=cfg.site.tz,
    )
    solar_df = load_solar(
        root / cfg.data.solar_csv,
        time_col=cfg.data.solar_time_col,
        prod_cols=cfg.data.solar_production_cols,
        tz=cfg.site.tz,
    )
    summary = validate(load_df, solar_df)
    summary["evaluation_year"] = cfg.data.evaluation_year
    print(f"  Load: {summary['load_start']} → {summary['load_end']} ({summary['load_rows']} hrs)")
    print(f"  Annual kWh: {summary['annual_kwh']:,.0f} | peak: {summary['peak_kw']:,.1f} kW")
    print(f"  Solar measured: {summary['solar_measured_kwh']:,.0f} kWh ({summary['solar_start']} → {summary['solar_end']})")

    # --- 2. Synthesize 2025 solar
    print("\n[2/9] Synthesizing 2025 solar ...")
    fit = fit_solar_model(solar_df["combined_kw"], dict(cfg.site), dict(cfg.solar_synthesis))
    print(f"  Solar model fit: k={fit['k']:.3f}  RMSE={fit['rmse']:.1f} kW  "
          f"measured Jan-Apr={fit['measured_sum_kwh']:,.0f} kWh  predicted={fit['predicted_sum_kwh']:,.0f} kWh")
    solar_2025 = synthesize_year(cfg.data.evaluation_year, cfg.site.tz, dict(cfg.site), dict(cfg.solar_synthesis), fit["k"])
    solar_annual = float(solar_2025.sum())
    print(f"  Synthesized 2025 solar: {solar_annual:,.0f} kWh ({solar_annual/(4000*8760)*100:.1f}% CF)")

    # --- 3. Synthesize LMPs
    print("\n[3/9] Synthesizing hourly LMPs ...")
    lmp = synthesize_hourly_lmp(
        cfg.data.evaluation_year, cfg.site.tz,
        anchor_per_kwh=cfg.supply_tariff.index.index_anchor_per_kwh,
    )
    print(f"  LMP annual avg: ${lmp.mean():.4f}/kWh | peak hr: ${lmp.max():.4f}/kWh | off-peak min: ${lmp.min():.4f}/kWh")

    # --- 4. Identify 5CP & NSPL hours
    print("\n[4/9] Identifying 5CP + NSPL proxy hours ...")
    plc_hours = find_proxy_5cp_hours(
        load_df, cfg.data.evaluation_year,
        anchored_hours=[pd.Timestamp("2025-08-15 18:00")],
    )
    nspl_hour = find_proxy_nspl_hour(load_df, cfg.data.evaluation_year,
                                      anchored_hour=pd.Timestamp("2025-08-15 18:00", tz=cfg.site.tz))
    print("  PJM 5CP proxy hours:")
    for h in plc_hours:
        print(f"    - {h}  (load {load_df.loc[h, 'combined_kw']:,.0f} kW)")
    print(f"  PJM NSPL (1CP) proxy: {nspl_hour}")

    # --- 5. Assemble full-year dataframe
    print("\n[5/9] Assembling full-year simulation frame ...")
    # Limit to evaluation year hours only
    idx = pd.date_range(f"{cfg.data.evaluation_year}-01-01", f"{cfg.data.evaluation_year+1}-01-01",
                         freq="1h", tz=cfg.site.tz, inclusive="left")
    year_df = pd.DataFrame(index=idx)
    ld = load_df.reindex(idx).ffill().bfill()
    year_df["mdp1"] = ld["mdp1"].values
    year_df["mdp2"] = ld["mdp2"].values
    year_df["load_kw"] = ld["combined_kw"].values
    year_df["solar_kw"] = solar_2025.reindex(idx).fillna(0).values
    year_df["lmp"] = lmp.reindex(idx).ffill().bfill().values
    print(f"  Year frame: {len(year_df)} hours")

    # --- 6. Set up tariff engines + battery
    delivery = ComEdVLLDelivery(dict(cfg.delivery_tariff))
    supply = IndexSupply(dict(cfg.supply_tariff.index))
    battery = BatterySpec.from_cfg(dict(cfg.battery))
    tag_costs = supply.annual_tag_cost(cfg.supply_tariff.index.capacity_obligation_kw,
                                        cfg.supply_tariff.index.nspl_kw)
    print(f"  Battery: {battery.power_kw} kW / {battery.energy_kwh} kWh, RTE {battery.rte_ac_ac}")
    print(f"  Annual tag cost baseline: capacity ${tag_costs['capacity_annual']:,.0f}  "
          f"transmission ${tag_costs['transmission_annual']:,.0f}")

    # --- 7. Run scenarios
    # Baseline = no battery (p_chg=p_dis=0)
    print("\n[6/9] Computing baseline (no-battery) annual cost ...")
    baseline_df = year_df.copy()
    baseline_df["p_chg"] = 0.0
    baseline_df["p_dis"] = 0.0
    baseline_df["soc_kwh"] = battery.soc_init_kwh
    baseline_df["grid_import"] = np.maximum(0.0, baseline_df["load_kw"] - baseline_df["solar_kw"])
    baseline_df["grid_export"] = np.maximum(0.0, baseline_df["solar_kw"] - baseline_df["load_kw"])
    baseline_result = annual_cost_from_dispatch(
        baseline_df, ["mdp1", "mdp2"], delivery, supply, lmp, plc_hours, [nspl_hour], "Baseline (no battery)"
    )
    print(f"  Baseline annual: delivery ${baseline_result.delivery_cost:,.0f}  "
          f"supply ${baseline_result.supply_cost:,.0f}  TOTAL ${baseline_result.total:,.0f}")

    # Perfect foresight
    print("\n[7/9] Running perfect-foresight dispatch ...")
    pf_res = perfect_foresight_dispatch(
        year_df, battery, dict(cfg.delivery_tariff.dfc_per_kw_monthly),
        plc_hours, [nspl_hour], dict(cfg.mpc),
        export_allowed=cfg.export.allowed,
        export_rate_per_kwh=cfg.export.rate_per_kwh_if_allowed,
    )
    pf_df = pf_res.df.join(year_df[["mdp1", "mdp2"]])
    pf_result = annual_cost_from_dispatch(
        pf_df, ["mdp1", "mdp2"], delivery, supply, lmp, plc_hours, [nspl_hour], "Perfect foresight"
    )
    comparison_pf = compare_scenarios(baseline_result, pf_result)
    print(f"  PF annual: ${pf_result.total:,.0f}  savings=${comparison_pf['battery_value_annual']:,.0f}")

    # MPC (noisy foresight)
    print("\n[8/9] Running MPC-realistic dispatch (noisy foresight) ...")
    mpc_df_raw = rolling_mpc_dispatch(
        year_df, battery, dict(cfg.delivery_tariff.dfc_per_kw_monthly),
        plc_hours, [nspl_hour], dict(cfg.mpc),
        export_allowed=cfg.export.allowed,
        export_rate_per_kwh=cfg.export.rate_per_kwh_if_allowed,
    )
    mpc_df = mpc_df_raw.join(year_df[["mdp1", "mdp2"]])
    mpc_result = annual_cost_from_dispatch(
        mpc_df, ["mdp1", "mdp2"], delivery, supply, lmp, plc_hours, [nspl_hour], "MPC-realistic"
    )
    comparison_mpc = compare_scenarios(baseline_result, mpc_result)
    print(f"  MPC annual: ${mpc_result.total:,.0f}  savings=${comparison_mpc['battery_value_annual']:,.0f}")

    # Rule-based
    print("\n  ... rule-based dispatch ...")
    rb_df_raw = rule_based_dispatch(year_df, battery)
    rb_df = rb_df_raw.join(year_df[["mdp1", "mdp2"]])
    rb_result = annual_cost_from_dispatch(
        rb_df, ["mdp1", "mdp2"], delivery, supply, lmp, plc_hours, [nspl_hour], "Rule-based"
    )
    comparison_rb = compare_scenarios(baseline_result, rb_result)
    print(f"  RB annual: ${rb_result.total:,.0f}  savings=${comparison_rb['battery_value_annual']:,.0f}")

    # --- 8. DR opportunity
    print("\n[9/11] Evaluating DR program opportunity value ...")
    dr_values = evaluate_all(cfg.dr_programs_evaluated, dfc_value_per_kw=160.0, plc_value_per_kw=98.70)
    for d in dr_values:
        print(f"  {d.name} ({d.kind}): gross ${d.gross:,.0f}  opp-cost ${d.opportunity_cost_vs_baseline:,.0f}  "
              f"NET ${d.net_annual_value:,.0f}")
    stack = recommend_stack(dr_values)
    print(f"  Recommended stack: {stack['recommended_stack']}")
    print(f"  Stacked annual value: ${stack['stacked_annual_value']:,.0f}")

    # --- 9. Tornado sensitivity
    print("\n[10/11] Tornado sensitivity (8 inputs, ~16 LP solves) ...")
    baseline_value_for_sens = comparison_mpc["battery_value_annual"]
    tornado = run_tornado(
        year_df, lmp, battery, delivery, supply,
        dict(cfg.delivery_tariff.dfc_per_kw_monthly),
        plc_hours, nspl_hour, dict(cfg.mpc),
        baseline_value=baseline_value_for_sens,
        export_allowed=cfg.export.allowed,
        export_rate=cfg.export.rate_per_kwh_if_allowed,
    )
    print("  Ranked by swing (|hi-lo|):")
    for r in tornado:
        print(f"    {r['label']:<45} lo=${r['lo']:>10,.0f}  hi=${r['hi']:>10,.0f}  span=${r['span']:>10,.0f}")

    # --- 10. Monte Carlo
    print("\n[11/11] Monte Carlo confidence band (20 samples) ...")
    mc = run_monte_carlo(
        year_df, lmp, battery, delivery, supply,
        dict(cfg.delivery_tariff.dfc_per_kw_monthly),
        plc_hours, nspl_hour, dict(cfg.mpc),
        baseline_value=baseline_value_for_sens,
        n_samples=20,
        export_allowed=cfg.export.allowed,
        export_rate=cfg.export.rate_per_kwh_if_allowed,
    )
    print(f"  P10: ${mc['p10']:,.0f}   P50: ${mc['p50']:,.0f}   P90: ${mc['p90']:,.0f}")
    print(f"  Mean: ${mc['mean']:,.0f}  StDev: ${mc['std']:,.0f}")

    # --- 11. Report
    out_dir = Path("results") / cfg.site.name
    print(f"\nBuilding report in {out_dir}/ ...")
    html = build_report(
        out_dir=root / "results" / cfg.site.name,
        site_cfg=dict(cfg.site),
        data_summary=summary,
        solar_annual_kwh=solar_annual,
        battery_cfg=dict(cfg.battery),
        comparison_mpc=comparison_mpc,
        comparison_pf=comparison_pf,
        comparison_rb=comparison_rb,
        dr_values=dr_values,
        baseline_result=baseline_result,
        mpc_result=mpc_result,
        pf_result=pf_result,
        rb_result=rb_result,
        mpc_df=mpc_df,
        pf_df=pf_df,
        rb_df=rb_df,
        plc_hours=plc_hours,
        nspl_hour=nspl_hour,
        tag_costs=tag_costs,
        tornado=tornado,
        mc=mc,
        dr_stack=stack,
    )
    print(f"Report: {html}")
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m bess_eval.cli <config.yaml>")
        sys.exit(1)
    run(sys.argv[1])
