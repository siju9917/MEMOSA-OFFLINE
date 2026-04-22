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
from .ingest import load_load, load_solar, validate, reconstruct_gross_load
from .solar_synth import fit_solar_model, synthesize_year
from .pjm_data import synthesize_hourly_lmp, find_proxy_5cp_hours, find_proxy_nspl_hour
from .tariff import build_delivery, build_supply
from .battery import BatterySpec
from .dispatch.perfect_foresight import perfect_foresight_dispatch
from .dispatch.rulebased import rule_based_dispatch
from .dispatch.mpc import rolling_mpc_dispatch
from .attribution import annual_cost_from_dispatch, compare_scenarios
from .dr_eval import evaluate_all, recommend_stack
from .sensitivity import run_tornado, run_monte_carlo
from .sizing import run_sweep, detect_knee, attach_payback_analysis, recommend_largest_worth_it
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
    solar_enabled = cfg.data.get("solar_enabled", True) and cfg.data.get("solar_csv") is not None
    if solar_enabled:
        solar_df = load_solar(
            root / cfg.data.solar_csv,
            time_col=cfg.data.solar_time_col,
            prod_cols=cfg.data.solar_production_cols,
            tz=cfg.site.tz,
        )
    else:
        solar_df = None
    summary = validate(load_df, solar_df)
    summary["evaluation_year"] = cfg.data.evaluation_year
    print(f"  Load: {summary['load_start']} → {summary['load_end']} ({summary['load_rows']} hrs)")
    print(f"  Annual kWh (as-read): {summary['annual_kwh']:,.0f} | peak: {summary['peak_kw']:,.1f} kW")
    if solar_df is not None:
        print(f"  Solar measured: {summary['solar_measured_kwh']:,.0f} kWh ({summary['solar_start']} → {summary['solar_end']})")
    else:
        print(f"  Solar: DISABLED for this scenario (no solar data or solar_enabled=false)")

    # If solar is embedded in the load data (site already has solar during eval year),
    # reconstruct gross pre-solar load so we can apply the synthesized solar cleanly in
    # both baseline and with-battery scenarios.
    if cfg.data.get("solar_embedded_in_load", False):
        print("  Solar embedded in load → reconstructing gross pre-solar load ...")
        meter_cols = [f"mdp{i+1}" for i in range(len(cfg.data.load_kw_cols))]
        load_df = reconstruct_gross_load(load_df, solar_df, meter_cols)
        new_annual = float(load_df["combined_kw"].sum())
        new_peak = float(load_df["combined_kw"].max())
        summary["annual_kwh_gross"] = new_annual
        summary["peak_kw_gross"] = new_peak
        print(f"  Gross reconstructed annual kWh: {new_annual:,.0f} | gross peak: {new_peak:,.1f} kW")

    # --- 2. Synthesize 2025 solar
    print("\n[2/9] Synthesizing 2025 solar ...")
    if solar_df is not None:
        fit = fit_solar_model(solar_df["combined_kw"], dict(cfg.site), dict(cfg.solar_synthesis))
        print(f"  Solar model fit: k={fit['k']:.3f}  RMSE={fit['rmse']:.1f} kW  "
              f"measured Jan-Apr={fit['measured_sum_kwh']:,.0f} kWh  predicted={fit['predicted_sum_kwh']:,.0f} kWh")
        solar_2025 = synthesize_year(cfg.data.evaluation_year, cfg.site.tz, dict(cfg.site), dict(cfg.solar_synthesis), fit["k"])
    elif cfg.solar_synthesis.get("hypothetical_from_nameplate", False):
        # No measured solar — build a hypothetical profile from clear-sky × monthly CSF
        # scaled so the annual total equals nameplate_kw × expected_annual_cf × 8760.
        print(f"  No measured solar — synthesizing hypothetical profile from nameplate kW")
        nameplate_kw = cfg.solar_synthesis.ac_inverter_limit_kw
        expected_cf = cfg.solar_synthesis.get("expected_annual_cf", 0.16)   # PA is ~15-17% CF
        target_annual_kwh = nameplate_kw * expected_cf * 8760
        # Use k=1.0, synthesize, then rescale to hit the target
        probe = synthesize_year(cfg.data.evaluation_year, cfg.site.tz, dict(cfg.site),
                                 dict(cfg.solar_synthesis), k=1.0)
        probe_sum = float(probe.sum())
        k_adjusted = target_annual_kwh / probe_sum if probe_sum > 0 else 0.0
        solar_2025 = synthesize_year(cfg.data.evaluation_year, cfg.site.tz, dict(cfg.site),
                                      dict(cfg.solar_synthesis), k=k_adjusted)
        print(f"  Hypothetical system: {nameplate_kw:,.0f} kW AC, expected CF {expected_cf*100:.1f}% → {target_annual_kwh:,.0f} kWh/yr")
    else:
        # No solar at all — return zero-solar series of correct length for this year
        idx = pd.date_range(f"{cfg.data.evaluation_year}-01-01",
                             f"{cfg.data.evaluation_year+1}-01-01",
                             freq="1h", tz=cfg.site.tz, inclusive="left")
        solar_2025 = pd.Series(0.0, index=idx, name="solar_kw")
        print(f"  No solar in this scenario (all zeros)")
    solar_annual = float(solar_2025.sum())
    inv_cap = cfg.solar_synthesis.get("ac_inverter_limit_kw", 4000)
    print(f"  2025 solar total: {solar_annual:,.0f} kWh ({solar_annual/(max(inv_cap,1)*8760)*100:.1f}% CF on {inv_cap:,.0f} kW AC)")

    # --- 3. Synthesize LMPs
    print("\n[3/9] Synthesizing hourly LMPs ...")
    # LMP anchor: prefer explicit supply_tariff.lmp_anchor_per_kwh, else fall back to
    # index structure's anchor (index-supply sites), else use the Freepoint hub rate.
    supply_cfg = dict(cfg.supply_tariff)
    if "lmp_anchor_per_kwh" in supply_cfg:
        lmp_anchor = supply_cfg["lmp_anchor_per_kwh"]
    elif supply_cfg.get("primary") == "index":
        lmp_anchor = supply_cfg["index"]["index_anchor_per_kwh"]
    elif supply_cfg.get("primary") == "freepoint":
        lmp_anchor = supply_cfg["freepoint"]["hub_energy_rate_per_kwh"]
    else:
        raise ValueError("Unable to determine LMP anchor for synthesis")
    lmp = synthesize_hourly_lmp(cfg.data.evaluation_year, cfg.site.tz, anchor_per_kwh=lmp_anchor)
    print(f"  LMP annual avg: ${lmp.mean():.4f}/kWh | peak hr: ${lmp.max():.4f}/kWh | off-peak min: ${lmp.min():.4f}/kWh")

    # --- 4. Identify 5CP & NSPL hours
    print("\n[4/9] Identifying 5CP + NSPL proxy hours ...")
    # PLC/NSPL anchor hours are optional per site. Joliet has a bill-confirmed 8/15 18:00
    # anchor; Bedford does not — all 5 hours are proxied.
    anchor_list_raw = cfg.get("plc_anchor_hours", [])
    plc_anchors_naive = [pd.Timestamp(s) for s in anchor_list_raw]
    nspl_anchor_raw = cfg.get("nspl_anchor_hour", None)
    if nspl_anchor_raw:
        nspl_anchor = pd.Timestamp(nspl_anchor_raw).tz_localize(cfg.site.tz)
    else:
        nspl_anchor = None
    plc_hours = find_proxy_5cp_hours(
        load_df, cfg.data.evaluation_year, anchored_hours=plc_anchors_naive,
    )
    nspl_hour = find_proxy_nspl_hour(
        load_df, cfg.data.evaluation_year, anchored_hour=nspl_anchor,
    )
    print("  PJM 5CP proxy hours:")
    for h in plc_hours:
        print(f"    - {h}  (load {load_df.loc[h, 'combined_kw']:,.0f} kW)")
    print(f"  PJM NSPL (1CP) proxy: {nspl_hour}")

    # --- 5. Assemble full-year dataframe
    print("\n[5/9] Assembling full-year simulation frame ...")
    idx = pd.date_range(f"{cfg.data.evaluation_year}-01-01", f"{cfg.data.evaluation_year+1}-01-01",
                         freq="1h", tz=cfg.site.tz, inclusive="left")
    year_df = pd.DataFrame(index=idx)
    ld = load_df.reindex(idx).ffill().bfill()
    # Generalize to 1 or more meters
    num_meters = len(cfg.data.load_kw_cols)
    meter_keys = [f"mdp{i+1}" for i in range(num_meters)]
    for k in meter_keys:
        year_df[k] = ld[k].values
    year_df["load_kw"] = ld["combined_kw"].values
    year_df["solar_kw"] = solar_2025.reindex(idx).fillna(0).values
    year_df["lmp"] = lmp.reindex(idx).ffill().bfill().values
    print(f"  Year frame: {len(year_df)} hours, {num_meters} meter(s)")

    # --- 6. Set up tariff engines + battery
    delivery = build_delivery(dict(cfg.delivery_tariff))
    supply = build_supply(supply_cfg)
    battery = BatterySpec.from_cfg(dict(cfg.battery))
    tag_costs = supply.annual_tag_cost(supply.capacity_obligation_kw, supply.nspl_kw)
    # Delivery-demand semantics differ: ComEd bills DFC on on-peak-only kW; PECO bills
    # the overall max kW. The dispatch LP uses this to decide which hours constrain
    # peak_kw. Flag is inferred from the delivery-tariff kind.
    demand_on_peak_only = dict(cfg.delivery_tariff).get("kind", "comed_vll_secondary") == "comed_vll_secondary"
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
        baseline_df, meter_keys, delivery, supply, lmp, plc_hours, [nspl_hour], "Baseline (no battery)"
    )
    print(f"  Baseline annual: delivery ${baseline_result.delivery_cost:,.0f}  "
          f"supply ${baseline_result.supply_cost:,.0f}  TOTAL ${baseline_result.total:,.0f}")

    # Perfect foresight
    print("\n[7/9] Running perfect-foresight dispatch ...")
    # For PECO, use flat dist rate. For ComEd, use monthly DFC schedule.
    _dt = dict(cfg.delivery_tariff)
    if "dfc_per_kw_monthly" in _dt:
        dfc_monthly_cfg = dict(_dt["dfc_per_kw_monthly"])
    else:
        flat = _dt.get("distribution_rate_per_kw", 7.12)
        dfc_monthly_cfg = {m: flat for m in range(1, 13)}

    pf_res = perfect_foresight_dispatch(
        year_df, battery, dfc_monthly_cfg,
        plc_hours, [nspl_hour], dict(cfg.mpc),
        export_allowed=cfg.export.allowed,
        export_rate_per_kwh=cfg.export.rate_per_kwh_if_allowed,
        demand_on_peak_only=demand_on_peak_only,
    )
    pf_df = pf_res.df.join(year_df[meter_keys])
    pf_result = annual_cost_from_dispatch(
        pf_df, meter_keys, delivery, supply, lmp, plc_hours, [nspl_hour], "Perfect foresight"
    )
    comparison_pf = compare_scenarios(baseline_result, pf_result)
    print(f"  PF annual: ${pf_result.total:,.0f}  savings=${comparison_pf['battery_value_annual']:,.0f}")

    # MEMOSA controls — average over 5 forecast-noise seeds for a stable point estimate.
    # Flat-load sites (cold storage) are very noise-sensitive on DFC target-hour
    # identification; more seeds dramatically reduce variance of the mean.
    print("\n[8/9] Running MEMOSA controls dispatch (5-seed mean for stable point estimate) ...")
    seeds = [42, 123, 7, 2025, 1337]
    seed_results = []
    seed_savings = []
    for sd in seeds:
        mpc_raw_i = rolling_mpc_dispatch(
            year_df, battery, dfc_monthly_cfg,
            plc_hours, [nspl_hour], dict(cfg.mpc),
            export_allowed=cfg.export.allowed,
            export_rate_per_kwh=cfg.export.rate_per_kwh_if_allowed,
            seed=sd,
            demand_on_peak_only=demand_on_peak_only,
        )
        mpc_df_i = mpc_raw_i.join(year_df[meter_keys])
        res_i = annual_cost_from_dispatch(
            mpc_df_i, meter_keys, delivery, supply, lmp, plc_hours, [nspl_hour],
            f"MEMOSA seed={sd}"
        )
        comp_i = compare_scenarios(baseline_result, res_i)
        seed_results.append((sd, mpc_raw_i, mpc_df_i, res_i, comp_i))
        seed_savings.append(comp_i["battery_value_annual"])
        print(f"    seed {sd:>4}: savings ${comp_i['battery_value_annual']:,.0f}")

    import statistics
    mean_savings = statistics.mean(seed_savings)
    stdev_savings = statistics.stdev(seed_savings) if len(seed_savings) > 1 else 0.0
    stderr_mean = stdev_savings / (len(seeds) ** 0.5)
    # Pick the seed whose savings are closest to the mean as the representative trajectory
    rep_idx = min(range(len(seed_savings)), key=lambda i: abs(seed_savings[i] - mean_savings))
    _, mpc_df_raw, mpc_df, mpc_result, comparison_mpc = seed_results[rep_idx]
    # Compute the PF-scaled central estimate using the gap-factor floor; this is the
    # primary reported number because it is STABLE for all sites (flat-load sites have
    # high seed variance that makes the mean itself uncertain).
    pf_savings_primary = comparison_pf["battery_value_annual"]
    raw_gap = (mean_savings / pf_savings_primary) if pf_savings_primary > 0 else 0.83
    mpc_gap_factor = max(0.70, min(1.00, raw_gap))
    central_memosa = pf_savings_primary * mpc_gap_factor
    # Expose both: reported headline = PF-scaled central; seed-mean + stderr = realized trajectory noise
    comparison_mpc["battery_value_annual"] = central_memosa
    comparison_mpc["battery_value_annual_pf_scaled"] = central_memosa
    comparison_mpc["battery_value_annual_seed_mean"] = mean_savings
    comparison_mpc["battery_value_annual_seed_stdev"] = stdev_savings
    comparison_mpc["battery_value_annual_seed_stderr"] = stderr_mean
    comparison_mpc["mpc_gap_factor"] = mpc_gap_factor
    comparison_mpc["raw_gap_factor"] = raw_gap
    print(f"  Seed-mean savings: ${mean_savings:,.0f}  (stdev ${stdev_savings:,.0f}, stderr ${stderr_mean:,.0f})")
    print(f"  PF-scaled central: ${central_memosa:,.0f}  (PF × gap={mpc_gap_factor:.2f})")
    if raw_gap < 0.70:
        print(f"  NOTE: raw proxy MPC/PF ratio = {raw_gap:.2f} < 0.70 floor → ")
        print(f"  Flat-load dispatch instability. Reporting PF × 0.70 floor as realistic controller assumption.")

    # Rule-based
    print("\n  ... rule-based dispatch ...")
    rb_df_raw = rule_based_dispatch(year_df, battery, demand_on_peak_only=demand_on_peak_only)
    rb_df = rb_df_raw.join(year_df[meter_keys])
    rb_result = annual_cost_from_dispatch(
        rb_df, meter_keys, delivery, supply, lmp, plc_hours, [nspl_hour], "Rule-based"
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
    headline_t1 = comparison_mpc["battery_value_annual"] + stack.get("tier1_total", 0)
    combined_t1_t2 = headline_t1 + stack.get("tier2_total", 0)
    print()
    print(f"  >>> HEADLINE (battery + currently-enrolled DR expansion): ${headline_t1:,.0f}/yr")
    print(f"        Core battery only:                 ${comparison_mpc['battery_value_annual']:,.0f}")
    print(f"        Tier 1 incremental DR (existing):  ${stack.get('tier1_total', 0):,.0f}  ({', '.join(stack.get('tier1_names', [])) or 'none'})")
    print(f"  >>> Optional tier-2 (NEW program enroll): +${stack.get('tier2_total', 0):,.0f}  ({', '.join(stack.get('tier2_names', [])) or 'none'})")
    print(f"  >>> Combined with both tiers: ${combined_t1_t2:,.0f}/yr")

    # --- 9. Tornado sensitivity
    print("\n[10/11] Tornado sensitivity (8 inputs, ~16 LP solves) ...")
    baseline_value_for_sens = comparison_mpc["battery_value_annual"]
    plc_reduction_value = comparison_mpc["streams"]["plc_capacity_tag_reduction"]
    tornado = run_tornado(
        year_df, lmp, battery, delivery, supply,
        dfc_monthly_cfg,
        plc_hours, nspl_hour, dict(cfg.mpc),
        baseline_value=baseline_value_for_sens,
        export_allowed=cfg.export.allowed,
        export_rate=cfg.export.rate_per_kwh_if_allowed,
        plc_reduction_value=plc_reduction_value,
        mpc_gap_factor=mpc_gap_factor,
        demand_on_peak_only=demand_on_peak_only,
    )
    print("  Ranked by swing (|hi-lo|):")
    for r in tornado:
        print(f"    {r['label']:<45} lo=${r['lo']:>10,.0f}  hi=${r['hi']:>10,.0f}  span=${r['span']:>10,.0f}")

    # --- 9.5 Battery-size sweep (PF-deterministic × gap_factor)
    print(f"\n[10/12] Battery-size sweep (6 block sizes, PF × {mpc_gap_factor:.2f} gap) ...")
    sizes_cfg = cfg.get("battery_sizes_to_sweep", [])
    sizes = [(s["power_kw"], s["energy_kwh"], s["label"]) for s in sizes_cfg]
    sizing_results = run_sweep(
        sizes, year_df, lmp, dict(cfg.battery), delivery, supply,
        dfc_monthly_cfg,
        plc_hours, nspl_hour, dict(cfg.mpc),
        export_allowed=cfg.export.allowed,
        export_rate=cfg.export.rate_per_kwh_if_allowed,
        mpc_gap_factor=mpc_gap_factor,
        demand_on_peak_only=demand_on_peak_only,
    )
    print(f"  {'Size':<20} {'Annual $':>12} {'Marg $/kWh':>14}")
    for r in sizing_results:
        print(f"  {r.label:<20} ${r.annual_savings:>10,.0f}  ${r.marginal_savings_per_added_kwh:>11.2f}")
    knee = detect_knee(sizing_results, extreme_drop_ratio=0.40)
    print(f"  >>> Plateau-based recommendation: {knee['knee_label']}  (${knee['knee_annual_savings']:,.0f}/yr)")
    if knee.get("knee_found"):
        print(f"      extreme plateau detected — overbuild starts above this size")
    else:
        print(f"      no extreme plateau in tested range — this is the largest tested size")

    # Secondary: marginal-payback economics
    econ_cfg = cfg.get("sizing_economics", {})
    capex_c = econ_cfg.get("marginal_capex_per_kwh_added", 350)
    capex_lo = econ_cfg.get("marginal_capex_low", 250)
    capex_hi = econ_cfg.get("marginal_capex_high", 500)
    target_yr = econ_cfg.get("target_simple_payback_years", 10)
    attach_payback_analysis(sizing_results, capex_c, capex_lo, capex_hi)
    payback_rec = recommend_largest_worth_it(sizing_results, target_yr)
    print(f"\n  Secondary (capex-based): assume ${capex_c}/kWh marginal capex, target payback ≤ {target_yr} yr")
    print(f"  {'Size':<20} {'Δ kWh':>7} {'Δ Savings/yr':>13} {'Δ Capex':>12} {'Payback':>10}")
    for r in sizing_results:
        pb = f"{r.marginal_payback_years:.1f} yr" if r.marginal_payback_years != float('inf') else "n/a"
        print(f"  {r.label:<20} {r.marginal_added_kwh:>7.0f} ${r.marginal_annual_savings:>11,.0f}  "
              f"${r.marginal_capex:>10,.0f}  {pb:>10}")
    print(f"  >>> Capex-based recommendation: {payback_rec['recommended_label']}")
    print(f"      {payback_rec['rationale']}")

    # --- 10. Monte Carlo
    print("\n[11/12] Monte Carlo confidence band (20 samples) ...")
    num_anchors = len(anchor_list_raw)
    mc = run_monte_carlo(
        year_df, lmp, battery, delivery, supply,
        dfc_monthly_cfg,
        plc_hours, nspl_hour, dict(cfg.mpc),
        baseline_value=baseline_value_for_sens,
        n_samples=20,
        export_allowed=cfg.export.allowed,
        export_rate=cfg.export.rate_per_kwh_if_allowed,
        plc_reduction_value=plc_reduction_value,
        num_known_anchor_hours=num_anchors,
        total_plc_hours=5,
        mpc_gap_factor=mpc_gap_factor,
        demand_on_peak_only=demand_on_peak_only,
    )
    print(f"  P10: ${mc['p10']:,.0f}   P50: ${mc['p50']:,.0f}   P90: ${mc['p90']:,.0f}")
    print(f"  Mean: ${mc['mean']:,.0f}  StDev: ${mc['std']:,.0f}")

    # --- 12. Report
    out_dir = Path("results") / cfg.site.name
    print(f"\n[12/12] Building report in {out_dir}/ ...")
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
        sizing_results=sizing_results,
        sizing_knee=knee,
        payback_rec=payback_rec,
        capex_assumptions={"central": capex_c, "low": capex_lo, "high": capex_hi, "target_yr": target_yr},
    )
    print(f"Report: {html}")
    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m bess_eval.cli <config.yaml>")
        sys.exit(1)
    run(sys.argv[1])
