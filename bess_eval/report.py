"""HTML + JSON report generation."""
from __future__ import annotations
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{{title}}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1, h2, h3 { color: #1a1a2e; }
  .hero { background: #f0f4f8; padding: 1.5em; border-radius: 8px; margin-bottom: 2em; }
  .hero .big { font-size: 2.5em; font-weight: 700; color: #0f766e; }
  table { width: 100%; border-collapse: collapse; margin: 1em 0; }
  th, td { padding: 0.5em; text-align: left; border-bottom: 1px solid #e5e7eb; }
  th { background: #f9fafb; font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .stream-row td.num { font-weight: 600; }
  .caveat { background: #fef3c7; padding: 0.8em; border-left: 4px solid #f59e0b; margin: 1em 0; }
  .note { background: #dbeafe; padding: 0.8em; border-left: 4px solid #3b82f6; margin: 1em 0; font-size: 0.95em; }
  img { max-width: 100%; border: 1px solid #e5e7eb; margin: 0.5em 0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }
</style>
</head>
<body>
<h1>{{title}}</h1>
<p style="color:#6b7280">Generated {{generated_at}}</p>

<div class="hero">
  <div>Annual $ saved by battery (MPC-realistic dispatch)</div>
  <div class="big">${{mpc_savings_fmt}}</div>
  <div style="color:#6b7280; margin-top: 0.5em">Perfect-foresight upper bound: ${{pf_savings_fmt}} &nbsp;|&nbsp; Rule-based floor: ${{rb_savings_fmt}}</div>
</div>

<h2>1. Site & data summary</h2>
<table>
  <tr><th>Site</th><td>{{site_name}} — {{site_addr}}</td></tr>
  <tr><th>Battery</th><td>{{battery_kw}} kW / {{battery_kwh}} kWh ({{battery_coupling}}-coupled, RTE {{battery_rte}})</td></tr>
  <tr><th>Evaluation year</th><td>{{year}}</td></tr>
  <tr><th>Annual load</th><td>{{annual_kwh}} kWh</td></tr>
  <tr><th>Peak load</th><td>{{peak_kw}} kW</td></tr>
  <tr><th>Synthesized annual solar</th><td>{{solar_annual}} kWh ({{solar_cf}}% capacity factor)</td></tr>
</table>

<h2>2. Annual $ saved by value stream (MPC-realistic)</h2>
<table>
  <tr><th>Stream</th><th class="num">Annual $ saved</th><th class="num">Share</th></tr>
  {{stream_rows}}
  <tr class="stream-row"><td><strong>Total</strong></td><td class="num"><strong>${{mpc_savings_fmt}}</strong></td><td class="num">100.0%</td></tr>
</table>

<h2>3. DR program opportunity value (if enrolled)</h2>
<p>These revenues are <em>additive</em> to the core battery value above, assuming enrollment. Only one can typically be held at a time for the full 1,500 kW capacity.</p>
<table>
  <tr><th>Program</th><th class="num">Capacity revenue</th><th class="num">Energy revenue</th><th class="num">Opportunity cost</th><th class="num">Net annual value</th></tr>
  {{dr_rows}}
</table>

<h2>4. Dispatch comparison (Perfect Foresight vs MPC vs Rule-Based)</h2>
<table>
  <tr><th>Scenario</th><th class="num">Annual delivery</th><th class="num">Annual supply</th><th class="num">Total annual</th><th class="num">$ saved vs baseline</th></tr>
  {{scenario_rows}}
</table>

<h2>5. Baseline vs With-Battery: monthly bill comparison</h2>
<img src="data:image/png;base64,{{plot_monthly_b64}}" alt="monthly bill">

<h2>6. Sample dispatch weeks</h2>
<div><strong>Winter sample week</strong> — shows baseline + battery dispatch + SOC</div>
<img src="data:image/png;base64,{{plot_winter_b64}}" alt="winter">
<div><strong>Summer peak week (around PJM 5CP)</strong></div>
<img src="data:image/png;base64,{{plot_summer_b64}}" alt="summer">

<h2>7. PJM tag reduction (PLC / NSPL)</h2>
<table>
  <tr><th>Tag</th><th class="num">Baseline kW</th><th class="num">With battery kW</th><th class="num">$/kW-yr</th><th class="num">Annual $ saved</th></tr>
  {{tag_rows}}
</table>

<h2>8. Modeling assumptions & caveats</h2>
<ul>{{caveat_list}}</ul>

<h2>9. Methodology — brief</h2>
<div class="note">
{{methodology}}
</div>

</body></html>
"""


def _fmt_dollar(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.0f}"


def _plot_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_monthly_bills(baseline_bills: list, with_battery_bills: list) -> str:
    months = [b["month"] for b in baseline_bills]
    b = [b["total"] for b in baseline_bills]
    w = [b_["total"] for b_ in with_battery_bills]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(months))
    ax.bar(x - 0.2, b, 0.4, label="Baseline", color="#64748b")
    ax.bar(x + 0.2, w, 0.4, label="With battery", color="#0f766e")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.set_ylabel("$ / month (delivery bill)")
    ax.set_title("Monthly delivery bill — baseline vs with battery")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _plot_to_b64(fig)


def plot_week(df: pd.DataFrame, title: str, battery_energy_kwh: float) -> str:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    t = df.index
    a1.plot(t, df["load_kw"], label="Load", color="#334155", lw=1.1)
    a1.plot(t, df["solar_kw"], label="Solar", color="#f59e0b", lw=1.0)
    a1.plot(t, df["grid_import"], label="Grid import (with battery)", color="#0ea5e9", lw=1.2)
    a1.plot(t, -df["p_chg"], label="Battery charge (−)", color="#7c3aed", lw=0.8, alpha=0.8)
    a1.plot(t, df["p_dis"], label="Battery discharge (+)", color="#10b981", lw=0.8, alpha=0.8)
    a1.legend(loc="upper right", fontsize=8)
    a1.set_ylabel("kW")
    a1.set_title(title)
    a1.grid(alpha=0.3)

    a2.plot(t, df["soc_kwh"] / battery_energy_kwh * 100, label="SOC %", color="#ef4444")
    a2.set_ylabel("SOC %")
    a2.set_ylim(0, 100)
    a2.grid(alpha=0.3)
    plt.setp(a2.get_xticklabels(), rotation=30, ha="right")
    return _plot_to_b64(fig)


def build_report(
    out_dir: Path,
    site_cfg: dict,
    data_summary: dict,
    solar_annual_kwh: float,
    battery_cfg: dict,
    comparison_mpc: dict,
    comparison_pf: dict,
    comparison_rb: dict,
    dr_values: list,
    baseline_result: object,
    mpc_result: object,
    pf_result: object,
    rb_result: object,
    mpc_df: pd.DataFrame,
    pf_df: pd.DataFrame,
    rb_df: pd.DataFrame,
    plc_hours: list,
    nspl_hour: pd.Timestamp,
    tag_costs: dict,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build stream rows from comparison_mpc
    streams = comparison_mpc["streams"]
    total = comparison_mpc["battery_value_annual"]
    stream_labels = {
        "dfc_demand_reduction": "Distribution Facility Charge reduction (on-peak demand)",
        "other_delivery_riders_taxes": "Delivery riders & taxes (scales with kWh)",
        "energy_arbitrage": "Energy arbitrage (LMP-shape)",
        "plc_capacity_tag_reduction": "PJM capacity tag reduction (PLC)",
        "nspl_transmission_tag_reduction": "PJM transmission tag reduction (NSPL)",
    }
    stream_rows = ""
    for key, val in sorted(streams.items(), key=lambda kv: -kv[1]):
        share = val / total * 100 if total else 0
        label = stream_labels.get(key, key)
        stream_rows += f'<tr><td>{label}</td><td class="num">${_fmt_dollar(val)}</td><td class="num">{share:.1f}%</td></tr>\n'

    dr_rows = ""
    for dr in dr_values:
        dr_rows += (
            f'<tr><td>{dr.name}</td>'
            f'<td class="num">${_fmt_dollar(dr.capacity_revenue)}</td>'
            f'<td class="num">${_fmt_dollar(dr.energy_revenue)}</td>'
            f'<td class="num">${_fmt_dollar(dr.opportunity_cost)}</td>'
            f'<td class="num">${_fmt_dollar(dr.net_annual_value)}</td></tr>\n'
        )

    scenario_rows = ""
    for label, comp, res in [
        ("Perfect foresight (upper bound)", comparison_pf, pf_result),
        ("MPC-realistic (reported)",         comparison_mpc, mpc_result),
        ("Rule-based (floor)",                comparison_rb, rb_result),
    ]:
        scenario_rows += (
            f'<tr><td>{label}</td>'
            f'<td class="num">${_fmt_dollar(res.delivery_cost)}</td>'
            f'<td class="num">${_fmt_dollar(res.supply_cost)}</td>'
            f'<td class="num">${_fmt_dollar(res.total)}</td>'
            f'<td class="num">${_fmt_dollar(comp["battery_value_annual"])}</td></tr>\n'
        )

    # Tag rows
    plc_base = baseline_result.per_stream["plc_kw_post"]
    plc_post = mpc_result.per_stream["plc_kw_post"]
    nspl_base = baseline_result.per_stream["nspl_kw_post"]
    nspl_post = mpc_result.per_stream["nspl_kw_post"]
    tag_rows = (
        f'<tr><td>PJM capacity (PLC)</td><td class="num">{plc_base:,.0f}</td>'
        f'<td class="num">{plc_post:,.0f}</td>'
        f'<td class="num">${tag_costs["capacity_per_kw_yr"]:,.0f}</td>'
        f'<td class="num">${_fmt_dollar((plc_base-plc_post)*tag_costs["capacity_per_kw_yr"])}</td></tr>\n'
        f'<tr><td>PJM transmission (NSPL)</td><td class="num">{nspl_base:,.0f}</td>'
        f'<td class="num">{nspl_post:,.0f}</td>'
        f'<td class="num">${tag_costs["transmission_per_kw_yr"]:,.0f}</td>'
        f'<td class="num">${_fmt_dollar((nspl_base-nspl_post)*tag_costs["transmission_per_kw_yr"])}</td></tr>\n'
    )

    # Sample weeks
    winter_start = pd.Timestamp("2025-01-13", tz=mpc_df.index.tz)
    winter_week = mpc_df.loc[(mpc_df.index >= winter_start) & (mpc_df.index < winter_start + pd.Timedelta(days=7))]
    if plc_hours:
        summer_anchor = pd.Timestamp(plc_hours[0]).normalize() - pd.Timedelta(days=3)
    else:
        summer_anchor = pd.Timestamp("2025-07-21", tz=mpc_df.index.tz)
    summer_week = mpc_df.loc[(mpc_df.index >= summer_anchor) & (mpc_df.index < summer_anchor + pd.Timedelta(days=7))]

    plot_monthly = plot_monthly_bills(baseline_result.monthly_bills, mpc_result.monthly_bills)
    plot_winter = plot_week(winter_week, f"Winter week {winter_start.date()}", battery_cfg["energy_kwh"])
    plot_summer = plot_week(summer_week, f"Summer peak week {summer_anchor.date()}", battery_cfg["energy_kwh"])

    caveats = [
        "Load data is hourly; ComEd bills demand on 30-minute intervals. This understates demand savings by ~2-3%.",
        "Solar is synthesized from a pvlib clear-sky model + Midwest monthly climatological cloud fractions, calibrated to 2026 Jan-Apr meter data. Hour-to-hour cloud variability not captured.",
        "PJM ComEd-zone LMPs are synthesized from published typical diurnal/seasonal shape, anchored to the Aug-Sep 2025 bill's observed period-average ($0.0327/kWh). No live ISO data available in this environment.",
        "PJM 5CP hours are proxied from the 5 highest weekday 2-7pm site-load hours in Jun-Sep 2025, anchored to the 8/15/25 18:00 hour confirmed on the Aug-Sep supply bill.",
        "MPC-realistic dispatch uses the 'noisy foresight' approximation: inputs degraded by realistic AR(1) forecast error (load MAPE 8%, solar 20%, LMP 15%), then perfect-foresight LP solves over the noised inputs. Matches 10-15% MPC-vs-PF gap reported in BESS literature.",
        "Battery is modeled as installed at the PCC upstream of both MDP meters, with battery flow allocated proportionally to each meter's load share for reporting billed demand. Real-world wiring may differ.",
        "ComEd DG net metering caps at 2 MW non-residential; the ~4 MW PV system is not eligible, so battery does not export to grid (export compensation = $0).",
        "Franchise and municipal tax rates back-solved from two 2025 bills; monthly-interpolated schedules approximate DSPR reconciliation drift.",
    ]
    caveat_list = "".join(f"<li>{c}</li>" for c in caveats)

    methodology = (
        "<b>Tariff engine:</b> ComEd VLL Secondary Voltage delivery rate (Customer Charge + Metering + "
        "Distribution Facility Charge on on-peak kW per-meter summed + IEDT on kWh + riders + franchise + IL Excise + Joliet MUT), "
        "calibrated against both 2025 bills within ±3.5%. Supply side: Aug-Sep bill index structure (LMP + losses + fixed adder + PLC capacity + NSPL transmission + TEC). "
        "<b>Dispatch:</b> single-variable Pyomo LP per month with SOC continuity across months, "
        "minimizing (supply energy × LMP + DFC $/kW × on-peak peak + high penalty × grid_import at PLC/NSPL hours). "
        "<b>Attribution:</b> same tariff engine applied to baseline (no battery) vs with-battery trajectories; delta by line item "
        "decomposes into demand, energy, and tag streams. <b>Value is battery-only</b> — solar is present in both scenarios. "
        "Capex, NPV, IRR are intentionally omitted per scope; raw annual savings enable external ROI analysis."
    )

    html = HTML_TEMPLATE
    replacements = {
        "{{title}}": f"Battery Value Evaluation — {site_cfg['name']}",
        "{{generated_at}}": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "{{site_name}}": site_cfg["name"],
        "{{site_addr}}": site_cfg["address"],
        "{{battery_kw}}": f"{battery_cfg['power_kw']:,.0f}",
        "{{battery_kwh}}": f"{battery_cfg['energy_kwh']:,.0f}",
        "{{battery_coupling}}": battery_cfg["coupling"],
        "{{battery_rte}}": f"{battery_cfg['rte_ac_ac']*100:.1f}%",
        "{{year}}": str(data_summary.get("evaluation_year", 2025)),
        "{{annual_kwh}}": f"{data_summary['annual_kwh']:,.0f}",
        "{{peak_kw}}": f"{data_summary['peak_kw']:,.1f}",
        "{{solar_annual}}": f"{solar_annual_kwh:,.0f}",
        "{{solar_cf}}": f"{solar_annual_kwh / (4000 * 8760) * 100:.1f}",
        "{{mpc_savings_fmt}}": _fmt_dollar(comparison_mpc["battery_value_annual"]),
        "{{pf_savings_fmt}}":  _fmt_dollar(comparison_pf["battery_value_annual"]),
        "{{rb_savings_fmt}}":  _fmt_dollar(comparison_rb["battery_value_annual"]),
        "{{stream_rows}}":    stream_rows,
        "{{dr_rows}}":        dr_rows,
        "{{scenario_rows}}":  scenario_rows,
        "{{tag_rows}}":       tag_rows,
        "{{plot_monthly_b64}}": plot_monthly,
        "{{plot_winter_b64}}":  plot_winter,
        "{{plot_summer_b64}}":  plot_summer,
        "{{caveat_list}}":    caveat_list,
        "{{methodology}}":    methodology,
    }
    for k, v in replacements.items():
        html = html.replace(k, str(v))

    html_path = out_dir / "report.html"
    html_path.write_text(html)

    # Raw JSON dump
    json_path = out_dir / "results.json"
    payload = {
        "site": site_cfg,
        "data_summary": data_summary,
        "battery": battery_cfg,
        "solar_annual_kwh": solar_annual_kwh,
        "comparison_mpc": comparison_mpc,
        "comparison_pf": comparison_pf,
        "comparison_rb": comparison_rb,
        "dr_programs": [d.__dict__ for d in dr_values],
        "plc_hours": [str(h) for h in plc_hours],
        "nspl_hour": str(nspl_hour),
        "tag_costs": tag_costs,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    # Hourly dispatch CSV
    mpc_df.to_csv(out_dir / "dispatch_mpc.csv")

    return html_path
