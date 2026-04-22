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
  <div>Annual $ saved — battery + expansion of currently-enrolled DR</div>
  <div class="big">${{headline_with_tier1_fmt}}</div>
  <div style="color:#374151; margin-top: 0.5em">
    &nbsp;&nbsp;= Core battery savings <strong>${{mpc_savings_fmt}}</strong>
    &nbsp; + &nbsp; Existing DR expansion (incremental) <strong>${{tier1_total_fmt}}</strong>
  </div>
  <div style="color:#0f766e; margin-top: 0.5em; font-style: italic">
    {{tier1_names_fmt}} — already enrolled (per site project list). Battery enables
    expanding committed kW via contract amendment with existing CSP. No new program
    registration, no new vendor, no new operational changes.
  </div>
  <div style="color:#1e3a8a; margin-top: 1em; padding: 0.5em 0.8em; background: #eff6ff; border-radius: 6px">
    <strong>Optional additional value if you also enroll in NEW programs:</strong>
    +${{tier2_total_fmt}}/yr from {{tier2_names_fmt}}<br>
    <span style="color:#3b82f6">→ Combined potential (battery + tier 1 + tier 2): <strong>${{combined_with_tier2_fmt}}/yr</strong></span>
  </div>
  <div style="color:#374151; margin-top: 1em; font-size: 0.95em">
    <strong>Battery-only confidence band (Monte Carlo, N=20):</strong>
    P10 <strong>${{mc_p10_fmt}}</strong> &nbsp;|&nbsp; P50 <strong>${{mc_p50_fmt}}</strong> &nbsp;|&nbsp; P90 <strong>${{mc_p90_fmt}}</strong>
  </div>
  <div style="color:#6b7280; margin-top: 0.5em; font-size: 0.9em">
    Perfect-foresight upper bound: ${{pf_savings_fmt}} &nbsp;|&nbsp; Rule-based floor: ${{rb_savings_fmt}}<br>
    Realized-dispatch reference (5-seed mean): ${{seed_mean_fmt}} (stderr ${{seed_stderr_fmt}})
  </div>
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

<h2>2. Annual $ saved by value stream (MEMOSA controls)</h2>
<table>
  <tr><th>Stream</th><th class="num">Annual $ saved</th><th class="num">Share</th></tr>
  {{stream_rows}}
  <tr class="stream-row"><td><strong>Total</strong></td><td class="num"><strong>${{mpc_savings_fmt}}</strong></td><td class="num">100.0%</td></tr>
</table>

<h2>3. DR program opportunity value (honest stacking + enrollment analysis)</h2>
<p>The "Status" column states whether the site is <strong>currently enrolled</strong> in this program (sites in this study are all enrolled in PJM Emergency Load Response via Voltus, manual control). For currently-enrolled programs, the headline net value is the <strong>incremental</strong> revenue from expanding the committed kW with battery capacity — it is NOT the gross program value (which would double-count what the site is already collecting). For non-enrolled programs, the headline equals the gross.</p>
<p>Capacity DR programs are mutually exclusive per asset (one registration only), so adding a different one means switching, not stacking. Economic Energy DR stacks freely with capacity DR.</p>
<table>
  <tr><th>Program</th><th>Type</th><th>Status</th><th class="num">Gross value (full enrollment)</th><th class="num">Reported net value</th></tr>
  {{dr_rows}}
</table>
<div class="note" style="margin-top: 1em">
  <strong>Recommended stack:</strong> {{dr_stack_names}}<br>
  <strong>Stacked incremental annual value:</strong> ${{dr_stack_total}}<br>
  This is purely additive to the core battery value above (no double-count with existing operations).
</div>

<h2>4. Dispatch comparison (Perfect Foresight vs MEMOSA controls vs Rule-Based)</h2>
<table>
  <tr><th>Scenario</th><th class="num">Annual delivery</th><th class="num">Annual supply</th><th class="num">Total annual</th><th class="num">$ saved vs baseline</th></tr>
  {{scenario_rows}}
</table>

<h2>5. Battery sizing — optimal block size</h2>
<p>MEMOSA-controls dispatch was re-run for each available block size. We flag a "knee" only when the curve shows an <strong>extreme</strong> plateau — i.e., a single inter-block step where marginal $/added-kWh collapses by &gt;60% (ratio &lt;0.40). Gentle concave diminishing returns are NOT flagged as a knee; those situations correctly recommend the largest tested size.</p>
<img src="data:image/png;base64,{{plot_sizing_b64}}" alt="sizing">
<table>
  <tr><th>Size</th><th class="num">Power (kW AC)</th><th class="num">Energy (kWh)</th><th class="num">Annual $ saved</th><th class="num">Marginal $/added kWh</th><th>Recommendation</th></tr>
  {{sizing_rows}}
</table>
<div class="note" style="margin-top: 1em">
  <strong>Plateau-based recommendation:</strong> {{knee_label}} &mdash; ${{knee_savings}}/yr.<br>
  <em>{{knee_rationale}}</em>
  {{knee_vs_primary_note}}
</div>

<h3>5a. Secondary sizing check — marginal capex payback</h3>
<p>The plateau-based recommendation above ignores what each additional block <em>costs</em>. This secondary check answers: <strong>if each incremental block costs roughly <code>${{capex_central}}/kWh</code> of added capacity (typical 2025 utility-scale Li-ion 2-hour BESS), is each step still worth the incremental capex?</strong> We compute simple payback on the incremental investment alone and flag the largest block whose marginal payback stays within the target (<code>{{target_yr}}-yr</code>) threshold. Capex band shown: <code>${{capex_low}}-${{capex_high}}/kWh</code>.</p>
<table>
  <tr><th>Size</th><th class="num">Δ kWh</th><th class="num">Δ Savings/yr</th><th class="num">Δ Capex @ ${{capex_central}}/kWh</th><th class="num">Marg payback (yr)</th><th class="num">Band (yr)</th></tr>
  {{payback_rows}}
</table>
<div class="note" style="margin-top: 1em">
  <strong>Capex-based recommendation:</strong> {{payback_rec_label}} &mdash; incremental payback {{payback_rec_years}} yr.<br>
  <em>{{payback_rationale}}</em>
</div>

<div class="caveat">
  <strong>Capex assumption is an estimate, not a quote.</strong> Installed BESS pricing varies widely with procurement strategy, labor market, domestic content requirements, permitting, interconnection studies, and soft costs. The central ${{capex_central}}/kWh assumes marginal module+BoS pricing typical of 2-hour Li-ion systems at this scale; actual quotes range roughly ${{capex_low}}-${{capex_high}}/kWh. <strong>Plug in your real quote to get a firm answer.</strong> If you enter a specific capex and payback target into <code>sizing_economics</code> in the config, this section will update automatically.
</div>

<h2>6. Baseline vs With-Battery: monthly bill comparison</h2>
<img src="data:image/png;base64,{{plot_monthly_b64}}" alt="monthly bill">

<h2>7. Sample dispatch weeks</h2>
<div><strong>Winter sample week</strong> — shows baseline + battery dispatch + SOC</div>
<img src="data:image/png;base64,{{plot_winter_b64}}" alt="winter">
<div><strong>Summer peak week (around PJM 5CP)</strong></div>
<img src="data:image/png;base64,{{plot_summer_b64}}" alt="summer">

<h2>8. PJM tag reduction (PLC / NSPL)</h2>
<table>
  <tr><th>Tag</th><th class="num">Baseline kW</th><th class="num">With battery kW</th><th class="num">$/kW-yr</th><th class="num">Annual $ saved</th></tr>
  {{tag_rows}}
</table>

<h2>9. Uncertainty — what could move the number and by how much</h2>

<h3>9a. Tornado — single-factor sensitivity</h3>
<p>Each bar shows how the annual battery value shifts if that assumption varies by ±1σ while everything else stays at baseline. Sorted by swing magnitude.</p>
<img src="data:image/png;base64,{{plot_tornado_b64}}" alt="tornado">

<h3>9b. Monte Carlo — joint uncertainty</h3>
<p>Samples all uncertain assumptions simultaneously from their distributions (solar ±10%, LMP level ±12%, LMP shape intensity ±15%, PLC identification miss prob 20% per non-anchor hour, controller forecast skill ±typical range), then re-runs full dispatch + attribution per sample.</p>
<img src="data:image/png;base64,{{plot_mc_b64}}" alt="monte carlo">
<p><strong>P10 ${{mc_p10_fmt}} &middot; P50 ${{mc_p50_fmt}} &middot; P90 ${{mc_p90_fmt}}</strong>. Half the plausible outcomes lie between P25 and P75; 80% lie between P10 and P90.</p>

<h2>10. Modeling assumptions & caveats</h2>
<ul>{{caveat_list}}</ul>

<h2>11. Methodology — brief</h2>
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


def plot_sizing(sizing_results: list, knee: dict) -> str:
    sr = sorted(sizing_results, key=lambda r: r.energy_kwh)
    x = [r.power_kw for r in sr]
    y = [r.annual_savings for r in sr]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, y, "o-", color="#1e40af", lw=2, markersize=9)
    for r in sr:
        color = "#0f766e" if r.label == knee.get("knee_label") else "#1e40af"
        weight = "bold" if r.label == knee.get("knee_label") else "normal"
        ax.annotate(
            f"{r.label}\n${r.annual_savings:,.0f}",
            xy=(r.power_kw, r.annual_savings),
            xytext=(6, 10), textcoords="offset points",
            fontsize=9, color=color, fontweight=weight,
        )
    # Mark the knee
    knee_r = next((r for r in sr if r.label == knee.get("knee_label")), None)
    if knee_r:
        ax.plot([knee_r.power_kw], [knee_r.annual_savings], "o",
                markersize=18, markerfacecolor="none", markeredgecolor="#0f766e", markeredgewidth=2.5,
                label=f"Knee: {knee_r.label}")
    ax.set_xlabel("Battery power (kW AC)")
    ax.set_ylabel("Annual $ saved (MEMOSA controls)")
    ax.set_title("Battery sizing sweep — returns vs size")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(x) * 1.15)
    return _plot_to_b64(fig)


def plot_tornado(tornado: list, baseline_value: float) -> str:
    labels = [r["label"] for r in tornado]
    lo = [r["delta_lo"] for r in tornado]
    hi = [r["delta_hi"] for r in tornado]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    for yi, l, h in zip(y, lo, hi):
        ax.plot([l, h], [yi, yi], color="#64748b", lw=10, solid_capstyle="butt")
        ax.plot([l], [yi], "o", color="#ef4444")
        ax.plot([h], [yi], "o", color="#10b981")
    ax.axvline(0, color="#222", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ annual battery value ($/yr) vs baseline")
    ax.set_title(f"Tornado: swing around baseline ${baseline_value:,.0f}/yr  (red=lo, green=hi)")
    ax.grid(axis="x", alpha=0.3)
    return _plot_to_b64(fig)


def plot_mc(mc: dict, baseline_value: float) -> str:
    samples = mc["samples"]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.hist(samples, bins=12, color="#3b82f6", edgecolor="white")
    ax.axvline(mc["p10"], color="#ef4444", linestyle="--", label=f"P10 ${mc['p10']:,.0f}")
    ax.axvline(mc["p50"], color="#10b981", linestyle="-", label=f"P50 ${mc['p50']:,.0f}")
    ax.axvline(mc["p90"], color="#ef4444", linestyle="--", label=f"P90 ${mc['p90']:,.0f}")
    ax.axvline(baseline_value, color="#000", linestyle=":", label=f"Base ${baseline_value:,.0f}")
    ax.set_xlabel("Annual battery value ($/yr)")
    ax.set_ylabel("Monte Carlo sample count")
    ax.set_title(f"Monte Carlo joint uncertainty (N={len(samples)})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
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
    tornado: list = None,
    mc: dict = None,
    dr_stack: dict = None,
    sizing_results: list = None,
    sizing_knee: dict = None,
    payback_rec: dict = None,
    capex_assumptions: dict = None,
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
    kind_label = {
        "capacity_emergency": "Capacity (Emergency)",
        "capacity_voluntary": "Capacity (Voluntary)",
        "economic_energy": "Economic (Energy-only)",
    }
    for dr in dr_values:
        if dr.currently_enrolled:
            status = (
                f"Already enrolled — current commit ~{dr.current_committed_kw:.0f} kW; "
                f"battery enables +{dr.incremental_committed_kw:.0f} kW via amendment"
            )
            row_style = ' style="background:#ecfdf5;"'
        else:
            status = "Not enrolled — would require new program registration"
            row_style = ''
        dr_rows += (
            f'<tr{row_style}><td>{dr.name}</td>'
            f'<td>{kind_label.get(dr.kind, dr.kind)}</td>'
            f'<td>{status}</td>'
            f'<td class="num">${_fmt_dollar(dr.net_annual_value_gross)}</td>'
            f'<td class="num"><strong>${_fmt_dollar(dr.net_annual_value)}</strong></td></tr>\n'
        )

    scenario_rows = ""
    for label, comp, res in [
        ("Perfect foresight (upper bound)", comparison_pf, pf_result),
        ("MEMOSA controls (reported)",       comparison_mpc, mpc_result),
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
    baseline_value_mpc = comparison_mpc["battery_value_annual"]
    plot_tornado_b64 = plot_tornado(tornado or [], baseline_value_mpc)
    plot_mc_b64 = plot_mc(mc or {"samples": [], "p10": 0, "p50": 0, "p90": 0},
                          baseline_value_mpc)

    # --- sizing section
    sizing_results = sizing_results or []
    sizing_knee = sizing_knee or {}
    sizing_rows = ""
    primary_label = f"{int(battery_cfg['power_kw'])}kW/{int(battery_cfg['energy_kwh'])}kWh"
    knee_label_str = sizing_knee.get("knee_label", "")
    knee_was_found = sizing_knee.get("knee_found", False)
    for r in sorted(sizing_results, key=lambda x: x.energy_kwh):
        is_rec = r.label == knee_label_str
        is_primary = r.label == primary_label
        if is_rec and knee_was_found:
            marker = "&#9733; recommended (pre-plateau)"
        elif is_rec and not knee_was_found:
            marker = "&#9733; recommended (largest tested — no plateau observed)"
        elif is_primary:
            marker = "primary"
        else:
            marker = ""
        sizing_rows += (
            f'<tr style="{"background:#ecfdf5;font-weight:600;" if is_rec else ""}">'
            f'<td>{r.label}</td>'
            f'<td class="num">{r.power_kw:,.0f}</td>'
            f'<td class="num">{r.energy_kwh:,.0f}</td>'
            f'<td class="num">${_fmt_dollar(r.annual_savings)}</td>'
            f'<td class="num">${r.marginal_savings_per_added_kwh:,.2f}</td>'
            f'<td>{marker}</td></tr>\n'
        )
    plot_sizing_b64 = plot_sizing(sizing_results, sizing_knee) if sizing_results else ""

    # Payback rows
    payback_rows = ""
    payback_rec = payback_rec or {}
    capex_assumptions = capex_assumptions or {"central": 350, "low": 250, "high": 500, "target_yr": 10}
    rec_label = payback_rec.get("recommended_label")
    for r in sorted(sizing_results, key=lambda x: x.energy_kwh):
        is_rec = r.label == rec_label
        pb_c = f"{r.marginal_payback_years:.1f}" if r.marginal_payback_years != float('inf') else "∞"
        pb_lo = f"{r.marginal_payback_years_lo:.1f}" if r.marginal_payback_years_lo != float('inf') else "∞"
        pb_hi = f"{r.marginal_payback_years_hi:.1f}" if r.marginal_payback_years_hi != float('inf') else "∞"
        ok = r.marginal_payback_years <= capex_assumptions["target_yr"]
        bg = "background:#ecfdf5;font-weight:600;" if is_rec else ("background:#fef3c7;" if not ok else "")
        payback_rows += (
            f'<tr style="{bg}">'
            f'<td>{r.label}</td>'
            f'<td class="num">{r.marginal_added_kwh:,.0f}</td>'
            f'<td class="num">${_fmt_dollar(r.marginal_annual_savings)}</td>'
            f'<td class="num">${_fmt_dollar(r.marginal_capex)}</td>'
            f'<td class="num">{pb_c}</td>'
            f'<td class="num">{pb_lo}–{pb_hi}</td>'
            f'</tr>\n'
        )

    # Knee-vs-primary side-by-side note
    knee_vs_primary_note = ""
    if knee_label_str and knee_label_str != primary_label:
        primary_r = next((r for r in sizing_results if r.label == primary_label), None)
        knee_r = next((r for r in sizing_results if r.label == knee_label_str), None)
        if primary_r and knee_r:
            delta = primary_r.annual_savings - knee_r.annual_savings
            extra_kwh = primary_r.energy_kwh - knee_r.energy_kwh
            eff = delta / extra_kwh if extra_kwh else 0.0
            knee_vs_primary_note = (
                f"<br><br><strong>Comparison vs previously-decided primary size "
                f"({primary_label}):</strong> the primary size earns "
                f"${_fmt_dollar(primary_r.annual_savings)}/yr, which is "
                f"${_fmt_dollar(delta)}/yr more than the knee but requires "
                f"{extra_kwh:,.0f} additional kWh of capacity — a marginal rate of "
                f"${eff:.2f}/added-kWh. Compare to the knee's marginal rate of "
                f"${sizing_knee['knee_marginal_per_kwh']:.2f}/added-kWh and the best block's "
                f"${sizing_knee['best_marginal_per_kwh']:.2f}/added-kWh."
            )
    elif knee_label_str == primary_label:
        if knee_was_found:
            knee_vs_primary_note = (
                f"<br><br>The recommended pre-plateau size coincides with the previously-decided "
                f"primary size ({primary_label}), which is consistent with that earlier recommendation."
            )
        else:
            knee_vs_primary_note = (
                f"<br><br>No extreme plateau was observed in the tested range. The recommendation "
                f"falls back to the largest tested size, which happens to match the previously-decided "
                f"primary size ({primary_label}). A larger block (if available) could earn additional value "
                f"— within the tested range, marginal returns are still healthy (declining gently but not collapsing)."
            )

    caveats = [
        # Data
        "Load data is hourly; ComEd bills demand on 30-minute intervals per the ratebook. This understates demand savings by ~2-3% (true 30-min peaks are slightly higher than hourly averages, and a battery could shave more of them).",
        "Solar is synthesized from a pvlib clear-sky (Ineichen) model + Midwest monthly climatological cloud fractions, fitted to 2026 Jan-Apr METERED production (A-North + B-South columns, not the design-estimate column). Hour-to-hour cloud variability is NOT captured — each day receives a smooth clear-sky-shape envelope.",
        "Winter-snow loss present in the Jan-Apr 2026 calibration data is absorbed into the fit scalar k. When k is applied to summer months (no snow), this likely UNDERSTATES summer solar by ~5-8%, which in turn slightly INFLATES the battery value (more grid import for the battery to shave).",
        "Smoothed solar output (no cloud-driven hour-to-hour variability) probably OVERSTATES peak-shave effectiveness by ~2-5% on what would in reality be broadly-cloudy afternoons.",
        "Tilt 20° / azimuth 180° are assumed (actual system geometry unknown); affects annual generation by ±3-5%.",
        # PJM / LMP
        "PJM ComEd-zone LMPs are synthesized from published typical diurnal/seasonal shapes, anchored to the Aug-Sep 2025 supply bill's observed period-average ($0.0327/kWh). No live ISO data available. Real 2025 LMP spikes (system-stress price events) are NOT captured; these would boost arbitrage value if present.",
        "PJM 5CP hours are proxied from the 5 highest weekday 14:00-19:00 site-load hours in Jun-Sep 2025, anchored to the 8/15/25 18:00 hour confirmed on the Aug-Sep supply bill. Each non-anchor hour has an estimated 20% probability of being misaligned with the true PJM system peak — captured in the Monte Carlo.",
        "NSPL (transmission tag, 1CP) proxy uses the single highest weekday 2-7pm site-load hour, also anchored to 8/15/25 18:00.",
        # Controls
        "MEMOSA controls dispatch is approximated by the 'noisy foresight' method: inputs degraded by realistic AR(1) forecast error (load MAPE 4%, solar 15%, LMP 10%), then a perfect-foresight LP solves over the noised inputs. The reported point estimate is the median of 3 random-seed realizations (stdev across seeds shown in console). True rolling re-optimization would likely recover an additional 5-10% of the PF-to-MPC gap, so the reported figure is MILDLY CONSERVATIVE.",
        "No battery cycling-cost term in the objective. The optimizer cycles opportunistically when it is economic — real deployments often add a $0.02-0.04/kWh-cycled penalty to prolong life, which would REDUCE dispatch aggressiveness and savings by ~2-5%.",
        "Monthly LP boundaries chain SOC with a 50% mid-range terminal hint but don't coordinate across months globally. Possible edge-case suboptimality if a month's peak day falls near month-boundary. Small effect (<2%).",
        # Billing / attribution
        "Battery is modeled as installed at the PCC upstream of both MDP meters, with battery flow allocated proportionally to each meter's load share for reporting billed demand. Real-world wiring may differ.",
        "ComEd DG net metering caps at 2 MW non-residential; the ~4 MW PV system is not eligible, so battery does not export to grid (export compensation = $0).",
        "Franchise and municipal tax rates back-solved from two 2025 bills; monthly-interpolated schedules approximate DSPR reconciliation drift. IL Excise Tax uses effective (not marginal) rate for kWh deltas — overstates tax savings from battery by ~$50/yr on round-trip losses.",
        "TEC pass-through formula uses NSPL × $0.00903/kW-day × days — blended effective rate derived from the Aug-Sep 2025 bill's mid-period rate change (19 days at $0.00893, 11 days at $0.00922).",
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

    mc = mc or {"samples": [], "p10": 0, "p50": 0, "p90": 0, "mean": 0, "std": 0}
    dr_stack = dr_stack or {"recommended_stack": [], "stacked_annual_value": 0}

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
        "{{solar_cf}}": f"{solar_annual_kwh / max(site_cfg.get('solar_ac_nameplate_kw', 4000) * 8760, 1) * 100:.1f}",
        "{{mpc_savings_fmt}}": _fmt_dollar(comparison_mpc["battery_value_annual"]),
        "{{pf_savings_fmt}}":  _fmt_dollar(comparison_pf["battery_value_annual"]),
        "{{rb_savings_fmt}}":  _fmt_dollar(comparison_rb["battery_value_annual"]),
        "{{seed_mean_fmt}}":   _fmt_dollar(comparison_mpc.get("battery_value_annual_seed_mean", 0)),
        "{{seed_stderr_fmt}}": _fmt_dollar(comparison_mpc.get("battery_value_annual_seed_stderr", 0)),
        "{{mc_p10_fmt}}": _fmt_dollar(mc["p10"]),
        "{{mc_p50_fmt}}": _fmt_dollar(mc["p50"]),
        "{{mc_p90_fmt}}": _fmt_dollar(mc["p90"]),
        "{{mc_mean_fmt}}": _fmt_dollar(mc["mean"]),
        "{{mc_std_fmt}}": _fmt_dollar(mc["std"]),
        "{{dr_stack_names}}": ", ".join(dr_stack["recommended_stack"]) or "none",
        "{{dr_stack_total}}": _fmt_dollar(dr_stack["stacked_annual_value"]),
        "{{dr_stack_total_fmt}}": _fmt_dollar(dr_stack["stacked_annual_value"]),
        "{{total_with_dr_fmt}}": _fmt_dollar(
            comparison_mpc["battery_value_annual"] + dr_stack["stacked_annual_value"]
        ),
        # Tier-1 (already-enrolled program expansion) headline
        "{{tier1_total_fmt}}": _fmt_dollar(dr_stack.get("tier1_total", 0)),
        "{{tier1_names_fmt}}": ", ".join(dr_stack.get("tier1_names", [])) or "(none currently enrolled)",
        "{{headline_with_tier1_fmt}}": _fmt_dollar(
            comparison_mpc["battery_value_annual"] + dr_stack.get("tier1_total", 0)
        ),
        # Tier-2 (new programs that could be added on top)
        "{{tier2_total_fmt}}": _fmt_dollar(dr_stack.get("tier2_total", 0)),
        "{{tier2_names_fmt}}": ", ".join(dr_stack.get("tier2_names", [])) or "(none recommended)",
        "{{combined_with_tier2_fmt}}": _fmt_dollar(
            comparison_mpc["battery_value_annual"]
            + dr_stack.get("tier1_total", 0)
            + dr_stack.get("tier2_total", 0)
        ),
        "{{stream_rows}}":    stream_rows,
        "{{dr_rows}}":        dr_rows,
        "{{scenario_rows}}":  scenario_rows,
        "{{tag_rows}}":       tag_rows,
        "{{plot_monthly_b64}}": plot_monthly,
        "{{plot_winter_b64}}":  plot_winter,
        "{{plot_summer_b64}}":  plot_summer,
        "{{plot_tornado_b64}}": plot_tornado_b64,
        "{{plot_mc_b64}}":      plot_mc_b64,
        "{{plot_sizing_b64}}":  plot_sizing_b64,
        "{{sizing_rows}}":      sizing_rows,
        "{{knee_label}}":       knee_label_str,
        "{{knee_savings}}":     _fmt_dollar(sizing_knee.get("knee_annual_savings", 0)),
        "{{knee_rationale}}":   sizing_knee.get("rationale", ""),
        "{{knee_vs_primary_note}}": knee_vs_primary_note,
        "{{payback_rows}}":     payback_rows,
        "{{capex_central}}":    str(capex_assumptions["central"]),
        "{{capex_low}}":        str(capex_assumptions["low"]),
        "{{capex_high}}":       str(capex_assumptions["high"]),
        "{{target_yr}}":        str(capex_assumptions["target_yr"]),
        "{{payback_rec_label}}": rec_label or "none",
        "{{payback_rec_years}}": f"{payback_rec.get('recommended_marginal_payback_yr', 0):.1f}" if payback_rec.get("recommended_marginal_payback_yr") is not None else "n/a",
        "{{payback_rationale}}": payback_rec.get("rationale", ""),
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
        "dr_recommended_stack": dr_stack,
        "plc_hours": [str(h) for h in plc_hours],
        "nspl_hour": str(nspl_hour),
        "tag_costs": tag_costs,
        "sensitivity": {
            "tornado": tornado,
            "monte_carlo": mc,
        },
        "sizing": {
            "sweep_results": [r.__dict__ for r in (sizing_results or [])],
            "knee": sizing_knee,
            "capex_based_recommendation": payback_rec,
            "capex_assumptions": capex_assumptions,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    # Hourly dispatch CSV
    mpc_df.to_csv(out_dir / "dispatch_memosa.csv")

    return html_path
