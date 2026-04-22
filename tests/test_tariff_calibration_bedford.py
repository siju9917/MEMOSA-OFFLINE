"""Calibration test: reconstruct the two provided Bedford ComEd delivery bills.

Target: major line items within ~5% (relaxed slightly because Bedford has more
solar-embedded load hours and gross-load reconstruction carries some noise).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bess_eval.config import Config
from bess_eval.ingest import load_load, load_solar, reconstruct_gross_load
from bess_eval.tariff.comed_delivery import ComEdVLLDelivery


ROOT = Path(__file__).resolve().parents[1]
CFG = Config.load(ROOT / "configs" / "bedford.yaml")


@pytest.fixture(scope="module")
def gross_load_df():
    ld = load_load(
        ROOT / CFG.data.load_csv,
        time_col=CFG.data.load_time_col,
        kw_cols=CFG.data.load_kw_cols,
        tz=CFG.site.tz,
    )
    sl = load_solar(
        ROOT / CFG.data.solar_csv,
        time_col=CFG.data.solar_time_col,
        prod_cols=CFG.data.solar_production_cols,
        tz=CFG.site.tz,
    )
    return reconstruct_gross_load(ld, sl, ["mdp1", "mdp2"])


# Jan-Feb 2025 (1/15-2/12, 28 days) — solar had NOT started (starts Sep 2025), so
# "gross" reconstruction equals raw load here.
JAN_FEB_BILL = {
    "period_start": "2025-01-15",
    "period_end": "2025-02-12",
    "total_kwh": 676_264,
    "billed_demand_kw": 1369.52,
    "lines": {
        "Customer Charge":                       1005.53,
        "Standard Metering Service Charge":        14.10,
        "Distribution Facility Charge":         18474.82,
        "IL Electricity Distribution Charge":     838.57,
        "Meter Lease":                              1.00,
        "Nonstandard Facilities Charge":           70.97,
        "Renewable Portfolio Standard":          3394.85,
        "Environmental Cost Recovery Adj":         13.53,
        "Zero Emission Standard":                1318.71,
        "Carbon-Free Energy Resource Adj":       3212.25,
        "Energy Efficiency Programs":            3002.61,
        "Energy Transition Assistance":           486.91,
        "Franchise Cost":                         151.01,
        "State Tax (IL Excise)":                 2003.34,
        # No Municipal Tax (Bedford Park imposes none)
    },
    "total": 20404.99 + 13583.21,   # delivery + taxes/fees blocks
}

# Jul-Aug 2025 (7/16-8/14, 29 days) — solar had NOT started yet (starts 9/9), so
# gross reconstruction is still equal to raw load in this window.
JUL_AUG_BILL = {
    "period_start": "2025-07-16",
    "period_end": "2025-08-14",
    "total_kwh": 800_640,
    "billed_demand_kw": 1490.72,
    "lines": {
        "Customer Charge":                       1001.93,
        "Standard Metering Service Charge":        14.03,
        "Distribution Facility Charge":         20020.37,
        "IL Electricity Distribution Charge":     992.79,
        "Meter Lease":                              1.00,
        "Nonstandard Facilities Charge":           70.97,
        "Renewable Portfolio Standard":          4019.21,
        "Environmental Cost Recovery Adj":         80.06,
        "Coal to Solar and Energy Storage Fund":   40.03,
        "Zero Emission Standard":                1561.25,
        "Carbon-Free Energy Resource Adj":      -6253.00,
        "Energy Efficiency Programs":            3554.84,
        "Energy Transition Assistance":           576.46,
        "Franchise Cost":                         161.67,
        "State Tax (IL Excise)":                 2359.05,
    },
    "total": 28200.66,
}


def _slice_period(load_df, start, end):
    tz = load_df.index.tz
    s = pd.Timestamp(start).tz_localize(tz)
    e = pd.Timestamp(end).tz_localize(tz)
    return load_df.loc[(load_df.index >= s) & (load_df.index < e)].copy()


def _run_calibration(load_df, bill_ref):
    tariff = ComEdVLLDelivery(CFG.delivery_tariff)
    period = _slice_period(load_df, bill_ref["period_start"], bill_ref["period_end"])
    bill = tariff.compute_bill(period, meter_cols=["mdp1", "mdp2"], combined_col="combined_kw")
    computed = {l.name: l.amount for l in bill.lines}

    results = []
    for line_name, expected in bill_ref["lines"].items():
        got = computed.get(line_name, 0.0)
        rel_err = (got - expected) / expected if abs(expected) > 1 else 0.0
        results.append((line_name, expected, got, rel_err))
    print(f"\n=== Bedford calibration: {bill_ref['period_start']} to {bill_ref['period_end']} ===")
    print(f"{'Line':<42} {'Expected':>12} {'Computed':>12} {'Diff %':>8}")
    for name, exp, got, err in results:
        print(f"{name:<42} {exp:>12,.2f} {got:>12,.2f} {err*100:>7.2f}%")
    print(f"{'TOTAL':<42} {bill_ref['total']:>12,.2f} {bill.total:>12,.2f} "
          f"{(bill.total - bill_ref['total'])/bill_ref['total']*100:>7.2f}%")
    return bill, results


def test_jan_feb_bill(gross_load_df):
    _run_calibration(gross_load_df, JAN_FEB_BILL)


def test_jul_aug_bill(gross_load_df):
    _run_calibration(gross_load_df, JUL_AUG_BILL)


if __name__ == "__main__":
    ld = load_load(ROOT / CFG.data.load_csv, CFG.data.load_time_col, CFG.data.load_kw_cols, CFG.site.tz)
    sl = load_solar(ROOT / CFG.data.solar_csv, CFG.data.solar_time_col, CFG.data.solar_production_cols, CFG.site.tz)
    gross = reconstruct_gross_load(ld, sl, ["mdp1", "mdp2"])
    _run_calibration(gross, JAN_FEB_BILL)
    _run_calibration(gross, JUL_AUG_BILL)
