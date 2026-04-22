"""Calibration test: reconstruct the two provided PECO delivery bills."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bess_eval.config import Config
from bess_eval.ingest import load_load
from bess_eval.tariff.peco_delivery import PECOHighTensionDelivery


ROOT = Path(__file__).resolve().parents[1]
CFG = Config.load(ROOT / "configs" / "philadelphia_nosolar.yaml")


@pytest.fixture(scope="module")
def load_df():
    return load_load(
        ROOT / CFG.data.load_csv,
        time_col=CFG.data.load_time_col,
        kw_cols=CFG.data.load_kw_cols,
        tz=CFG.site.tz,
    )


# Jan 27 – Feb 25, 2025 (29 days)
JAN_FEB_BILL = {
    "period_start": "2025-01-27",
    "period_end": "2025-02-25",
    "total_kwh": 209_392,
    "measured_peak_kw": 475.20,
    "pf": 0.923,
    "billed_demand_kw": 475.0,      # PF > 0.9 → no adjustment
    "lines": {
        "Customer Charge":                    383.12,
        "Distribution Charges":              3382.00,     # 475 × 7.12
        "Distribution Credit":               -125.64,     # 209392 × -0.00060
        "Energy Eff & Nonbypassable Trans":   558.60,     # 245 PLC × 2.28
        "Sales Tax (Delivery)":              335.84,
    },
    # Delivery + sales tax
    "total": 4533.92,
    # NOTE: Feb bill used PLC=245 (prior PJM delivery year 2024/25).
    "eent_plc_override": 245,
}

# Jun 27 – Jul 29, 2025 (32 days)
JUN_JUL_BILL = {
    "period_start": "2025-06-27",
    "period_end": "2025-07-29",
    "total_kwh": 390_920,
    "measured_peak_kw": 702.0,
    "pf": 0.875,
    "billed_demand_kw": 722.0,      # 702 × 0.9/0.875 = 722.06
    "lines": {
        "Customer Charge":                    383.13,
        "Distribution Charges":              5140.64,     # 722 × 7.12
        "Distribution Credit":               -234.55,     # 390920 × -0.00060
        "Energy Eff & Nonbypassable Trans":   544.67,     # 217 PLC × 2.51
        "Sales Tax (Delivery)":              466.72,
    },
    "total": 6300.61,
    "eent_plc_override": 217,
}


def _slice_period(load_df, start, end):
    tz = load_df.index.tz
    s = pd.Timestamp(start).tz_localize(tz)
    e = pd.Timestamp(end).tz_localize(tz)
    return load_df.loc[(load_df.index >= s) & (load_df.index < e)].copy()


def _run_calibration(load_df, bill_ref):
    cfg = dict(CFG.delivery_tariff)
    cfg["eent_plc_kw"] = bill_ref["eent_plc_override"]
    tariff = PECOHighTensionDelivery(cfg)
    period = _slice_period(load_df, bill_ref["period_start"], bill_ref["period_end"])
    bill = tariff.compute_bill(period, meter_cols=["mdp1"], combined_col="combined_kw")
    computed = {l.name: l.amount for l in bill.lines}

    results = []
    for line_name, expected in bill_ref["lines"].items():
        got = computed.get(line_name, 0.0)
        rel_err = (got - expected) / expected if abs(expected) > 1 else 0.0
        results.append((line_name, expected, got, rel_err))
    print(f"\n=== Philadelphia calibration: {bill_ref['period_start']} to {bill_ref['period_end']} ===")
    print(f"  Measured peak kW: {bill_ref['measured_peak_kw']:.2f}  (bill)")
    print(f"  Site PF at bill: {bill_ref['pf']:.3f}  → billed kW {bill_ref['billed_demand_kw']:.2f}")
    print(f"{'Line':<42} {'Expected':>12} {'Computed':>12} {'Diff %':>8}")
    for name, exp, got, err in results:
        print(f"{name:<42} {exp:>12,.2f} {got:>12,.2f} {err*100:>7.2f}%")
    print(f"{'TOTAL':<42} {bill_ref['total']:>12,.2f} {bill.total:>12,.2f} "
          f"{(bill.total - bill_ref['total'])/bill_ref['total']*100:>7.2f}%")
    return bill


def test_jan_feb_bill(load_df):
    _run_calibration(load_df, JAN_FEB_BILL)


def test_jun_jul_bill(load_df):
    _run_calibration(load_df, JUN_JUL_BILL)


if __name__ == "__main__":
    ld = load_load(
        ROOT / CFG.data.load_csv, CFG.data.load_time_col, CFG.data.load_kw_cols, CFG.site.tz,
    )
    _run_calibration(ld, JAN_FEB_BILL)
    _run_calibration(ld, JUN_JUL_BILL)
