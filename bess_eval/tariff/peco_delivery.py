"""PECO Electric High Tension Service >500 kW delivery tariff.

Sources:
  - Jun 27 – Jul 29 2025 PECO bill (calibration reference)
  - Jan 27 – Feb 25 2025 PECO bill (calibration reference)

Structure (much simpler than ComEd VLL):
  Customer Charge         flat $/month
  Distribution Charges    billed_kw × $7.12/kW  (all-hours max demand)
  Distribution Credit     kWh × $-0.00060/kWh   (small kWh credit)
  EENT                    PLC_kw × $2.28-2.51/kW  (Energy Eff. & Nonbypassable Trans.)
  Sales Tax               flat 8% of delivery subtotal (PA 6% state + Philadelphia 2% city)

Where billed_kw = measured_max_kw × (0.9 / PF) if PF < 0.9, else measured_max_kw.
PF = actual site power factor over the billing period.

Note the EENT charge is based on PJM PLC kW (same tag value as used for capacity
charges on the supply bill). The EENT rate itself varies by PJM delivery year.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import pandas as pd

# Reuse the Bill/BillLine dataclasses from ComEd module to keep the attribution
# code agnostic of which delivery tariff produced the bill.
from .comed_delivery import Bill, BillLine


class PECOHighTensionDelivery:
    """Electric High Tension Service >500 kW (PECO)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.customer_charge = cfg["customer_charge_monthly"]
        self.distribution_rate_per_kw = cfg["distribution_rate_per_kw"]
        self.distribution_credit_per_kwh = cfg.get("distribution_credit_per_kwh", 0.0)
        # EENT rate varies by PJM delivery year (roughly Jun 1 change). Store as monthly schedule.
        self.eent_per_kw_monthly = cfg["eent_per_kw_monthly"]
        self.sales_tax_rate = cfg.get("sales_tax_rate", 0.08)
        self.pf_adjust_threshold = cfg.get("pf_adjustment_threshold", 0.90)
        # If not provided, treat as 1.0 (no PF penalty — conservative assumption for
        # with-battery scenarios where we don't know future PF)
        self.assumed_pf_monthly = cfg.get("power_factor_monthly", {m: 1.0 for m in range(1, 13)})
        # PECO's PLC used for EENT. Same PJM-wide PLC tag the supplier uses for capacity.
        self.eent_plc_kw = cfg.get("eent_plc_kw", None)

    def eent_rate(self, month: int) -> float:
        return self.eent_per_kw_monthly[month]

    def billed_demand_kw(self, max_kw: float, month: int) -> tuple[float, float]:
        """Apply PF adjustment if site PF at that month is < threshold.
        Returns (billed_kw, applied_pf)."""
        pf = self.assumed_pf_monthly.get(month, 1.0)
        if pf < self.pf_adjust_threshold and pf > 0:
            billed = max_kw * (self.pf_adjust_threshold / pf)
            return billed, pf
        return max_kw, pf

    def compute_bill(
        self,
        period_df: pd.DataFrame,
        meter_cols: Sequence[str],          # for PECO, a single-meter list like ["mdp1"]
        combined_col: str = "combined_kw",
        period_start: pd.Timestamp | None = None,
        period_end: pd.Timestamp | None = None,
    ) -> Bill:
        """Compute itemized PECO HT delivery bill for a billing period."""
        if period_start is None:
            period_start = period_df.index.min()
        if period_end is None:
            period_end = period_df.index.max()
        month = period_start.month

        clipped = period_df[combined_col].clip(lower=0)
        max_kw = float(clipped.max())
        total_kwh = float(clipped.sum())
        billed_kw, applied_pf = self.billed_demand_kw(max_kw, month)

        bill = Bill(period_start=period_start, period_end=period_end)
        bill.lines.append(BillLine("Customer Charge", self.customer_charge, ""))
        bill.lines.append(
            BillLine("Distribution Charges", billed_kw * self.distribution_rate_per_kw,
                     f"{billed_kw:,.2f} kW @ ${self.distribution_rate_per_kw}/kW  (PF {applied_pf:.3f})")
        )
        bill.lines.append(
            BillLine("Distribution Credit", total_kwh * self.distribution_credit_per_kwh,
                     f"{total_kwh:,.0f} kWh @ ${self.distribution_credit_per_kwh}/kWh")
        )
        eent_rate = self.eent_rate(month)
        # EENT billed on PJM PLC value (the tag). For battery scenarios, the tag is the
        # argument `eent_plc_kw` from config (baseline assumption); if the dispatch is
        # reducing load at PLC hours, the attribution layer overrides this kW separately.
        eent_kw = self.eent_plc_kw if self.eent_plc_kw is not None else billed_kw
        bill.lines.append(
            BillLine("Energy Eff & Nonbypassable Trans", eent_kw * eent_rate,
                     f"{eent_kw:,.2f} PLC @ ${eent_rate}/kW")
        )

        delivery_subtotal = sum(l.amount for l in bill.lines)
        bill.lines.append(
            BillLine("Sales Tax (Delivery)", delivery_subtotal * self.sales_tax_rate,
                     f"${delivery_subtotal:,.2f} @ {self.sales_tax_rate*100:.1f}%")
        )
        return bill

    # Satisfy the delivery-tariff interface used by attribution.annual_cost_from_dispatch.
    # ComEd's per-meter DFC lookup is not relevant for PECO (single max kW × rate) but
    # the attribution code calls this; return the delivery rate for a monthly look-up.
    def dfc_rate(self, month: int) -> float:
        """Return the distribution $/kW rate (flat; not month-varying here)."""
        return self.distribution_rate_per_kw
