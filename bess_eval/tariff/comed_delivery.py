"""ComEd Very Large Load (VLL) Secondary Voltage delivery-service tariff engine.

Sources:
  - A_Guide_to_the_Retail_Customer_s_Billed_Delivery_Service_Charges.pdf (p.2, VLL row)
  - 2026_Ratebook.pdf  line 9026 (Retail Peak Period = Mon-Fri 9AM-10PM CPT ex-holidays)
  - 2026_Ratebook.pdf  line 9315 (VLL = 30-min demand 1,001-10,000 kW)
  - Feb 2025 ComEd bill (calibration reference): DFC $13.49/kW, 1,866.48 billed kW
  - Aug 2025 ComEd bill (calibration reference): DFC $13.43/kW, 2,628.32 billed kW
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence
import calendar

import numpy as np
import pandas as pd


def us_federal_holidays(year: int) -> set[date]:
    """ComEd-observed federal holidays for Retail Peak Period exclusion.
    Uses standard US federal holiday rules."""
    h: set[date] = set()
    h.add(date(year, 1, 1))  # New Year's Day
    # Memorial Day: last Monday of May
    d = date(year, 5, 31)
    while d.weekday() != 0:
        d = d.replace(day=d.day - 1)
    h.add(d)
    h.add(date(year, 7, 4))  # Independence Day
    # Labor Day: first Monday of September
    d = date(year, 9, 1)
    while d.weekday() != 0:
        d = date(year, 9, d.day + 1)
    h.add(d)
    # Thanksgiving: 4th Thursday of November
    d = date(year, 11, 1)
    while d.weekday() != 3:
        d = date(year, 11, d.day + 1)
    h.add(date(year, 11, d.day + 21))
    h.add(date(year, 12, 25))  # Christmas
    return h


def is_on_peak_hour(ts: pd.Timestamp, holidays: set[date] | None = None) -> bool:
    """ComEd Retail Peak Period: Mon-Fri 9AM <= hour < 10PM CPT, excluding federal holidays."""
    if holidays is not None and ts.date() in holidays:
        return False
    if ts.weekday() >= 5:
        return False
    return 9 <= ts.hour < 22


def on_peak_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Vectorized on-peak mask."""
    years = set(pd.DatetimeIndex(index).year.unique())
    holidays = set()
    for y in years:
        holidays.update(us_federal_holidays(int(y)))
    is_hol = pd.Series(index.date, index=index).isin(holidays)
    is_wkday = index.weekday < 5
    is_peak_hr = (index.hour >= 9) & (index.hour < 22)
    return pd.Series((is_wkday & is_peak_hr) & (~is_hol.values), index=index)


@dataclass
class BillLine:
    name: str
    amount: float
    detail: str = ""


@dataclass
class Bill:
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    lines: list[BillLine] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(l.amount for l in self.lines)

    def subtotal_by_section(self, names: Sequence[str]) -> float:
        return sum(l.amount for l in self.lines if l.name in names)

    def as_dict(self) -> dict:
        return {
            "period": f"{self.period_start.date()} to {self.period_end.date()}",
            "total": self.total,
            "lines": [(l.name, l.amount, l.detail) for l in self.lines],
        }


class ComEdVLLDelivery:
    """Very Large Load Secondary Voltage delivery tariff."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.customer_charge = cfg["customer_charge_monthly"]
        self.metering_charge = cfg["metering_charge_monthly"]
        self.meter_lease = cfg["meter_lease_monthly"]
        self.dfc_per_kw_monthly = cfg["dfc_per_kw_monthly"]
        self.iedt = cfg["iedt_per_kwh"]
        self.riders = cfg["riders_per_kwh"]
        self.franchise_monthly = cfg["franchise_effective_rate_monthly"]
        self.franchise_basis = cfg.get("franchise_basis", "dfc_iedt")
        self.il_excise_eff = cfg["il_excise_effective_per_kwh"]
        self.municipal_per_kwh = cfg["municipal_tax_per_kwh"]

    def dfc_rate(self, month: int) -> float:
        return self.dfc_per_kw_monthly[month]

    def rider_rate_for_month(self, name: str, month: int) -> float:
        v = self.riders[name]
        if isinstance(v, dict):
            return v[month]
        return v

    def _rider_lookup(self, key: str, month: int) -> float | None:
        """Handle both flat and monthly rider configs transparently."""
        if key in self.riders:
            return self.rider_rate_for_month(key, month)
        alt = key + "_monthly"
        if alt in self.riders:
            return self.riders[alt][month]
        return None

    def compute_billed_demand_kw(
        self, period_df: pd.DataFrame, meter_cols: Sequence[str]
    ) -> tuple[float, dict[str, float]]:
        """Billed DFC demand = Σ across meters of (max on-peak 30-min demand).
        Our data is hourly. We use the hourly peak as a proxy for the 30-min peak;
        this slightly understates (-2-3%) vs actual billing demand (documented limitation).
        """
        mask = on_peak_mask(period_df.index)
        on_peak_df = period_df.loc[mask]
        per_meter = {}
        total = 0.0
        for c in meter_cols:
            pk = float(on_peak_df[c].max())
            per_meter[c] = pk
            total += pk
        return total, per_meter

    def compute_bill(
        self,
        period_df: pd.DataFrame,
        meter_cols: Sequence[str],
        combined_col: str = "combined_kw",
        period_start: pd.Timestamp | None = None,
        period_end: pd.Timestamp | None = None,
    ) -> Bill:
        """Compute itemized ComEd VLL delivery bill for a billing period.

        period_df: hourly DataFrame of grid-imported kW at each meter and combined.
                   Grid-imported kW = site load - solar - battery_discharge + battery_charge.
                   Negative values (export) are clipped to 0 for demand calcs (ComEd does not
                   net export against demand).
        """
        if period_start is None:
            period_start = period_df.index.min()
        if period_end is None:
            period_end = period_df.index.max()
        month = period_start.month

        clipped = period_df.copy()
        for c in list(meter_cols) + [combined_col]:
            if c in clipped.columns:
                clipped[c] = clipped[c].clip(lower=0.0)

        total_kwh = float(clipped[combined_col].sum())  # hourly kW = kWh/hour
        billed_demand_kw, per_meter = self.compute_billed_demand_kw(clipped, meter_cols)

        bill = Bill(period_start=period_start, period_end=period_end)

        bill.lines.append(BillLine("Customer Charge", self.customer_charge, ""))
        bill.lines.append(BillLine("Standard Metering Service Charge", self.metering_charge, ""))
        dfc_rate = self.dfc_rate(month)
        bill.lines.append(
            BillLine("Distribution Facility Charge", billed_demand_kw * dfc_rate,
                     f"{billed_demand_kw:,.2f} kW @ ${dfc_rate}/kW")
        )
        bill.lines.append(
            BillLine("IL Electricity Distribution Charge", total_kwh * self.iedt,
                     f"{total_kwh:,.0f} kWh @ ${self.iedt}/kWh")
        )
        bill.lines.append(BillLine("Meter Lease", self.meter_lease, ""))
        delivery_subtotal = sum(l.amount for l in bill.lines)

        rider_order = [
            ("renewable_portfolio_standard",   "Renewable Portfolio Standard"),
            ("environmental_cost_recovery",    "Environmental Cost Recovery Adj"),
            ("coal_to_solar_storage_fund",     "Coal to Solar and Energy Storage Fund"),
            ("zero_emission_standard",         "Zero Emission Standard"),
            ("carbon_free_energy_adj",         "Carbon-Free Energy Resource Adj"),
            ("energy_efficiency_programs",     "Energy Efficiency Programs"),
            ("energy_transition_assistance",   "Energy Transition Assistance"),
        ]
        for key, label in rider_order:
            rate = self._rider_lookup(key, month)
            if rate is None:
                continue
            bill.lines.append(
                BillLine(label, total_kwh * rate, f"{total_kwh:,.0f} kWh @ ${rate}/kWh")
            )

        dfc_iedt_basis = billed_demand_kw * dfc_rate + total_kwh * self.iedt
        franchise_rate = self.franchise_monthly[month]
        franchise_basis_val = dfc_iedt_basis if self.franchise_basis == "dfc_iedt" else delivery_subtotal
        bill.lines.append(
            BillLine("Franchise Cost", franchise_basis_val * franchise_rate,
                     f"${franchise_basis_val:,.2f} @ {franchise_rate*100:.4f}%")
        )
        bill.lines.append(
            BillLine("State Tax (IL Excise)", total_kwh * self.il_excise_eff,
                     f"{total_kwh:,.0f} kWh @ ${self.il_excise_eff}/kWh (effective)")
        )
        bill.lines.append(
            BillLine("Municipal Tax", total_kwh * self.municipal_per_kwh,
                     f"{total_kwh:,.0f} kWh @ ${self.municipal_per_kwh}/kWh")
        )
        return bill
