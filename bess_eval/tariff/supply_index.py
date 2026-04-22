"""Index/LMP supply tariff (primary case, 2025/2026 forward-looking).

Structure derived from Aug-Sep 2025 supply bill:
  Index Energy Charge = kWh × LMP
  Distribution Losses Charge = losses_kWh × LMP
  Fixed Retail Adder = kWh × $0.01093
  Capacity = PLC_kW × $0.27043/kW-day × days
  Transmission = NSPL_kW × $46.025/kW-yr × days/365
  TEC Pass Through = NSPL_kW × ($0.00893 + $0.00922)
"""
from __future__ import annotations
import calendar
from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class IndexSupplyBill:
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    energy_charge: float
    losses_charge: float
    fixed_adder: float
    capacity_charge: float
    transmission_charge: float
    tec_pass_through: float

    @property
    def total(self) -> float:
        return (
            self.energy_charge
            + self.losses_charge
            + self.fixed_adder
            + self.capacity_charge
            + self.transmission_charge
            + self.tec_pass_through
        )

    def as_lines(self) -> list[tuple[str, float, str]]:
        return [
            ("Index Energy Charge",    self.energy_charge, ""),
            ("Distribution Losses",    self.losses_charge, ""),
            ("Fixed Retail Adder",     self.fixed_adder, ""),
            ("Capacity (PLC)",         self.capacity_charge, ""),
            ("Transmission (NSPL)",    self.transmission_charge, ""),
            ("TEC Pass Through",       self.tec_pass_through, ""),
        ]


class IndexSupply:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.losses_factor = cfg["distribution_losses_factor"]
        self.fixed_adder = cfg["fixed_retail_adder_per_kwh"]
        self.capacity_obligation_kw = cfg["capacity_obligation_kw"]
        self.capacity_rate = cfg["capacity_rate_per_kw_day"]
        self.nspl_kw = cfg["nspl_kw"]
        self.transmission_rate_yr = cfg["transmission_rate_per_kw_yr"]
        self.tec_rates = cfg["tec_pass_through_rates"]

    def compute_bill(
        self,
        period_df: pd.DataFrame,
        lmp_series: pd.Series,
        capacity_obligation_kw: float | None = None,
        nspl_kw: float | None = None,
        combined_col: str = "combined_kw",
    ) -> IndexSupplyBill:
        """Compute an index-supply bill for a billing period.

        period_df: hourly grid-imported kW (clipped to non-negative).
        lmp_series: hourly PJM ComEd-zone LMP in $/kWh, aligned to period_df.
        capacity_obligation_kw / nspl_kw: override if running a counterfactual
            with a battery-reduced tag.
        """
        clipped = period_df[combined_col].clip(lower=0)
        total_kwh = float(clipped.sum())
        lmp = lmp_series.reindex(clipped.index).ffill().bfill()

        energy_charge = float((clipped * lmp).sum())
        losses_kwh = total_kwh * self.losses_factor
        losses_charge = losses_kwh * float(lmp.mean())

        fixed_adder = total_kwh * self.fixed_adder

        days = (period_df.index.max() - period_df.index.min()).days + 1
        cap_kw = capacity_obligation_kw if capacity_obligation_kw is not None else self.capacity_obligation_kw
        capacity_charge = cap_kw * self.capacity_rate * days

        nspl = nspl_kw if nspl_kw is not None else self.nspl_kw
        transmission_charge = nspl * self.transmission_rate_yr * days / 365.0

        tec = nspl * sum(self.tec_rates) * (days / 30.0)

        return IndexSupplyBill(
            period_start=period_df.index.min(),
            period_end=period_df.index.max(),
            energy_charge=energy_charge,
            losses_charge=losses_charge,
            fixed_adder=fixed_adder,
            capacity_charge=capacity_charge,
            transmission_charge=transmission_charge,
            tec_pass_through=tec,
        )

    def annual_tag_cost(self, plc_kw: float, nspl_kw: float) -> dict:
        """Annual $ cost of the PLC + NSPL tags (battery target for reduction)."""
        cap = plc_kw * self.capacity_rate * 365
        tx = nspl_kw * self.transmission_rate_yr
        tec = nspl_kw * sum(self.tec_rates) * 12
        return {
            "capacity_annual": cap,
            "transmission_annual": tx,
            "tec_annual": tec,
            "total_tag_annual": cap + tx + tec,
            "capacity_per_kw_yr": self.capacity_rate * 365,
            "transmission_per_kw_yr": self.transmission_rate_yr,
        }
