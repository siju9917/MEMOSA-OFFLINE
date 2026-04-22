"""Freepoint Energy Solutions passthrough supply tariff.

Structure derived from Bedford 2025 supply bills:
  Hub Energy           = kWh_billed × hub_rate ($/kWh)
  Fixed Retail Adder   = kWh_metered × fixed_adder_rate ($/kWh)
  Basis Residual Agg   = kWh_billed × basis_residual_rate ($/kWh)
  Basis Passthrough    = kWh_billed × basis_passthrough_rate ($/kWh)   (can be negative)
  Capacity Passthrough = PLC_kW × capacity_rate_per_kw_day × days
  Transmission Passth. = NSPL_kW × transmission_rate_per_kw_day × days

'kWh_billed' = kWh_metered minus a small losses-adjustment factor; the bill applies
the hub/basis rates on kWh_billed and the fixed adder on kWh_metered. We model
kWh_billed = kWh_metered × (1 - losses_reduction) where the factor is derived from
the Jul-Aug bill (~0.14%).

Same interface shape as IndexSupply so the attribution module can be agnostic:
  - capacity_rate (per-kW-day)
  - transmission_rate_yr = $/kW-yr (we translate from $/kW-day × 365)
  - tec_rate_per_kw_day = 0.0 (Freepoint has no TEC passthrough)
  - losses_factor, fixed_adder (for kWh-proportional costs)
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd


@dataclass
class FreepointSupplyBill:
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    hub_energy: float
    fixed_adder_charge: float
    basis_residual: float
    basis_passthrough: float
    capacity_charge: float
    transmission_charge: float

    @property
    def total(self) -> float:
        return (
            self.hub_energy
            + self.fixed_adder_charge
            + self.basis_residual
            + self.basis_passthrough
            + self.capacity_charge
            + self.transmission_charge
        )


class FreepointSupply:
    """Passthrough supply contract (Bedford 2025 — Freepoint Energy Solutions)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hub_rate = cfg["hub_energy_rate_per_kwh"]
        self.fixed_adder = cfg["fixed_retail_adder_per_kwh"]
        self.basis_residual_rate = cfg["basis_residual_rate_per_kwh"]
        self.basis_passthrough_rate = cfg["basis_passthrough_rate_per_kwh"]
        # billed_kWh = metered_kWh × (1 - losses_reduction_factor).
        # From Jul-Aug bill: (800639.94 - 799509.19)/800639.94 ≈ 0.00141
        self.losses_reduction_factor = cfg.get("losses_reduction_factor", 0.00141)
        self.capacity_obligation_kw = cfg["capacity_obligation_kw"]
        self.capacity_rate = cfg["capacity_rate_per_kw_day"]
        self.nspl_kw = cfg["nspl_kw"]
        self.transmission_rate_per_kw_day = cfg["transmission_rate_per_kw_day"]
        # Interface compatibility with IndexSupply (for attribution)
        self.transmission_rate_yr = self.transmission_rate_per_kw_day * 365
        self.tec_rate_per_kw_day = 0.0
        # losses_factor × hub_rate ≈ basis-residual-net contribution
        self.losses_factor = max(
            0.0,
            (self.basis_residual_rate + self.basis_passthrough_rate) / max(self.hub_rate, 1e-6),
        )

    def compute_bill(
        self,
        period_df: pd.DataFrame,
        lmp_series: pd.Series,  # ignored — Freepoint uses flat hub_rate
        capacity_obligation_kw: float | None = None,
        nspl_kw: float | None = None,
        combined_col: str = "combined_kw",
    ) -> FreepointSupplyBill:
        clipped = period_df[combined_col].clip(lower=0)
        metered_kwh = float(clipped.sum())
        billed_kwh = metered_kwh * (1.0 - self.losses_reduction_factor)

        hub_energy = billed_kwh * self.hub_rate
        fixed_adder_charge = metered_kwh * self.fixed_adder
        basis_residual = billed_kwh * self.basis_residual_rate
        basis_passthrough = billed_kwh * self.basis_passthrough_rate

        days = (period_df.index.max() - period_df.index.min()).days + 1
        cap_kw = capacity_obligation_kw if capacity_obligation_kw is not None else self.capacity_obligation_kw
        capacity_charge = cap_kw * self.capacity_rate * days
        nspl = nspl_kw if nspl_kw is not None else self.nspl_kw
        transmission_charge = nspl * self.transmission_rate_per_kw_day * days

        return FreepointSupplyBill(
            period_start=period_df.index.min(),
            period_end=period_df.index.max(),
            hub_energy=hub_energy,
            fixed_adder_charge=fixed_adder_charge,
            basis_residual=basis_residual,
            basis_passthrough=basis_passthrough,
            capacity_charge=capacity_charge,
            transmission_charge=transmission_charge,
        )

    def annual_tag_cost(self, plc_kw: float, nspl_kw: float) -> dict:
        cap = plc_kw * self.capacity_rate * 365
        tx = nspl_kw * self.transmission_rate_per_kw_day * 365
        return {
            "capacity_annual": cap,
            "transmission_annual": tx,
            "tec_annual": 0.0,
            "total_tag_annual": cap + tx,
            "capacity_per_kw_yr": self.capacity_rate * 365,
            "transmission_per_kw_yr": self.transmission_rate_per_kw_day * 365,
        }
