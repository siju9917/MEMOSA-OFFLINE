"""DR program opportunity evaluation.

Treats DR separately from the main dispatch because DR revenue is largely independent:
the battery commits to being available for N events/hours per year at a promised kW.

Value model per program:
  Annual value = capacity_payment × committed_kW
               + energy_payment × expected_energy_delivered
               - expected_opportunity_cost_of_lost_dispatch_during_events
               - expected_penalty_for_underperformance

We approximate "expected opportunity cost" as the avg $/kWh discharge value of the
non-DR streams at DR-event hours (hot weekday afternoons). DR usually stacks with
DFC & PLC since events happen in the same hours we'd want to discharge anyway;
opportunity cost is typically small (10-30% of energy_payment).

Only ONE DR program can typically be enrolled simultaneously for a given kW capacity.
We report each program's standalone annual value and recommend the winner.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DRProgramValue:
    name: str
    capacity_revenue: float
    energy_revenue: float
    opportunity_cost: float
    net_annual_value: float


def evaluate_dr_program(
    program_cfg: dict,
    baseline_battery_dfc_value_per_kw: float = 140.0,  # typical $/kW-yr opportunity if we reserve SOC
) -> DRProgramValue:
    name = program_cfg["name"]
    kw = program_cfg["committed_kw"]
    cap_pay = program_cfg["capacity_payment_per_kw_yr"]
    en_pay = program_cfg["energy_payment_per_kwh"]
    max_hrs = program_cfg["max_hours_per_year"]
    evt_dur = program_cfg["event_duration_hr"]

    capacity_rev = cap_pay * kw
    # Expected energy delivered = committed_kW × expected annual event hours
    # Assume program dispatches at ~70% of max hours on average
    expected_event_hrs = min(max_hrs, program_cfg["max_events_per_year"] * evt_dur) * 0.70
    energy_rev = en_pay * kw * expected_event_hrs

    # Opportunity cost: DR event hours conflict with DFC & PLC windows. The battery would
    # have discharged anyway at those times, but the DR commitment locks kW to full output
    # (can't partial-dispatch for DFC peak-shaping). Estimate 20% of capacity value lost.
    opp = baseline_battery_dfc_value_per_kw * kw * 0.20 * (expected_event_hrs / max(max_hrs, 1))

    net = capacity_rev + energy_rev - opp
    return DRProgramValue(name=name, capacity_revenue=capacity_rev,
                          energy_revenue=energy_rev, opportunity_cost=opp,
                          net_annual_value=net)


def evaluate_all(programs_cfg: list, dfc_value_per_kw: float = 140.0) -> list[DRProgramValue]:
    return [evaluate_dr_program(p, dfc_value_per_kw) for p in programs_cfg]
