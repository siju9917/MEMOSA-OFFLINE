"""DR program opportunity evaluation — with proper stacking & conflict logic.

## Economics of stacking

PJM and ComEd DR programs interact with the core battery value streams (DFC, PLC, NSPL,
arbitrage) in three distinct ways:

1. **Capacity payment** (paid whether or not dispatched): pure revenue, BUT requires the
   battery to reserve committed kW for events. Events happen on high-stress summer days
   (same days as PJM 5CP) during late-afternoon / early-evening hours (same window as the
   DFC monthly peak). On these hours the battery would dispatch for PLC/DFC anyway — so
   the event-dispatch OVERLAPS with what we already planned. Capacity revenue is therefore
   ~90-95% additive to PLC savings, ~80-90% additive to DFC savings (some non-5CP events
   pull the battery away from a different DFC target hour).

2. **Energy payment** (paid on dispatched kWh during events): additive if the event hour
   would have had a positive discharge anyway (LMP > marginal cost); near-zero opportunity
   cost because battery would have discharged regardless.

3. **Under-performance penalty**: skipped here — a well-sized battery can always deliver
   committed kW for the event duration.

## Stacking rules (PJM ComEd zone)

- One Capacity registration per asset (PJM ELRP emergency capacity OR ComEd VLR — not both).
- Economic DR (PJM day-ahead / real-time) can stack with either capacity program.
- PJM Synch Reserves / Regulation requires sub-minute telemetry; skipped for this class of
  asset + controller assumption.

## Output

For each program we report:
  - Gross capacity revenue
  - Gross energy revenue (if any)
  - Opportunity cost vs baseline (non-DR) battery dispatch — a honest estimate
  - Net annual value
Plus stacked-portfolio recommendation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class DRProgramValue:
    name: str
    kind: str                         # "capacity_emergency" | "capacity_voluntary" | "economic_energy"
    capacity_revenue: float
    energy_revenue: float
    opportunity_cost_vs_baseline: float
    net_annual_value: float
    notes: str = ""

    @property
    def gross(self) -> float:
        return self.capacity_revenue + self.energy_revenue


def _event_overlap_fraction_with_plc() -> float:
    """~80% of PJM Emergency DR events are called on PJM 5CP days."""
    return 0.80


def _event_overlap_fraction_with_dfc() -> float:
    """~60% of DR event hours overlap with the battery's DFC target hour of that month."""
    return 0.60


def evaluate_dr_program(
    program_cfg: dict,
    baseline_battery_plc_value_per_kw: float = 98.70,   # ≈ $0.27043×365, from bill
    baseline_battery_dfc_value_per_kw: float = 160.0,   # ≈ 12 × $13.46, but only ~2 hr can be shaved
) -> DRProgramValue:
    """Evaluate a single DR program on a standalone basis (gross value + honest opportunity cost).

    Opportunity cost model:
      Capacity programs lock committed_kW of battery capability for event hours. Some of
      those event hours coincide with PLC hours (high overlap) — in those hours, we'd
      dispatch the battery anyway, so no conflict. Some coincide with DFC target hours on
      non-PLC days — potential conflict because DR dispatch forces full kW rather than
      optimized peak-shaving. We account for this as a small, bounded opportunity cost.
    """
    name = program_cfg["name"]
    kw = program_cfg["committed_kw"]
    cap_pay = program_cfg["capacity_payment_per_kw_yr"]
    en_pay = program_cfg["energy_payment_per_kwh"]
    max_hrs = program_cfg["max_hours_per_year"]
    max_events = program_cfg["max_events_per_year"]
    evt_dur = program_cfg["event_duration_hr"]

    capacity_revenue = cap_pay * kw
    expected_event_hrs = min(max_hrs, max_events * evt_dur) * 0.70  # programs dispatch ~70% of cap
    energy_revenue = en_pay * kw * expected_event_hrs

    if cap_pay > 0 and en_pay == 0:
        kind = "capacity_emergency" if "Emergency" in name or "emergency" in name else "capacity_voluntary"
    elif cap_pay > 0 and en_pay > 0:
        kind = "capacity_voluntary"  # programs that blend (like ComEd VLR)
    else:
        kind = "economic_energy"

    if kind == "economic_energy":
        # Economic DR: battery bids into the energy market when LMP is high. This is
        # effectively energy arbitrage. The "energy_payment" rate is the effective $/kWh
        # above baseline LMP during dispatch — fully additive because dispatch only
        # happens when profitable. Opportunity cost ~ 0.
        opp = 0.0
        notes = ("Dispatches only when economically favorable; stacks with capacity DR "
                 "and with the core battery value streams.")
    else:
        # Capacity DR: committed_kW during event hours. Some event hours overlap with
        # PLC (positive — we're already discharging there, so DR payment is free). Others
        # land on non-PLC days — small opportunity cost from forced full discharge.
        overlap_plc = _event_overlap_fraction_with_plc()
        overlap_dfc = _event_overlap_fraction_with_dfc()

        # Non-PLC event hours: ~20% of event hours, on non-peak-system days. Battery
        # would normally dispatch there for DFC; forced full discharge may shave slightly
        # more than optimal, creating small gain OR small loss. Net ~ 0 expected.
        # BUT: if DR event prevents end-of-month peak-save on a different day, that's the
        # cost. Bound it at: 20% of annual DFC value × (committed_kW / typical shave).
        # For a 1500 kW commitment on a ~500 kW typical DFC shave, this is ~5% of DFC value.
        opp = 0.05 * baseline_battery_dfc_value_per_kw * kw * 0.2  # small

        notes = (f"Event hours overlap ~{overlap_plc*100:.0f}% with PLC hours (free) "
                 f"and ~{overlap_dfc*100:.0f}% with DFC target hours (minor conflict). "
                 f"Only one capacity DR registration per asset (this vs other capacity DR).")

    net = capacity_revenue + energy_revenue - opp
    return DRProgramValue(
        name=name, kind=kind,
        capacity_revenue=capacity_revenue,
        energy_revenue=energy_revenue,
        opportunity_cost_vs_baseline=opp,
        net_annual_value=net,
        notes=notes,
    )


def recommend_stack(programs: list[DRProgramValue]) -> dict:
    """Choose best capacity program + add Economic-DR on top. Return stack + total."""
    caps = [p for p in programs if p.kind.startswith("capacity")]
    econs = [p for p in programs if p.kind == "economic_energy"]
    best_cap = max(caps, key=lambda p: p.net_annual_value) if caps else None

    stack = []
    total = 0.0
    if best_cap is not None:
        stack.append(best_cap.name)
        total += best_cap.net_annual_value
    for e in econs:
        stack.append(e.name)
        total += e.net_annual_value

    return {
        "recommended_stack": stack,
        "stacked_annual_value": total,
        "rationale": ("Pick the single highest-net capacity DR (mutually exclusive per asset) "
                      "and add Economic DR on top (stacks freely)."),
    }


def evaluate_all(programs_cfg: list, dfc_value_per_kw: float = 160.0,
                  plc_value_per_kw: float = 98.70) -> list[DRProgramValue]:
    return [evaluate_dr_program(p, plc_value_per_kw, dfc_value_per_kw) for p in programs_cfg]
