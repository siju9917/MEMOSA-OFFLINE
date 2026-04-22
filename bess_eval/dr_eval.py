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
    capacity_revenue_gross: float     # at full battery committed_kw
    energy_revenue_gross: float
    opportunity_cost_vs_baseline: float
    net_annual_value_gross: float     # value if newly enrolled at full battery committed kW
    # ── New: incremental value over current enrollment ─────────────────────────────
    currently_enrolled: bool          # is the site already in this program?
    current_committed_kw: float       # current commitment (manual/curtailment-based)
    incremental_committed_kw: float   # added kW from battery (committed_kw − current_committed_kw)
    net_annual_value_incremental: float   # incremental net value (battery-enabled expansion)
    notes: str = ""

    @property
    def gross(self) -> float:
        return self.capacity_revenue_gross + self.energy_revenue_gross

    # Back-compat alias for older code paths that referenced these names
    @property
    def capacity_revenue(self) -> float:
        return self.capacity_revenue_gross

    @property
    def energy_revenue(self) -> float:
        return self.energy_revenue_gross

    @property
    def net_annual_value(self) -> float:
        """The HEADLINE net value used in stacking and reporting — incremental if
        already enrolled (since gross would double-count current revenue), gross otherwise."""
        return self.net_annual_value_incremental if self.currently_enrolled else self.net_annual_value_gross


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
    """Evaluate a single DR program with both gross and incremental value.

    Gross value = revenue if NEWLY enrolling the battery at full committed kW.
    Incremental value = revenue from EXPANDING an existing enrollment by adding the
    battery's capacity (= incremental_kw × capacity_rate). For programs the site is
    NOT currently enrolled in, gross == incremental.

    The headline `net_annual_value` (used in stacking and reporting) returns the
    incremental value if the site is already enrolled, since claiming the gross
    value would double-count revenue the site is already collecting.
    """
    name = program_cfg["name"]
    kw = program_cfg["committed_kw"]
    cap_pay = program_cfg["capacity_payment_per_kw_yr"]
    en_pay = program_cfg["energy_payment_per_kwh"]
    max_hrs = program_cfg["max_hours_per_year"]
    max_events = program_cfg["max_events_per_year"]
    evt_dur = program_cfg["event_duration_hr"]

    currently_enrolled = bool(program_cfg.get("currently_enrolled", False))
    current_committed_kw = float(program_cfg.get("current_committed_kw", 0.0))
    incremental_kw = max(0.0, kw - current_committed_kw) if currently_enrolled else kw

    # Gross at full kW
    capacity_revenue_gross = cap_pay * kw
    expected_event_hrs = min(max_hrs, max_events * evt_dur) * 0.70  # programs dispatch ~70% of cap
    energy_revenue_gross = en_pay * kw * expected_event_hrs

    if cap_pay > 0 and en_pay == 0:
        kind = "capacity_emergency" if "Emergency" in name or "emergency" in name else "capacity_voluntary"
    elif cap_pay > 0 and en_pay > 0:
        kind = "capacity_voluntary"
    else:
        kind = "economic_energy"

    if kind == "economic_energy":
        # Economic DR: dispatch only when LMP > marginal cost. Opportunity cost ~ 0.
        opp_gross = 0.0
        opp_inc = 0.0
        if currently_enrolled:
            notes = ("Already enrolled — gross value shown is the full re-enrollment "
                     "value; incremental shown subtracts current committed kW.")
        else:
            notes = ("Dispatches only when economically favorable; stacks with capacity DR "
                     "and with the core battery value streams. NEW program registration required.")
    else:
        # Capacity DR opportunity cost (small, ~$15-25/kW-yr equivalent at typical assumptions)
        opp_gross = 0.05 * baseline_battery_dfc_value_per_kw * kw * 0.2
        opp_inc = 0.05 * baseline_battery_dfc_value_per_kw * incremental_kw * 0.2
        if currently_enrolled:
            notes = (f"Already enrolled (current commit ~{current_committed_kw:.0f} kW). "
                     f"Battery enables expanding commitment by {incremental_kw:.0f} kW × ${cap_pay}/kW-yr "
                     f"via contract amendment with existing CSP — no new program enrollment.")
        else:
            notes = (f"Not currently enrolled. Full commitment of {kw:.0f} kW × ${cap_pay}/kW-yr. "
                     f"Mutually exclusive with other capacity DR programs.")

    net_gross = capacity_revenue_gross + energy_revenue_gross - opp_gross
    # Incremental: scale revenues by incremental_kw / kw ratio
    inc_ratio = (incremental_kw / kw) if kw > 0 else 0.0
    net_inc = (capacity_revenue_gross * inc_ratio) + (energy_revenue_gross * inc_ratio) - opp_inc

    return DRProgramValue(
        name=name, kind=kind,
        capacity_revenue_gross=capacity_revenue_gross,
        energy_revenue_gross=energy_revenue_gross,
        opportunity_cost_vs_baseline=opp_inc if currently_enrolled else opp_gross,
        net_annual_value_gross=net_gross,
        currently_enrolled=currently_enrolled,
        current_committed_kw=current_committed_kw,
        incremental_committed_kw=incremental_kw,
        net_annual_value_incremental=net_inc,
        notes=notes,
    )


def recommend_stack(programs: list[DRProgramValue]) -> dict:
    """Two-tier DR recommendation:

    Tier 1 — "Already-doing" (lowest-friction add):
        Programs the site is currently enrolled in. Battery just expands the committed
        kW via a contract amendment with the existing CSP. No new program registration,
        no new vendor, ops team is already comfortable with the program. INCREMENTAL
        value is the headline because we're not double-counting current revenue.

    Tier 2 — "Additional new program" (low-friction, but new enrollment required):
        Programs not currently active. Stack additively on Tier 1. Requires setting up
        a new CSP registration but no operational disruption (battery handles dispatch).

    The headline savings the user pitches is Battery + Tier 1. Tier 2 is shown as
    an additional option.
    """
    enrolled = [p for p in programs if p.currently_enrolled]
    not_enrolled = [p for p in programs if not p.currently_enrolled]

    # Tier 1: pick best currently-enrolled program (typically the only one — sites can't
    # double-dip capacity DR). Incremental value is the headline.
    tier1 = []
    tier1_total = 0.0
    if enrolled:
        # If somehow multiple enrolled, pick the one with highest incremental value
        best_enrolled = max(enrolled, key=lambda p: p.net_annual_value)
        tier1.append(best_enrolled.name)
        tier1_total = best_enrolled.net_annual_value

    # Tier 2: among non-enrolled programs, exclude conflicting capacity-DR (since
    # already in Tier 1 capacity). Include economic-energy + any non-conflicting types.
    tier2 = []
    tier2_total = 0.0
    has_capacity_in_tier1 = any(p.kind.startswith("capacity") for p in enrolled) if enrolled else False
    for p in not_enrolled:
        if has_capacity_in_tier1 and p.kind.startswith("capacity"):
            continue   # mutually exclusive with Tier 1 capacity DR
        tier2.append(p.name)
        tier2_total += p.net_annual_value

    return {
        "tier1_names": tier1,
        "tier1_total": tier1_total,
        "tier2_names": tier2,
        "tier2_total": tier2_total,
        # Back-compat fields used by the existing report template
        "recommended_stack": tier1 + tier2,
        "stacked_annual_value": tier1_total + tier2_total,
        "rationale": ("Tier 1 = currently-enrolled program (battery expands committed kW via "
                      "contract amendment — no new enrollment needed). Tier 2 = additional "
                      "non-enrolled programs that stack freely (require new CSP registration "
                      "but no operational impact)."),
    }


def evaluate_all(programs_cfg: list, dfc_value_per_kw: float = 160.0,
                  plc_value_per_kw: float = 98.70) -> list[DRProgramValue]:
    return [evaluate_dr_program(p, plc_value_per_kw, dfc_value_per_kw) for p in programs_cfg]
