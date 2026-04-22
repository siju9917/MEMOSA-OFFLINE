# Battery Value Evaluation Tool — Engineering Specification

## 1. Overview & Goals

Build a Python tool that quantifies the economic value of adding an on-site battery energy storage system (BESS) to an industrial facility that already has solar PV. The tool must work across many sites with different tariffs, different solar/load profiles, and different available value streams.

The tool compares two scenarios using identical inputs:
- **Baseline:** solar + load, no battery
- **With battery:** solar + load + battery dispatched by the controller logic

The "battery-only value" is the delta between these two scenarios, computed through the same tariff and revenue engine. This isolates battery contribution from solar contribution.

### Core design principles

1. **Realistic dispatch is the primary output.** The tool must simulate how a battery would *actually* operate — making decisions in the moment with imperfect forecasts, not with hindsight. Perfect-foresight dispatch is computed only as an upper-bound benchmark, never as the reported ROI.
2. **Modular value streams.** Each value stream (demand charge reduction, TOU arbitrage, demand response, etc.) is a pluggable module that can be enabled/disabled per site based on what's actually available.
3. **Tariff fidelity.** The tool must reproduce the site's actual utility bill within a tight tolerance (target: within 1–2% of the example bills) before any battery analysis is trusted. If the baseline bill can't be reconstructed, the analysis is invalid.
4. **Co-optimization, not stacking.** Value streams must be co-optimized in a single dispatch problem. Summing independently computed value streams double-counts energy and inflates value by 15–40%.
5. **Site-agnostic.** The tool takes a config file per site; no code changes should be required to evaluate a new site.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Site Config (YAML/JSON)                 │
│  (paths to data, tariff, location, battery specs, enabled    │
│   value streams, DR programs, etc.)                          │
└─────────────────────────────────────────────────────────────┘
                              │
      ┌───────────────────────┼───────────────────────┐
      ▼                       ▼                       ▼
┌───────────┐          ┌─────────────┐         ┌──────────────┐
│ Data      │          │ Tariff      │         │ Battery      │
│ Ingestion │          │ Engine      │         │ Model        │
│ & Clean   │          │ (+bill      │         │ (physics,    │
│           │          │  calibration│         │  degradation)│
└───────────┘          └─────────────┘         └──────────────┘
      │                       │                       │
      └───────────────────────┼───────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Forecast Module  │
                    │ (load, solar,    │
                    │  prices, DR)     │
                    └──────────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │  Dispatch Controller       │
                │  - Rule-based              │
                │  - Rolling MPC (primary)   │
                │  - Perfect foresight       │
                │    (benchmark only)        │
                └───────────────────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │  Value Stream Modules     │
                │  (tariff bill, DR revenue,│
                │   resilience, ancillary,  │
                │   etc.)                   │
                └───────────────────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │  Reporting & Sensitivity  │
                │  (NPV, IRR, payback,      │
                │   per-stream attribution) │
                └───────────────────────────┘
```

---

## 3. Input Data (Per Site)

### Required

- **Load time series:** interval meter data, at minimum 12 months, 15-minute resolution preferred (1-minute if available — matters for demand charges). CSV with timestamp (ISO 8601, timezone-aware) and kW columns.
- **Solar time series:** measured AC output at the same resolution and timestamps as the load. If only DC is available, note inverter specs to model clipping.
- **Example winter bill and example summer bill:** PDF or itemized breakdown. Must include every line item: energy kWh by TOU period, demand kW (by period if applicable), fixed charges, riders, taxes, power factor penalties, etc.
- **Utility tariff schedule:** the current rate sheet for the site's rate class. If multiple rates are available to the site, include all candidates — the tool may recommend a rate switch.
- **Location:** lat/lon, utility service territory, ISO/RTO region. Drives which grid-services and DR programs are available.
- **Battery options to evaluate:** a list of (power kW, energy kWh, chemistry, round-trip efficiency, expected cycle life, capex, O&M). The tool runs the analysis for each option.

### Optional / site-specific

- List of DR programs the site is enrolled in or could enroll in, with terms (notification, event duration, event frequency cap, payment structure).
- ISO market prices if the site could participate in wholesale energy or ancillary services (BTM participation varies by ISO — check rules).
- Historical outage data (duration, frequency) for resilience valuation, and site Value of Lost Load ($/kWh of unserved load).
- Coincident peak tag data if the utility charges capacity or transmission based on ICAP/PLC/4CP tags.
- Power factor history if a PF penalty exists in the tariff.
- Export compensation rate (feed-in, net metering tier, or zero).

### Data validation (mandatory at load time)

- Check timestamp alignment between load and solar; resample to common cadence.
- Flag missing intervals; fill short gaps (<1 hour) with linear interpolation, longer gaps with same-hour-same-day-of-week average, and log all fills.
- Check unit sanity: load should be positive (unless the site exports), solar non-negative, magnitudes match nameplate.
- Compute and report: annual kWh consumed, annual kWh generated, peak demand, load factor, solar penetration. These are sanity checks for the user.

---

## 4. Tariff Engine

This is the most error-prone part of any battery tool. Build it carefully.

### Tariff data model

Represent a tariff as a structured object supporting these component types:

- **Fixed charges:** flat $/month.
- **Energy charges:** $/kWh, optionally by TOU period (defined by month-of-year × day-of-week × hour-of-day matrix). Support tiered rates if applicable.
- **Demand charges:**
  - *Facility demand:* max kW over the billing period.
  - *TOU demand:* max kW within a defined window (e.g., on-peak demand).
  - *Ratcheted demand:* billed demand = max(current month peak, X% of peak from the last N months).
  - *Coincident peak:* demand measured at specific system-peak hours (4CP, ICAP tag, etc.). Requires a list of peak hours to be supplied or forecasted.
- **Riders / adjustments:** per-kWh or per-kW surcharges (e.g., fuel adjustment, transmission rider, renewable charge).
- **Power factor penalty:** formula-based, usually a multiplier on demand when PF is below a threshold.
- **Export compensation:** $/kWh for net export, possibly TOU, possibly zero, possibly capped.
- **Taxes:** applied at end.

Implement as a class hierarchy with a `compute_bill(meter_flows, billing_period)` method that returns an itemized breakdown. All tariff components should be independently testable.

### Bill calibration (critical)

1. Ingest the site's actual interval data for the billing periods of the two example bills.
2. Compute what the tariff engine *says* the bill should be.
3. Compare line-by-line to the actual bill.
4. Target: each line item within 1–2% of actual.
5. If calibration fails, the tariff definition is wrong — iterate until resolved. Common issues: misidentified TOU windows, wrong demand measurement interval (15-min vs 30-min vs 60-min), missing rider, wrong definition of "peak demand" (e.g., some utilities use the average of the top X intervals).
6. Log the final calibrated tariff as the source of truth for that site. Flag if calibration tolerance cannot be met.

### Demand measurement intervals

The utility's demand integration interval matters a lot. A 15-minute demand charge is much easier to shave than a 1-minute demand charge. Read this off the bill / rate sheet and apply it consistently in the simulation. Resample interval data to match.

---

## 5. Battery Model

### Core physics (per timestep)

State variables:
- `SOC[t]`: state of charge (kWh), bounded by `[SOC_min, SOC_max]`
- `P_chg[t]`, `P_dis[t]`: charge and discharge power (kW), non-negative, bounded by `P_max`

Dynamics:
```
SOC[t+1] = SOC[t] + (eta_chg * P_chg[t] - P_dis[t] / eta_dis) * dt
```
where `eta_chg * eta_dis = round_trip_efficiency`. Typical: ~90–95% RTE for Li-ion AC-coupled, ~95–98% for DC-coupled (avoids one inverter conversion).

Prevent simultaneous charge/discharge: enforce `P_chg[t] * P_dis[t] = 0` (use complementarity or binary variable in MILP).

### Coupling configuration

- **AC-coupled:** battery has its own inverter. Solar inverter is separate. Battery can charge from grid or solar (after solar inverter losses).
- **DC-coupled:** battery sits on the DC bus with solar. Shared inverter. Can capture solar clipping losses. Has an inverter export limit that constrains combined solar+battery discharge.

The config should specify which, because it affects the dispatch constraints. DC-coupled systems are notably better at capturing clipped solar and can only charge from grid through the inverter's reverse capability (often limited or disallowed).

### Degradation

Offer two models, user-selectable:

1. **Simple cycle-based:** capacity fade = (cycles / rated_cycle_life) × (1 − EOL_capacity). Count cycles via equivalent full cycle tracking (sum of discharge energy / rated energy, divided by 2).
2. **Rainflow + calendar:** combine depth-of-discharge-dependent cycle counting (rainflow algorithm on SOC trajectory) with calendar aging. More accurate, more complex. Use for longer project-life simulations (>5 years).

Degradation should feed back into the simulation: year-N battery has reduced usable capacity. For a 10-year project, step through years and recompute.

### Round-trip efficiency and aux loads

- Include parasitic/HVAC load of the battery system (typically 0.5–2% of rated energy per day standby). Small but compounds.
- If degradation includes power fade (max C-rate decline), model it too.

---

## 6. Forecast Module (for realistic dispatch)

This is the key to "no hindsight." At every control timestep, the dispatch controller must make decisions using only information available *at that moment*. The forecast module generates those forecasts.

### Forecast targets

- **Load** (next 24–48 hours at the control timestep resolution)
- **Solar** (same horizon)
- **Grid prices** — trivial if TOU (known deterministically), required if real-time / wholesale
- **Coincident peak risk** — probability that a given hour is a system peak hour (utility-specific, may use temperature as a predictor)
- **DR event probability** — if DR is event-driven with short notification

### Forecast methods (implement at least these)

1. **Persistence:** tomorrow = today. Naive baseline.
2. **Seasonal naive / same-hour-day-of-week average:** use last N weeks' same-hour-same-dow average.
3. **Exponential smoothing (Holt-Winters):** handles trend and seasonality.
4. **Gradient-boosted regression (XGBoost or LightGBM):** features = hour, day-of-week, month, lagged load, temperature forecast, holiday flag. Train on first 6–9 months of historical data, validate on the rest. **Critical:** enforce strict temporal separation to avoid data leakage.
5. **Solar-specific:** clear-sky model (pvlib) × cloud cover forecast, or persistence-of-clear-sky-index. If weather forecast data isn't available, use day-ahead persistence with a clear-sky adjustment.

The forecaster should produce both a point forecast and an uncertainty band (quantile forecasts from the GBM, or residual standard deviation). The MPC can use the point forecast; robust variants can use the band.

### Forecast evaluation

Log forecast errors (MAPE, RMSE) per target. Report them in the final output so the user knows how realistic the "realistic" dispatch actually is. If forecast error is very high, the reported value is more conservative.

---

## 7. Dispatch Controllers

Implement three dispatch modes. All three use the same value-stream modules and the same tariff engine — the only difference is what information they have access to.

### 7.1 Rule-based (reactive)

No forecasts. Simple heuristics:
- Charge battery from excess solar (solar > load).
- Discharge battery when load > threshold (set to avoid exceeding current-month peak demand).
- Honor TOU: preferentially discharge on-peak, charge off-peak.
- DR events: on notification, discharge to contracted level.

This is the worst-case floor. Useful as a sanity check and as a "simple controller" ROI for customers who won't deploy sophisticated software.

### 7.2 Rolling Model Predictive Control (PRIMARY — this is the reported result)

At each control timestep `t`:

1. Generate forecasts for load, solar, prices, DR risk over horizon `H` (typically 24–48 hours).
2. Solve a MILP / LP that minimizes total cost (net of revenues) over the horizon, subject to battery and tariff constraints.
3. Execute only the first timestep's decision.
4. Advance to `t+1` with the *actual* load and solar (from the historical data), update SOC based on what actually happened, and re-solve.

This simulates real-world operation: the controller commits to an action based on a forecast, then the world reveals what actually happened, and the controller adjusts.

**Horizon length:** long enough to capture a full TOU cycle (≥24 hours) and ideally reach into the next day's peak window. For coincident peak management or slow demand charge management, longer horizons or hierarchical control (see below) help.

**Demand charge handling in MPC (tricky):**
Demand charges are monthly but the MPC horizon is hours-to-days. Naïve MPC will under-weight demand charges because it only sees part of the month. Handle this via a hierarchical structure:

- **Upper layer (monthly):** track the running monthly peak. Forecast the expected final monthly peak (e.g., highest-so-far + expected future peak based on distribution of historical peaks). Set a demand target as a constraint in the MPC.
- **Lower layer (MPC):** treat the demand target as a soft constraint with a high penalty ($/kW large enough to dominate energy arbitrage decisions). Update the target monthly as peaks are revealed.

For coincident peak: if the utility issues day-ahead peak notifications, treat them as known the day before. If not, maintain a peak probability per hour (based on temperature, load forecast, historical peak patterns) and dispatch probabilistically — discharge preemptively on high-risk days.

**Solver:** Pyomo + HiGHS (free) is a reasonable default. Gurobi or CPLEX if performance is an issue (large sites, long simulations). cvxpy is simpler for pure LP/QP but less flexible for MILP integer logic.

### 7.3 Perfect foresight (benchmark, NOT reported as ROI)

Solve a single MILP over the entire simulation horizon (e.g., one year) with complete knowledge of every future load, solar, and price value.

Purpose: establishes the theoretical upper bound. The gap between MPC and perfect foresight indicates how much value is lost to forecast error. Report this gap in the output — it's informative for the customer and for evaluating whether better forecasting/controls would be worth investing in.

Typical gap: 5–20%. If the gap is >30%, investigate forecast quality.

---

## 8. Value Stream Modules

Each module is independently enable-able per site via config. Each module has:
- Constraints it adds to the dispatch optimization
- Revenue/savings calculation given a dispatch trajectory
- Attribution method for the final report

### 8.1 Solar self-consumption

- Relevant when export compensation < retail import rate.
- Battery charges from solar that would otherwise export; discharges to offset later grid import.
- Value = (avoided import rate − earned export rate) × shifted energy.
- Already naturally captured by the co-optimization; attribution requires comparing against a baseline where battery doesn't charge from solar.

### 8.2 Energy arbitrage (TOU or real-time)

- Charge during low-price periods, discharge during high-price periods.
- Constrained by round-trip efficiency: arbitrage only profitable if `(peak_price − off_peak_price) / off_peak_price > 1/RTE − 1` (roughly).
- Attribution: savings in energy charges between baseline and with-battery.

### 8.3 Demand charge reduction

- Most valuable stream for most industrial sites.
- Battery discharges during expected peak intervals to cap billed demand.
- Requires careful MPC handling (see above).
- Attribution: (baseline peak − with-battery peak) × demand rate, per demand component (facility, TOU, ratcheted).
- Also evaluate: does the battery enable rate switching to a rate with higher energy but lower demand (or vice versa)? Implement rate-switch analysis as an outer loop.

### 8.4 Coincident peak avoidance (ICAP/PLC/4CP)

- Applies in ERCOT (4CP), PJM (PLC), NYISO (ICAP), ISO-NE (ICAP), and many transmission tariffs.
- Value is typically very large per avoided kW (hundreds to thousands of $/kW-year) but only a few hours per year matter.
- Requires peak-day prediction; utilities sometimes notify day-ahead.
- Implement as a separate module with region-specific logic. For sites not in a coincident peak regime, disable.

### 8.5 Demand response

- Parameterize each DR program as: event trigger type (scheduled / dispatch / economic), notification lead time, event duration, max events per year, max hours per year, payment structure (capacity $/kW-month, energy $/kWh, or both), penalty for under-performance.
- In simulation: generate synthetic DR events based on historical program call patterns for that utility/region (or real historical events if available).
- Dispatch rule: on event notification, MPC adds a constraint requiring discharge of committed kW during the event window. Co-optimize against other value streams.
- Attribution: DR revenue = capacity payment + (energy payment × delivered kWh) − any performance penalties.
- Important: DR commitment reduces flexibility for other value streams during event windows — the co-optimization handles this naturally.

### 8.6 Ancillary services / frequency regulation

- Only if ISO allows BTM participation (CAISO, NYISO, PJM have some pathways; ERCOT is limited).
- Typically requires a more granular simulation (seconds-to-minutes) and a different revenue model (regulation-up + regulation-down, mileage payments).
- Implement as an optional module; skip unless site is in a relevant ISO and customer intends to participate.

### 8.7 Resilience / backup power

- Value = Value of Lost Load ($/kWh) × expected unserved load avoided.
- Need: outage frequency/duration distribution for the location (utility reliability reports, or a default from EIA).
- Simulate N outage scenarios; compute load that could have been served by battery given SOC at outage time.
- Attribution: annualized expected VoLL × served kWh.
- Sensitive to inputs; always run as a sensitivity.

### 8.8 Power factor correction

- If tariff has a PF penalty, battery inverters can supply reactive power.
- Value = avoided PF penalty (computed from tariff engine).
- Requires inverter specs (reactive power capability, typically 0.8–1.0 leading/lagging).

### 8.9 ITC / tax incentives

- Battery capex qualifies for ITC under IRA; apply to capex in financial model. Percentage depends on year, domestic content, energy community, labor compliance bonuses.
- Not a dispatch decision, but a capex adjustment. Model in the NPV/IRR calc.

### 8.10 Avoided export curtailment / clipping recovery

- If solar is interconnection-limited (can't export above X kW) or inverter-clipped, battery can absorb otherwise-lost solar.
- Only relevant for DC-coupled (for inverter clipping) or AC-coupled with export cap.
- Attribution: recovered kWh × value (self-consumption or discharged later).

---

## 9. Counterfactual Methodology (Isolating Battery Value)

This is how you ensure the number you report is battery-only value, not bundled solar+battery value.

1. **Baseline run:** same solar, same load, **no battery**. Pass through the full tariff engine and all enabled value streams (some of which, like DR, may be zero without a battery). Record total annual cost.
2. **With-battery run:** identical inputs, add battery with chosen dispatch controller. Record total annual cost and any external revenues.
3. **Battery value = Baseline_cost − With_battery_cost + External_revenues_with_battery − External_revenues_without_battery.**
4. Repeat for each battery size option.

Run the counterfactual with the **MPC dispatch controller** as the primary case. Also run perfect foresight for upper bound. Report both.

**Do not** sum independently computed value streams — always use the dispatch output to attribute value after the fact.

---

## 10. Financial Rollup

Given annual battery value (from step 9), compute:
- **Lifetime value:** step through years 1 through N (typical project life = battery warranty, 10–15 years). Apply degradation to capacity; optionally escalate tariffs (default 2–3%/year, sensitivity-tunable). Discount to present value at user-supplied discount rate (default 7–10%).
- **Capex:** battery cost (supplied per option), installation, interconnection study, software/controls, ITC deduction.
- **Opex:** O&M ($/kW-year), augmentation (capacity top-up at year X), insurance.
- **Outputs:** NPV, IRR, simple payback, discounted payback, LCOS if useful.

Run for each battery size option; report a chart of NPV vs. size so user can see the sweet spot.

---

## 11. Sensitivity & Scenario Analysis

At minimum, run sensitivities on:

- **Battery size** (all configured options)
- **Forecast quality** (MPC with perfect forecast vs. naive forecast vs. tuned forecast)
- **Tariff escalation** (1%, 3%, 5%/yr)
- **Load growth** (0%, 1%, 3%/yr)
- **Degradation rate** (nominal, +25%, −25%)
- **Discount rate** (5%, 8%, 12%)
- **DR event frequency** (nominal, 2×, 0.5×)
- **Outage frequency** (for resilience value — often the biggest swing factor)

Present as tornado chart or equivalent.

---

## 12. Outputs

Generate a site report with:

1. **Input summary:** data quality flags, tariff calibration result, forecast skill metrics.
2. **Baseline profile:** annual energy, demand, bill breakdown (by line item).
3. **Per-battery-size results table:** capex, annual savings broken down by value stream, NPV, IRR, payback.
4. **Dispatch visualization:** sample weeks (one winter, one summer) showing load, solar, SOC, grid import/export, and battery action. Crucial for user to sanity-check behavior.
5. **Value attribution:** how much of the total savings comes from each value stream. Pie chart + table.
6. **MPC vs. perfect foresight gap:** shows forecast-related value loss.
7. **Sensitivity charts.**
8. **Monthly bill comparison:** 12 months side-by-side (baseline vs. with-battery), to show seasonality of value.

Export report as HTML or PDF plus all raw results as a structured JSON/CSV so downstream tools can consume.

---

## 13. Recommended Python Stack

- **Data handling:** pandas, numpy, pyarrow (parquet for large time series).
- **Optimization:** Pyomo (preferred for MILP flexibility) or cvxpy (simpler for LP/QP). Solver: HiGHS (free, good enough) or Gurobi (fast, commercial).
- **Forecasting:** scikit-learn, xgboost / lightgbm, statsmodels (for Holt-Winters), pvlib (for solar clear-sky).
- **Solar physics (if needed):** pvlib, NSRDB API for irradiance.
- **Visualization:** plotly (interactive HTML reports) or matplotlib + Jinja2 templates.
- **Config:** pydantic for typed, validated site configs. YAML for human editing.
- **Testing:** pytest. Must include unit tests for tariff components, battery dynamics, and a regression test against a known reference site.
- **Orchestration:** pipeline as CLI with a single command that takes a site config and produces a report. Consider Snakemake or simple Python CLI (Typer / Click).

---

## 14. Critical Things to Get Right (Common Failure Modes)

1. **Tariff bill calibration must match.** If you can't reproduce the actual bill from the tariff engine, everything downstream is noise. Don't skip this.
2. **Demand interval resolution.** 15-minute vs. 1-minute demand charges behave very differently. A battery that shaves a 15-min average can't necessarily shave a 1-min peak.
3. **MPC demand charge handling.** A naive MPC with a 24h horizon will ignore monthly demand charges because they look free over a day. Use hierarchical control with an explicit monthly peak target.
4. **No data leakage in forecasts.** Forecasts at time `t` may only use data from time `<= t`. Any use of future data makes the "realistic" dispatch secretly perfect-foresight.
5. **Round-trip efficiency and auxiliary loads matter.** A 2% RTE miss is easily a 10–20% swing in arbitrage value.
6. **Degradation over project life.** Year 10 capacity may be 70–80% of year 1. Don't assume nameplate for lifetime value.
7. **Timezone handling.** Utility billing periods, TOU windows, and interval data must all resolve correctly around DST transitions. Use timezone-aware timestamps everywhere.
8. **Simultaneous charge/discharge.** Must be prevented; a relaxed LP will exploit it to fake free energy.
9. **Battery SOC continuity across billing periods.** Don't reset SOC monthly for the simulation — it's continuous.
10. **Co-optimization not stacking.** Never sum independently optimized value streams. Always attribute from a single dispatch trajectory.
11. **Sanity-check dispatch visualizations manually.** Look at a few sample weeks. If the battery is doing something weird (charging on-peak, ignoring a demand spike), your constraints or objective are wrong.

---

## 15. Suggested Development Order

1. Data ingestion + validation module.
2. Tariff engine with bill calibration. Validate on at least 2 real sites before moving on.
3. Battery physics model + simple rule-based dispatch. End-to-end run producing a bill comparison.
4. Perfect-foresight MILP. Verify battery value is sensible on a test case.
5. Forecasting module with at least persistence and seasonal naive.
6. Rolling MPC. Verify MPC value is lower than perfect-foresight but higher than rule-based.
7. Add value stream modules one at a time, each with its own tests.
8. Financial rollup and sensitivity.
9. Reporting.
10. Multi-site config and batch runner.

---

## 16. Deliverables

- Python package with CLI (`python -m bess_eval --site-config path/to/site.yaml`).
- Per-site HTML + PDF report.
- Raw results JSON/CSV for downstream consumption.
- Unit tests + one integration test per reference site.
- README with usage examples and site config schema.
- A reference site config committed to the repo so the tool can be smoke-tested without real data.
