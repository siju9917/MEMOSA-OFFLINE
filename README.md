# MEMOSA-OFFLINE — Battery Value Evaluation Tool

Quantifies the annual $ savings a behind-the-meter battery would provide to an industrial
site, stream-by-stream, using the site's actual load data, synthesized solar, and
calibrated utility tariff + supply contract. Output is **battery-only value** via a
same-inputs counterfactual (baseline = solar + load, no battery; with-battery = same +
dispatched BESS). Capex / NPV / IRR are intentionally left to downstream analysis.

## Current reference site

1101 Cherry Hill Rd, Joliet, IL 60433 — 2 MW industrial, ComEd Very Large Load
(Secondary Voltage) delivery class, PJM ComEd zone. 4 MW AC PV installed 2026,
retrospective 2025-value evaluation. Battery under evaluation: 1,500 kW / 3,042 kWh AC-coupled.

## Usage

```
pip install -r requirements.txt
python -m bess_eval.cli configs/joliet.yaml
```

Outputs land in `results/<site_name>/`:
- `report.html` — self-contained HTML report with plots
- `results.json` — structured numbers for downstream tools
- `dispatch_mpc.csv` — hourly MPC dispatch trajectory

## Verifying tariff calibration

```
python tests/test_tariff_calibration.py
```
Reconstructs the two provided 2025 ComEd delivery bills and prints line-by-line deltas.
Targets: total within 2%, line items within 3.5%.

## Architecture

```
bess_eval/
├── config.py              # YAML site config loader
├── ingest.py              # load + solar CSV validators
├── solar_synth.py         # pvlib clear-sky + calibration to 2026 meter data
├── pjm_data.py            # synthetic LMP shape + 5CP/NSPL hour identification
├── tariff/
│   ├── comed_delivery.py  # ComEd VLL Secondary delivery — calibrated to bills
│   └── supply_index.py    # Index/LMP supply: energy + PLC + NSPL + TEC
├── battery.py             # physics (SOC dynamics, RTE, aux, SOC bounds)
├── dispatch/
│   ├── perfect_foresight.py  # monthly LP (HiGHS), SOC-chained across months
│   ├── rulebased.py         # reactive floor
│   └── mpc.py               # MPC-realistic via noisy-foresight
├── attribution.py         # per-stream battery value from dispatch trajectories
├── dr_eval.py             # DR program opportunity value
├── report.py              # HTML + JSON + CSV generation
└── cli.py                 # end-to-end runner
```

## Scope notes

- **Value streams enabled for Joliet:** DFC demand reduction, PJM PLC (5CP capacity tag),
  PJM NSPL (1CP transmission tag), energy arbitrage on LMP, solar self-consumption.
  PF correction, resilience, ancillary services are intentionally disabled.
- **Dispatch:** perfect-foresight is the upper bound; MPC-realistic is the reported
  realistic case (noisy-foresight approximation with load 4%/solar 15%/LMP 10% AR(1)
  forecast noise); rule-based is the floor.
- **Net metering:** ComEd's DG net metering caps at 2 MW non-residential, so the 4 MW
  PV system is ineligible — battery does not export to grid.
- **Data limitations:** load is hourly (ComEd bills on 30-min demand interval — minor
  understatement); solar is synthesized from 2026 Jan–Apr actuals + climatological
  cloud fractions; LMPs are synthesized from a typical PJM ComEd-zone shape anchored to
  the observed Aug 2025 period-average. Documented in report caveats.
