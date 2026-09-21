# Replication code

This repository contains the replication code and delivered result files for the JBES submission, available at <https://github.com/yaosong-zhan/jbes-replication-code>. The Monte Carlo drivers use 2,000 replications per configuration (and per true break type for classification). The dimension-11 critical-value calibration uses 50,000 Brownian bridges on a 20,000-point grid.

## Contents

- `monte_carlo/`: simulation engines, drivers, calibration scripts, and delivered CSV/JSON results.
- `empirical/`: scripts for constructing the event panel, estimating break statistics, placebo checks, and market-quality comparisons.
- `requirements.txt`: Python dependencies.
- `DATA_AVAILABILITY.md`: the data and licensing statement for the empirical application.

## Monte Carlo replication

Create an environment from `requirements.txt`. Run the drivers from `monte_carlo/` so that local imports resolve:

```sh
python calibrate_full_vector_cv.py
python rerun_revision_mc.py size
python rerun_revision_mc.py comparison
python run_remaining_revision_mc.py
```

The delivered result files are in `monte_carlo/Results/`; `revision_2000_manifest.json` records replication counts, row counts, and SHA-256 hashes. The drivers recreate the size, power, classification, multiple-break, robustness, cross-sectional-correlation, and competing-method results used in the paper and supplement.

## Empirical replication

The empirical scripts require licensed tick-level limit-order-book snapshots, earnings-report dates from RESSET, and a compatible trading calendar. Those data are not included because the providers prohibit redistribution. Before running the empirical pipeline, place the licensed files in local storage and update the path constants in `empirical/build_earnings_panel.py`; the remaining empirical scripts consume the generated panel and result files.

The main steps are:

1. `build_earnings_panel.py` constructs the stock-day panel and detects/classifies breaks.
2. `empirical_revision_analysis.py` produces event-window contrasts, clustered inference, placebo tests, and threshold sensitivity.
3. `compute_alt_measures.py` and `table_alt_measures.py` compute conventional market-quality comparisons.
4. `find_joint_example.py` and `plot_break_examples.py` produce the illustrative order-book plots.

No proprietary data are distributed with this folder.

