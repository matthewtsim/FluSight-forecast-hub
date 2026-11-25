# New York FluSight pilot

The model forecasts weekly flu hospital admissions in New York at FluSight horizons 0–2. It combines Prophet's annual seasonality, two-week admissions momentum, and the latest weekly change in the share of ED visits due to flu. The prediction intervals use lead-specific empirical log errors from the 2024–25 backtest at their original width.

## Reproduce the forecast

From the repository root, use Python 3.9 or newer with `prophet==1.3.0` installed. The scripts read the hub's Git history, including the 2024–25 ED snapshot, so use a clone with that history available.

```sh
python src/forecast_ny.py 2026-11-25 > /tmp/ny-points.csv
python src/forecast_ny_intervals.py --submission /tmp/ny-points.csv > /tmp/ny-submission.csv
```

The date argument controls which committed inputs the forecast can use. To match the evaluated leads, the admissions cutoff is two weeks before the reference Saturday. The ED signal uses the newest available observation as of the issue date.

For a local submission check, put the CSV at `model-output/TEAM-MODEL/YYYY-MM-DD-TEAM-MODEL.csv` and the matching contributor and model details at `model-metadata/TEAM-MODEL.yml`. Follow the [hub validation instructions](../model-output/README.md#local-forecast-validation). A submission branch should start from the hub's upstream main and contain the forecast and metadata files.

## Reproduce the evaluation

```sh
python src/backtest_ny.py
python src/backtest_ny.py --residuals > /tmp/ny-residuals.csv
python src/compare_ny_hub.py /tmp/ny-residuals.csv
python src/forecast_ny_intervals.py
```

The backtest uses revised ED values during 2024–25 tuning and the archived ED values available by each Monday in 2025–26. Hub comparisons use archived CSV medians on matched rounds. The hub's Wednesday deadline gives submissions a later data cutoff than the Monday replay. The interval-width experiment compares candidate widths by WIS. Submission exports use the original width because the validation-selected width performed worse in the test season.
