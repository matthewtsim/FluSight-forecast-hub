"""Forecast three weeks of NY flu admissions using Prophet 1.3.0 and ED data."""

import argparse
import csv
from datetime import date, datetime, time, timedelta
from math import expm1, log, log1p
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd
from prophet import Prophet


ROOT = Path(__file__).resolve().parents[1]
ADMISSIONS_PATH = Path("target-data/target-hospital-admissions.csv")
ED_PATH = Path("target-data/target-ed-visits-prop.csv")
WEEK = timedelta(weeks=1)
LEADS = (2, 3, 4)
TRAINING_START = date(2022, 8, 1)
EASTERN = ZoneInfo("America/New_York")
LOCATION = "36"


def reference_date(issue):
    return issue + timedelta(days=(5 - issue.weekday()) % 7)


def ny_counts(lines, *, date_column="date", value_type=int):
    return {
        date.fromisoformat(row[date_column]): value_type(row["value"])
        for row in csv.DictReader(lines)
        if row["location"] == LOCATION and row["value"] not in ("", "NA")
    }


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def snapshot(path, revision):
    return git("show", f"{revision}:{path.as_posix()}").splitlines()


def snapshot_as_of(path, issue):
    cutoff = datetime.combine(issue, time.max, tzinfo=EASTERN).isoformat(timespec="seconds")
    revision = git("log", "-1", f"--before={cutoff}", "--format=%H", "--", str(path)).strip()
    if not revision:
        raise ValueError(f"No committed {path.name} available on {issue}")
    return snapshot(path, revision)


def ed_as_of(issue):
    return ny_counts(snapshot_as_of(ED_PATH, issue), value_type=float)


def ed_growth(ed, latest):
    previous = ed.get(latest - WEEK, 0)
    if previous <= 0 or ed[latest] <= 0:
        return 0.0
    return log(ed[latest] / previous)


def momentum_forecast(counts, origin, targets):
    training = pd.DataFrame(
        [(week, log1p(count)) for week, count in sorted(counts.items())
         if TRAINING_START <= week <= origin],
        columns=["ds", "y"],
    )
    training["ds"] = pd.to_datetime(training["ds"])
    model = Prophet(
        yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False,
        uncertainty_samples=0,
    )
    model.fit(training, seed=42)
    seasonal = model.predict(pd.DataFrame({"ds": pd.to_datetime(targets)}))["yhat"]

    latest = log1p(counts[origin])
    growth = (latest - log1p(counts[origin - 2 * WEEK])) / 2
    points = []
    for target, prediction in zip(targets, seasonal):
        lead = (target - origin) // WEEK
        weight = 2 ** (-lead / 2)
        # Preserve the count rounding used in the evaluated model.
        seasonal_count = max(0, round(expm1(prediction)))
        trend = max(0, latest + lead * growth)
        points.append(round(expm1((1 - weight) * log1p(seasonal_count) + weight * trend)))
    return points


def apply_ed_signal(point, growth):
    return max(0, round(expm1(log1p(point) + growth)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("as_of", type=date.fromisoformat)
    issue = parser.parse_args().as_of
    reference = reference_date(issue)
    counts = ny_counts(snapshot_as_of(ADMISSIONS_PATH, issue))
    # Match the admissions cutoff used in the evaluated Monday workflow.
    origin = reference - LEADS[0] * WEEK
    if origin not in counts or origin - 2 * WEEK not in counts:
        parser.error(f"Admissions data are incomplete for the evaluated cutoff {origin}.")
    targets = [origin + lead * WEEK for lead in LEADS]
    ed = ed_as_of(issue)
    latest_ed = max((week for week in ed if week <= issue), default=None)
    if latest_ed is None:
        parser.error(f"No New York ED data available on {issue}.")
    growth = ed_growth(ed, latest_ed)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["as_of", "target_end_date", "location", "last_observed_week",
                     "latest_ed_week", "momentum_forecast", "forecast"])
    for target, point in zip(targets, momentum_forecast(counts, origin, targets)):
        writer.writerow([issue, target, LOCATION, origin, latest_ed, point,
                         apply_ed_signal(point, growth)])


if __name__ == "__main__":
    main()
