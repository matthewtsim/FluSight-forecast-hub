"""Replay the NY forecast on the tuning and test seasons using archived inputs."""

import argparse
from collections import defaultdict
import csv
from datetime import date, timedelta
from statistics import mean
import sys

from forecast_ny import (
    ADMISSIONS_PATH, ED_PATH, LEADS, ROOT, WEEK, apply_ed_signal, ed_as_of,
    ed_growth, momentum_forecast, ny_counts, snapshot,
)


def rounds(truth, year):
    start, end = date(year, 10, 1), date(year + 1, 5, 31)
    for origin in sorted(week for week in truth if start <= week <= end):
        archive_date = origin + WEEK
        path = ROOT / "weekly-summaries" / str(archive_date) / (
            f"{archive_date}_flu_target_hospital_admissions_data.csv"
        )
        if not path.is_file():
            continue
        targets = [origin + lead * WEEK for lead in LEADS]
        with path.open(newline="") as file:
            counts = ny_counts(file, date_column="week_ending_date")
        if (all(origin - offset * WEEK in counts for offset in range(4))
                and all(target in truth for target in targets)):
            yield origin, targets, counts


def evaluate(truth, year):
    # This pre-test snapshot contains revised ED values for the tuning season.
    tuning_ed = ny_counts(snapshot(ED_PATH, "5ed1c873"), value_type=float) if year == 2024 else None
    for origin, targets, counts in rounds(truth, year):
        issue = origin + timedelta(days=9)
        if tuning_ed is None:
            ed, cutoff = ed_as_of(issue), issue
        else:
            ed, cutoff = tuning_ed, origin
        growth = ed_growth(ed, max(week for week in ed if week <= cutoff))
        for lead, target, point in zip(LEADS, targets, momentum_forecast(counts, origin, targets)):
            observed = truth[target]
            for weight, forecast in ((0.0, point), (1.0, apply_ed_signal(point, growth))):
                yield origin, target, lead, weight, observed, forecast, observed - forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", action="store_true", help="Write 2025–26 forecast errors.")
    args = parser.parse_args()
    with (ROOT / ADMISSIONS_PATH).open(newline="") as file:
        truth = ny_counts(file)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    if args.residuals:
        writer.writerow(["last_observed_week", "target_end_date", "weeks_ahead", "ed_weight",
                         "observed", "forecast", "observed_minus_forecast"])
        writer.writerows(evaluate(truth, 2025))
        return

    writer.writerow(["period", "ed_weight", "weeks_ahead", "n_predictions", "mae", "mean_error"])
    for year, period in ((2024, "tuning"), (2025, "test")):
        errors = defaultdict(list)
        for _, _, lead, weight, _, _, residual in evaluate(truth, year):
            errors[weight, "all" if year == 2024 else lead].append(-residual)
        for (weight, lead), values in sorted(errors.items()):
            writer.writerow([period, weight, lead, len(values), round(mean(map(abs, values)), 1),
                             round(mean(values), 1)])


if __name__ == "__main__":
    main()
