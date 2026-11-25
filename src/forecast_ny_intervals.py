"""Export NY forecast quantiles or evaluate interval widths by WIS."""

import argparse
from collections import defaultdict
import csv
from datetime import date
from math import expm1, log1p
from pathlib import Path
from statistics import mean
import sys

from backtest_ny import evaluate
from forecast_ny import ADMISSIONS_PATH, LEADS, LOCATION, ROOT, WEEK, ny_counts, reference_date


QUANTILES = (0.01, 0.025, 0.05) + tuple(i / 100 for i in range(10, 95, 5)) + (0.95, 0.975, 0.99)
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
MIN_ERRORS = 8
TARGET = "wk inc flu hosp"


def quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    index = int(position)
    following = min(index + 1, len(ordered) - 1)
    return ordered[index] + (position - index) * (ordered[following] - ordered[index])


def error_offsets(rows):
    errors = sorted(log1p(observed) - log1p(point) for _, _, _, observed, point in rows)
    center = quantile(errors, 0.5)
    return tuple(quantile(errors, p) - center for p in QUANTILES)


def bounds(point, offsets, scale):
    return tuple(max(0, expm1(log1p(point) + scale * offset)) for offset in offsets)


def weighted_interval_score(observed, forecast):
    # For symmetric quantile levels, WIS is the mean of twice the pinball loss.
    return mean(
        2 * max(p * (observed - value), (1 - p) * (value - observed))
        for p, value in zip(QUANTILES, forecast)
    )


def score(observed, forecast):
    lower = forecast[QUANTILES.index(0.1)]
    upper = forecast[QUANTILES.index(0.9)]
    return weighted_interval_score(observed, forecast), lower <= observed <= upper


def by_lead(truth, year):
    rows = defaultdict(list)
    for origin, target, lead, weight, observed, point, _ in evaluate(truth, year):
        if weight == 1.0:
            rows[lead].append((origin, target, lead, observed, point))
    return rows


def tune(rows_by_lead):
    scores = {scale: [] for scale in SCALES}
    for rows in rows_by_lead.values():
        for origin, _, _, observed, point in rows:
            # Exclude outcomes whose target week had not yet been observed.
            available = [row for row in rows if row[1] <= origin]
            if len(available) < MIN_ERRORS:
                continue
            offsets = error_offsets(available)
            for scale in SCALES:
                scores[scale].append(score(observed, bounds(point, offsets, scale)))
    selected = min(SCALES, key=lambda scale: (mean(wis for wis, _ in scores[scale]),
                                              abs(scale - 1.0)))
    return selected, scores


def write_score(writer, key, scores):
    wis, coverage = zip(*scores)
    writer.writerow([*key, len(scores), round(mean(wis), 2), round(100 * mean(coverage), 1)])


def export_quantiles(writer, path, tuning, submission):
    with path.open(newline="") as file:
        points = list(csv.DictReader(file))
    if len(points) != len(LEADS) or len({row["as_of"] for row in points}) != 1:
        raise ValueError("Expected three New York forecasts from a single issue date.")
    reference = reference_date(date.fromisoformat(points[0]["as_of"]))
    expected_dates = [reference + horizon * WEEK for horizon in range(len(LEADS))]
    points.sort(key=lambda row: row["target_end_date"])
    if [row["target_end_date"] for row in points] != [str(target) for target in expected_dates]:
        raise ValueError("Expected one forecast each for horizons 0, 1 and 2.")

    rows = []
    for horizon, (row, target, lead) in enumerate(zip(points, expected_dates, LEADS)):
        origin = date.fromisoformat(row["last_observed_week"])
        point = int(row["forecast"])
        if row["location"] != LOCATION or target - origin != lead * WEEK or point < 0:
            raise ValueError("Expected nonnegative New York forecasts at calibrated leads 2–4.")
        offsets = error_offsets(tuning[lead])
        for probability, value in zip(QUANTILES, bounds(point, offsets, 1.0)):
            if submission:
                rows.append([reference, LOCATION, horizon, TARGET, target,
                             "quantile", probability, round(value)])
            else:
                rows.append([row["as_of"], target, LOCATION, 1.0, probability, round(value)])

    if submission:
        writer.writerow(["reference_date", "location", "horizon", "target", "target_end_date",
                         "output_type", "output_type_id", "value"])
    else:
        writer.writerow(["as_of", "target_end_date", "location", "scale", "quantile", "value"])
    writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--forecast", type=Path, help="Export original-width quantiles.")
    mode.add_argument("--submission", type=Path, help="Export a local FluSight submission CSV.")
    args = parser.parse_args()
    with (ROOT / ADMISSIONS_PATH).open(newline="") as file:
        truth = ny_counts(file)
    tuning = by_lead(truth, 2024)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    if args.forecast or args.submission:
        try:
            export_quantiles(writer, args.forecast or args.submission, tuning, bool(args.submission))
        except (ValueError, KeyError) as error:
            parser.error(str(error))
        return

    selected, validation = tune(tuning)
    offsets = {lead: error_offsets(rows) for lead, rows in tuning.items()}
    writer.writerow(["period", "scale", "selected", "weeks_ahead", "n", "mean_wis", "coverage_80_pct"])
    for scale, scores in validation.items():
        write_score(writer, ("validation", scale, scale == selected, "all"), scores)

    test = by_lead(truth, 2025)
    for scale in sorted({1.0, selected}):
        pooled = []
        for lead in LEADS:
            scores = [score(observed, bounds(point, offsets[lead], scale))
                      for _, _, _, observed, point in test[lead]]
            write_score(writer, ("test", scale, scale == selected, lead), scores)
            pooled.extend(scores)
        write_score(writer, ("test", scale, scale == selected, "all"), pooled)


if __name__ == "__main__":
    main()
