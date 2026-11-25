"""Compare the New York ED forecast with hub medians on matching forecast rounds."""

import argparse
import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_OUTPUT = ROOT / "model-output"


def read_test_forecasts(path):
    rounds = defaultdict(dict)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if float(row["ed_weight"]) != 1:
                continue
            origin = date.fromisoformat(row["last_observed_week"])
            reference = origin + timedelta(weeks=2)
            target = date.fromisoformat(row["target_end_date"])
            rounds[reference][target] = int(row["observed"]), int(row["forecast"])
    return rounds


def hub_medians(path, reference, targets):
    with path.open(newline="") as file:
        return {
            row["target_end_date"]: float(row["value"])
            for row in csv.DictReader(file)
            if row["reference_date"] == reference
            and row["target"] == "wk inc flu hosp"
            and row["location"] == "36"
            and row["target_end_date"] in targets
            and row["output_type"] == "quantile"
            and row["output_type_id"] in ("0.5", "0.50", "0.500")
            and row["value"] not in ("", "NA")
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("residuals", type=Path)
    args = parser.parse_args()
    rounds = read_test_forecasts(args.residuals)
    results = defaultdict(list)
    for model in sorted(MODEL_OUTPUT.iterdir()):
        if not model.is_dir():
            continue
        for reference, targets in rounds.items():
            path = model / f"{reference}-{model.name}.csv"
            if not path.is_file():
                continue
            medians = hub_medians(path, str(reference), {str(target) for target in targets})
            for target, (observed, our_forecast) in targets.items():
                if str(target) in medians:
                    lead = (target - reference).days // 7 + 2
                    errors = abs(medians[str(target)] - observed), abs(our_forecast - observed)
                    results[model.name, lead].append(errors)
                    results[model.name, "all"].append(errors)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["hub_model", "weeks_from_our_last_observation", "matched_predictions",
                     "hub_mae", "our_mae_on_same_predictions"])
    for (model, lead), errors in sorted(results.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        hub_errors, our_errors = zip(*errors)
        writer.writerow([model, lead, len(errors), round(mean(hub_errors), 1),
                         round(mean(our_errors), 1)])


if __name__ == "__main__":
    main()
