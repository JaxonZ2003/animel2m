import os
import glob
import csv
import statistics as stats

pattern = os.path.join("out", "seed*_fold*", "checkpoint", "*", "metrics.csv")
files = glob.glob(pattern)

by_model = {}

for f in files:
    model = os.path.basename(os.path.dirname(f))
    by_model.setdefault(model, {"acc": [], "auc": []})
    with open(f, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            by_model[model]["acc"].append(float(row["acc"]))
            by_model[model]["auc"].append(float(row["auc"]))


def fmt(values):
    mean = stats.mean(values)
    std = stats.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


save_name = "summary.csv"
with open(save_name, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["metric", "mean±std"])
    for model in sorted(by_model.keys()):
        writer.writerow([f"{model}_acc", fmt(by_model[model]["acc"])])
        writer.writerow([f"{model}_auc", fmt(by_model[model]["auc"])])
