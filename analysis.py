import os
import glob
import csv
import statistics as stats

# 找所有 metrics.csv
pattern = os.path.join("out", "seed*_fold*", "checkpoint", "*", "metrics.csv")
files = glob.glob(pattern)

losses, accs, aucs = [], [], []

for f in files:
    with open(f, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            losses.append(float(row["loss"]))
            accs.append(float(row["acc"]))
            aucs.append(float(row["auc"]))

def fmt(values):
    mean = stats.mean(values)
    std = stats.stdev(values)
    return f"{mean:.4f} ± {std:.4f}"

save_name = "summary.csv"  # 和 out 同一级目录
with open(save_name, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["metric", "mean±std"])
    writer.writerow(["loss", fmt(losses)])
    writer.writerow(["acc", fmt(accs)])
    writer.writerow(["auc", fmt(aucs)])
