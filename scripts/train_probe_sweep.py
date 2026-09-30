#!/usr/bin/env python
import argparse, csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from statepatch.probes import sweep_probes

p = argparse.ArgumentParser()
p.add_argument("--features", default="results/features")
p.add_argument("--out", default="results/probe_sweep.csv")
p.add_argument("--cv", type=int, default=5)
a = p.parse_args()
res = sweep_probes(a.features, cv=a.cv)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["state_type","call","block","n","accuracy","balanced_accuracy","auc"])
    for r in res: w.writerow([r.state_type,r.call,r.block,r.n,r.accuracy,r.balanced_accuracy,r.auc])
print(f"Wrote {len(res)} probe results to {a.out}")
