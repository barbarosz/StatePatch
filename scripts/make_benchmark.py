#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from statepatch.benchmark import make_statepatchbench

p = argparse.ArgumentParser()
p.add_argument("--out", default="data")
p.add_argument("--pairs-per-type", type=int, default=8)
a = p.parse_args()
items = make_statepatchbench(a.out, a.pairs_per_type)
print(f"Wrote {len(items)} items to {a.out}/statepatchbench.jsonl")
