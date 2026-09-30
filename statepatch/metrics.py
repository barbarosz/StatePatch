from __future__ import annotations
import numpy as np


def intervention_specificity(target_change: float, collateral_change: float, eps: float = 1e-8) -> float:
    return float(target_change / (abs(collateral_change) + eps))


def paired_effect(success_unpatched, success_patched):
    a = np.asarray(success_unpatched, dtype=float)
    b = np.asarray(success_patched, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    return {
        "n": int(a.size),
        "baseline_rate": float(a.mean()),
        "patched_rate": float(b.mean()),
        "absolute_gain": float((b - a).mean()),
    }
