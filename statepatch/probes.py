from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class ProbeResult:
    state_type: str
    call: int
    block: int
    n: int
    accuracy: float
    balanced_accuracy: float
    auc: Optional[float]


def fit_binary_probe(X: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 0):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("Binary probe requires exactly two classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )
    n_splits = min(cv, np.bincount(y).min())
    if n_splits < 2:
        raise ValueError("Need at least two samples per class")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=skf, method="predict")
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
    }
    model.fit(X, y)
    return model, metrics


def save_probe(model, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_probe(path: str | Path):
    return joblib.load(path)


def load_feature_rows(feature_dir: str | Path, label_key: str = "label"):
    rows = []
    for path in sorted(Path(feature_dir).glob("*.npz")):
        d = np.load(path, allow_pickle=True)
        label = int(d[label_key])
        sample_id = str(d.get("sample_id", path.stem))
        state_type = str(d.get("state_type", "unknown"))
        calls = d["calls"].astype(int)
        blocks = d["blocks"].astype(int)
        feats = d["features"].astype(np.float32)
        for call, block, feat in zip(calls, blocks, feats):
            rows.append((sample_id, state_type, label, int(call), int(block), feat))
    return rows


def sweep_probes(feature_dir: str | Path, cv: int = 5, seed: int = 0) -> List[ProbeResult]:
    rows = load_feature_rows(feature_dir)
    groups: Dict[Tuple[int, int], List] = {}
    for sample_id, state_type, label, call, block, feat in rows:
        groups.setdefault((state_type, call, block), []).append((label, feat))
    out = []
    for (state_type, call, block), vals in sorted(groups.items()):
        y = np.asarray([v[0] for v in vals])
        X = np.stack([v[1] for v in vals], axis=0)
        # pooled feature may retain B=1
        X = X.reshape(X.shape[0], -1)
        if len(np.unique(y)) < 2 or np.bincount(y).min() < 2:
            continue
        _, m = fit_binary_probe(X, y, cv=cv, seed=seed)
        out.append(ProbeResult(state_type, call, block, len(y), m["accuracy"], m["balanced_accuracy"], m["auc"]))
    return out
