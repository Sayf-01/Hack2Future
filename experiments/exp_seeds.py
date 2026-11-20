"""Is 'rf-importance select + rf' beating plain RF, or is +0.005 just this
one fold split? Same comparison across four different fold seeds.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from peptide_classifier import cv, features, leak          # noqa: E402
from peptide_classifier.load import load                   # noqa: E402

from sklearn.base import clone              # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import SelectFromModel  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline       # noqa: E402

T0 = time.time()


def rf(seed=42):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=4, random_state=seed)


tr, _, _ = load(str(REPO / "participant_datasets/initial/anticancer_train.csv"))
seqs = tr["sequence"].tolist()
y = tr["label"].to_numpy()
groups = leak.similarity_groups(seqs, 0.6)
X = pd.concat([features.build(seqs, ("aac",)),
               features.build(seqs, ("dpc",)),
               features.build(seqs, ("phys",))], axis=1).to_numpy(dtype=np.float32)

cands = {
    "plain rf": rf(),
    "rf-select + rf": Pipeline([("sel", SelectFromModel(rf(seed=7), threshold="median")),
                                ("m", rf())]),
}

rows = []
for seed in (42, 7, 13, 2026):
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                  .split(X, y, groups))
    row = {"fold_seed": seed}
    for name, model in cands.items():
        p = np.zeros(len(y))
        for tr_i, te_i in splits:
            p[te_i] = clone(model).fit(X[tr_i], y[tr_i]).predict_proba(X[te_i])[:, 1]
        row[name] = cv.metrics(y, p)["accuracy"]
    row["delta"] = row["rf-select + rf"] - row["plain rf"]
    rows.append(row)
    print(f"[{time.time() - T0:6.1f}s] seed {seed}: plain {row['plain rf']:.4f}  "
          f"select {row['rf-select + rf']:.4f}  delta {row['delta']:+.4f}", flush=True)

res = pd.DataFrame(rows)
print("\n" + res.to_string(index=False))
print(f"\nmean delta {res['delta'].mean():+.4f}  (sd {res['delta'].std():.4f})  "
      f"wins {int((res['delta'] > 0).sum())}/4")
print(f"plain rf across seeds: mean {res['plain rf'].mean():.4f}  "
      f"min {res['plain rf'].min():.4f}  max {res['plain rf'].max():.4f}  "
      f"spread {res['plain rf'].max() - res['plain rf'].min():.4f}")
