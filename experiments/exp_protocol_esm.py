"""(a) What does our own frozen model score under a PLAIN random split?
    (b) Does ESM-2 concatenated with aac+dpc+phys beat aac+dpc+phys alone?

Both run entirely out of the scratchpad against copies of the data. Nothing in
models/, frozen_v1/ or src/predict.py is touched or retrained.

(a) answers "is the other team's 95% a better model, or a looser protocol?"
    Same winning model per target, same feature set, same seeds; the ONLY thing
    that changes is grouped folds vs ungrouped folds.
(b) tests the combination train.py never built: ESM embeddings alongside the
    full hand-made feature set. Judged across four fold seeds, not one.
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402

T0 = time.time()
SEEDS = (42, 7, 13, 2026)


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def score(X, y, model, splits):
    p = np.zeros(len(y))
    for tr_i, te_i in splits:
        p[te_i] = clone(model).fit(X[tr_i], y[tr_i]).predict_proba(X[te_i])[:, 1]
    return cv.metrics(y, p)


def folds(X, y, groups, seed, grouped):
    if grouped:
        return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                    .split(X, y, groups))
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y))


def prep(name, csv, feature_groups):
    tr, _, _ = load(str(REPO / "participant_datasets/initial" / csv))
    seqs = tr["sequence"].tolist()
    y = tr["label"].to_numpy()
    groups = leak.similarity_groups(seqs, 0.6)
    X = pd.concat([features.build(seqs, (g,)) for g in feature_groups], axis=1)
    log(f"{name}: {len(seqs)} seqs, {X.shape[1]} features, "
        f"{len(np.unique(groups))} clusters (largest {np.bincount(groups).max()})")
    return seqs, X.to_numpy(dtype=np.float32), y, groups


# ---------------------------------------------------------------- (a) protocol
print("=" * 78)
print("(a) GROUPED vs PLAIN RANDOM SPLIT - same model, same features, same seeds")
print("=" * 78)

TARGETS = {
    "anticancer": ("anticancer_train.csv", ("aac", "dpc", "phys"), "rf"),
    "hemolytic": ("hemolytic_train.csv", ("aac", "phys"), "lgbm"),
}

data = {}
rows = []
for name, (csv, fg, mname) in TARGETS.items():
    seqs, X, y, groups = prep(name, csv, fg)
    data[name] = (seqs, X, y, groups)
    model = cv.make_models(y, (mname,))[mname]
    for grouped in (True, False):
        accs = []
        for seed in SEEDS:
            accs.append(score(X, y, model, folds(X, y, groups, seed, grouped))["accuracy"])
        rows.append({"target": name, "model": mname,
                     "protocol": "grouped" if grouped else "random",
                     "mean_acc": float(np.mean(accs)), "min": float(np.min(accs)),
                     "max": float(np.max(accs)), "seed42": accs[0]})
        log(f"{name:11s} {('grouped' if grouped else 'random '):8s} "
            f"mean {np.mean(accs):.4f}  range {np.min(accs):.4f}-{np.max(accs):.4f}  "
            f"(seed42 {accs[0]:.4f})")

pa = pd.DataFrame(rows)
print("\n" + pa.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
for name in TARGETS:
    g = pa[(pa.target == name) & (pa.protocol == "grouped")]["mean_acc"].iloc[0]
    r = pa[(pa.target == name) & (pa.protocol == "random")]["mean_acc"].iloc[0]
    log(f"{name}: protocol alone is worth {r - g:+.4f} ({r:.4f} random vs {g:.4f} grouped)")

# ------------------------------------------------------------------- (b) ESM
print("\n" + "=" * 78)
print("(b) ESM-2 (8M, cached) + aac+dpc+phys  vs  aac+dpc+phys  - grouped, 4 seeds")
print("=" * 78)

seqs, Xh, y, groups = data["anticancer"]
from peptide_classifier.esm import embed  # noqa: E402
E = embed(seqs)
log(f"ESM embeddings {E.shape}")
Xe = np.hstack([E, Xh]).astype(np.float32)
log(f"combined matrix {Xe.shape}")

rf = cv.make_models(y, ("rf",))["rf"]
brows = []
for seed in SEEDS:
    sp = folds(Xh, y, groups, seed, True)
    a = score(Xh, y, rf, sp)
    b = score(Xe, y, rf, sp)
    brows.append({"fold_seed": seed, "hand_only": a["accuracy"], "esm+hand": b["accuracy"],
                  "delta": b["accuracy"] - a["accuracy"],
                  "esm_sens": b["sensitivity"], "esm_spec": b["specificity"],
                  "esm_AUC": b["AUC"], "esm_MCC": b["MCC"]})
    log(f"seed {seed:5d}: hand {a['accuracy']:.4f}  esm+hand {b['accuracy']:.4f}  "
        f"delta {b['accuracy'] - a['accuracy']:+.4f}")

pb = pd.DataFrame(brows)
print("\n" + pb.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
log(f"mean delta {pb['delta'].mean():+.4f}  (sd {pb['delta'].std():.4f})  "
    f"wins {int((pb['delta'] > 0).sum())}/{len(pb)}")

out = Path(__file__).resolve().parent
pa.to_csv(out / "exp_protocol_results.csv", index=False)
pb.to_csv(out / "exp_esm_results.csv", index=False)
log(f"written to {out}")
