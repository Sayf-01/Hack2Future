"""PRIORITY 2: bounded anticancer-only experiment.

Apples-to-apples with the saved RF: identical similarity groups, identical
StratifiedGroupKFold(5, shuffle=True, random_state=42) splits, identical
pooled out-of-fold scoring. Every scaler and feature selector lives INSIDE
the pipeline, so it is fit on the training folds only -- no leakage.

Nothing here touches models/ or frozen_v1/.
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
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline       # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC                 # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def rf(seed=42):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=4, random_state=seed)


def svc(kernel="rbf", C=1.0, gamma="scale"):
    return SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced",
               probability=True, random_state=42)


def main():
    tr, _, info = load(str(REPO / "participant_datasets/initial/anticancer_train.csv"))
    seqs = tr["sequence"].tolist()
    y = tr["label"].to_numpy()
    log(f"{len(seqs)} anticancer sequences, {y.mean():.1%} positive")

    groups = leak.similarity_groups(seqs, 0.6)
    log(f"{len(np.unique(groups))} similarity clusters, largest {np.bincount(groups).max()}")

    aac = features.build(seqs, ("aac",))
    dpc = features.build(seqs, ("dpc",))
    phys = features.build(seqs, ("phys",))
    X = pd.concat([aac, dpc, phys], axis=1)
    Xn = X.to_numpy(dtype=np.float32)
    log(f"feature matrix {Xn.shape} (aac+dpc+phys, same columns the saved model uses)")

    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                  .split(Xn, y, groups))

    cands = {
        "rf_baseline (saved model)": rf(),
        "svc_rbf C=1": Pipeline([("sc", StandardScaler()), ("m", svc("rbf", 1.0))]),
        "svc_rbf C=10": Pipeline([("sc", StandardScaler()), ("m", svc("rbf", 10.0))]),
        "svc_rbf C=100": Pipeline([("sc", StandardScaler()), ("m", svc("rbf", 100.0))]),
        "svc_linear C=0.1": Pipeline([("sc", StandardScaler()), ("m", svc("linear", 0.1))]),
        "svc_linear C=1": Pipeline([("sc", StandardScaler()), ("m", svc("linear", 1.0))]),
    }
    for k in (50, 100, 200):
        cands[f"kbest{k} + svc_rbf C=10"] = Pipeline([
            ("sc", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=k)),
            ("m", svc("rbf", 10.0))])
        cands[f"kbest{k} + rf"] = Pipeline([
            ("sel", SelectKBest(f_classif, k=k)),
            ("m", rf())])
    cands["rf-importance select + rf"] = Pipeline([
        ("sel", SelectFromModel(rf(seed=7), threshold="median")),
        ("m", rf())])

    rows = []
    for name, model in cands.items():
        t = time.time()
        p = np.zeros(len(y))
        fold_acc = []
        for tr_i, te_i in splits:
            m = clone(model).fit(Xn[tr_i], y[tr_i])
            p[te_i] = m.predict_proba(Xn[te_i])[:, 1]
            fold_acc.append(cv.metrics(y[te_i], p[te_i])["accuracy"])
        mt = cv.metrics(y, p)
        rows.append({"candidate": name, "accuracy": mt["accuracy"], "sensitivity": mt["sensitivity"],
                     "specificity": mt["specificity"], "AUC": mt["AUC"], "MCC": mt["MCC"],
                     "acc_fold_sd": float(np.std(fold_acc)),
                     "acc_fold_min": float(np.min(fold_acc)), "acc_fold_max": float(np.max(fold_acc)),
                     "secs": round(time.time() - t, 1)})
        log(f"{name:28s} acc {mt['accuracy']:.4f}  sens {mt['sensitivity']:.3f}  "
            f"spec {mt['specificity']:.3f}  AUC {mt['AUC']:.3f}  MCC {mt['MCC']:.3f}  "
            f"sd {np.std(fold_acc):.3f}  ({time.time() - t:.0f}s)")

    res = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    out = Path(__file__).resolve().parent / "exp_anticancer_results.csv"
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 4):
        print("\n" + res.to_string(index=False) + "\n")
    base = res.loc[res["candidate"].str.startswith("rf_baseline"), "accuracy"].iloc[0]
    top = res.iloc[0]
    log(f"baseline RF on these folds: {base:.4f}  (saved model reports 0.7386)")
    log(f"best candidate: {top['candidate']} at {top['accuracy']:.4f}  "
        f"-> delta {top['accuracy'] - base:+.4f} (fold sd {top['acc_fold_sd']:.3f})")
    log(f"results written to {out}")


if __name__ == "__main__":
    main()
