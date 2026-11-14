"""Models, metrics and (grouped) cross-validation."""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
N_JOBS = 4   # the laptop is short on RAM; don't fan out to all 16 threads


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "MCC": matthews_corrcoef(y, pred),
        "AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "F1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }


def make_models(y, which=("logreg", "rf", "xgb", "lgbm"), seed=42):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    spw = float((y == 0).sum()) / max((y == 1).sum(), 1)
    all_models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")),
        "rf": RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                     class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=seed),
        "xgb": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.5, scale_pos_weight=spw, tree_method="hist",
                             eval_metric="logloss", n_jobs=N_JOBS, random_state=seed),
        "lgbm": LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, class_weight="balanced",
                               n_jobs=N_JOBS, random_state=seed, verbose=-1),
    }
    return {k: all_models[k] for k in which}


def run_cv(X, y, models, groups=None, n_splits=5, seed=42):
    """Returns (results DataFrame, {model: out-of-fold probabilities})."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))
    rows, oof = [], {}
    for name, model in models.items():
        p = np.zeros(len(y))
        fold_scores = []
        for tr, te in splits:
            m = clone(model).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
            fold_scores.append(metrics(y[te], p[te])["MCC"])
        oof[name] = p
        r = {"model": name, **metrics(y, p), "MCC_fold_sd": float(np.std(fold_scores))}
        rows.append(r)
    return pd.DataFrame(rows), oof


SCORERS = {"accuracy": accuracy_score, "MCC": matthews_corrcoef}


def best_threshold(y, p, metric="accuracy", tol=0.01, min_gain=0.01):
    """Threshold maximising `metric` on out-of-fold predictions.

    The organizers rank on plain accuracy averaged over the two classifiers,
    so accuracy is the default. The value returned is slightly optimistic
    because the threshold was chosen on the same predictions it is scored on.
    """
    fn = SCORERS[metric]
    grid = np.linspace(0.05, 0.95, 181)
    scores = np.array([fn(y, (p >= t).astype(int)) for t in grid])

    # Among thresholds statistically tied with the best (within `tol`), take the
    # one with the most balanced sensitivity/specificity. The organizers warn
    # that a classifier good at only one of those "isn't actually working", and
    # a 0.2% accuracy gain bought by collapsing one class is noise that costs
    # real true positives (their rank-2 tiebreaker).
    best = scores.max()
    tied = np.flatnonzero(scores >= best - tol)
    balance = []
    for i in tied:
        m = metrics(y, p, grid[i])
        balance.append(min(m["sensitivity"], m["specificity"]))
    i = int(tied[int(np.argmax(balance))])

    # Both training sets are exactly 50/50, so 0.5 is the principled default and
    # a threshold tuned on out-of-fold predictions is optimistic by construction.
    # Only move off 0.5 if tuning earns a gain bigger than the fold-to-fold noise.
    at_half = fn(y, (p >= 0.5).astype(int))
    if scores[i] - at_half < min_gain:
        return 0.5, float(at_half)
    return float(grid[i]), float(scores[i])
