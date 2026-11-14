"""
train.py — train the two classifiers and save them for predict.py.

OWNED BY: A (Sayf). Run from the repo root:  python src/train.py

What it does, in order:
  1. load one training file (anticancer or hemolytic — they are SEPARATE datasets)
  2. turn every sequence into numbers with features.py
  3. hold back 20% of the peptides so we can measure honestly
  4. train a few candidate models and compare them on the held-back part
  5. pick the decision threshold that maximises accuracy
  6. save {model, columns, threshold} to models/<name>.pkl

Why accuracy?  The judges rank on "average accuracy of the two classifiers"
first. Both training sets are exactly 50/50, so accuracy is a fair metric here
and a coin flip scores 50%. That's the number to beat.
"""

import sys
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_feature_matrix

RANDOM_STATE = 42          # fixed so results are reproducible run to run
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def candidate_models():
    """
    The models we compare. All three are standard and explainable.

    StandardScaler puts every feature on the same scale first. Needed because
    'length' runs 3-98 while 'frac_positive' runs 0-1, and logistic regression
    and SVM would otherwise let the big numbers dominate.
    """
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,     # don't let a leaf memorise a single peptide
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def best_threshold(y_true, proba):
    """
    Find the probability cutoff that gives the highest accuracy.

    Default is 0.5, but that is only optimal if the model's probabilities are
    perfectly calibrated. Sweeping the cutoff is free accuracy. The challenge
    doc calls this out explicitly: two independent thresholds to tune.
    """
    grid = np.linspace(0.05, 0.95, 181)
    scores = [accuracy_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))]), float(max(scores))


def train_one(name: str):
    path = os.path.join(REPO, "participant_datasets", "initial", f"{name}_train.csv")
    df = pd.read_csv(path)

    X = build_feature_matrix(df["sequence"])
    y = df["label"].astype(int).values

    # stratify keeps the 50/50 balance in both halves
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n{'='*62}\n{name.upper()}   {len(df)} peptides, {X.shape[1]} features")
    print(f"train {len(y_tr)} / validation {len(y_val)}\n")
    print(f"{'model':<22}{'train acc':>11}{'val acc':>10}{'val AUC':>10}{'gap':>8}")
    print("-" * 62)

    results = {}
    for mname, model in candidate_models().items():
        model.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, model.predict(X_tr))
        val_proba = model.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, (val_proba >= 0.5).astype(int))
        val_auc = roc_auc_score(y_val, val_proba)
        gap = tr_acc - val_acc          # big gap = memorising, not learning
        print(f"{mname:<22}{tr_acc:>11.3f}{val_acc:>10.3f}{val_auc:>10.3f}{gap:>8.3f}")
        results[mname] = (model, val_acc, val_auc, val_proba, tr_acc)

    # pick on validation accuracy, since that is the judges' primary metric
    winner = max(results, key=lambda m: results[m][1])
    model, val_acc, val_auc, val_proba, tr_acc = results[winner]

    thr, thr_acc = best_threshold(y_val, val_proba)
    pred = (val_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
    sensitivity = tp / (tp + fn)        # of the real positives, how many we caught
    specificity = tn / (tn + fp)        # of the real negatives, how many we correctly rejected

    print(f"\nchosen: {winner}")
    print(f"  threshold 0.50 -> val acc {val_acc:.3f}")
    print(f"  threshold {thr:.2f} -> val acc {thr_acc:.3f}   (AUC {val_auc:.3f})")
    print(f"  sensitivity {sensitivity:.3f}  specificity {specificity:.3f}")
    if abs(sensitivity - specificity) > 0.15:
        print("  ** lopsided: good at one class, weak at the other - check this **")
    if tr_acc - thr_acc > 0.15:
        print("  ** large train/validation gap: likely memorising **")

    # Refit the winner on ALL the data before saving. The split existed to
    # measure honestly; now that the choice is made, use every peptide we have.
    model.fit(X, y)

    bundle = {
        "model": model,
        "columns": list(X.columns),     # so predict.py can line columns up identically
        "threshold": thr,
        "meta": {
            "name": name,
            "algorithm": winner,
            "val_accuracy": round(thr_acc, 4),
            "val_auc": round(val_auc, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "n_train": int(len(df)),
        },
    }
    out = os.path.join(REPO, "models", f"{name}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(bundle, out)
    print(f"  saved -> models/{name}.pkl")
    return bundle["meta"]


if __name__ == "__main__":
    metas = [train_one("anticancer"), train_one("hemolytic")]
    avg = np.mean([m["val_accuracy"] for m in metas])
    print(f"\n{'='*62}")
    print(f"AVERAGE VALIDATION ACCURACY: {avg:.3f}   <- the judges' primary metric")
    print(f"{'='*62}")
