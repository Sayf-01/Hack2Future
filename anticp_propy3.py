"""
"""

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import joblib

try:
    from propy.PyPro import GetProDes
except ImportError:
    raise ImportError(
        "propy3 not installed or import path wrong. Try: pip install propy3\n"
        "If it still fails, use build_your_own_features_example.py (manual features) instead."
    )

# Path is relative to the folder you run the script from (the top-level
# challenge folder). Swap in hemolytic_train.csv to train the hemolytic model.
TRAIN_PATH = "participant_datasets/initial/anticancer_train.csv"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)


def extract_features_propy(seq: str) -> dict:
    seq = clean_sequence(seq)
    if len(seq) < 2:
        seq = "AA"  # guard: propy needs at least a couple residues for DPC

    desc = GetProDes(seq)
    feats = {}

    # Amino acid composition (20 features)
    feats.update({f"aac_{k}": v for k, v in desc.GetAAComp().items()})

    # Dipeptide composition (400 features)
    feats.update({f"dpc_{k}": v for k, v in desc.GetDPComp().items()})

    # CTD: Composition/Transition/Distribution of physicochemical properties
    # (charge, hydrophobicity, polarity, etc.) - propy's signature descriptor set
    try:
        feats.update({f"ctd_{k}": v for k, v in desc.GetCTD().items()})
    except Exception:
        pass  # some propy3 versions/sequences can throw on very short peptides

    # simple length/charge, since propy doesn't give these directly
    feats["length"] = len(seq)
    feats["net_charge"] = seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')

    return feats


def build_feature_matrix(sequences) -> pd.DataFrame:
    return pd.DataFrame([extract_features_propy(s) for s in sequences]).fillna(0.0)


def load_data(path=TRAIN_PATH):
    df = pd.read_csv(path)
    assert "sequence" in df.columns and "label" in df.columns
    return df


def evaluate_model(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return acc.mean(), acc.std(), auc.mean()


def main():
    print("Loading data...")
    df = load_data(TRAIN_PATH)
    print(f"Loaded {len(df)} sequences")

    print("Extracting propy3 features (AAC + DPC + CTD)...")
    X = build_feature_matrix(df["sequence"])
    y = df["label"].values
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        "SVM_RBF": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True),
    }

    print("\n--- 5-fold CV (model selection) ---")
    best_name, best_score = None, -1
    Xs_full = StandardScaler().fit_transform(X)
    for name, model in candidates.items():
        acc_mean, acc_std, auc_mean = evaluate_model(model, Xs_full, y)
        print(f"{name:20s} | accuracy: {acc_mean:.4f} (+/-{acc_std:.4f}) | AUC: {auc_mean:.4f}")
        if acc_mean > best_score:
            best_score, best_name = acc_mean, name

    print(f"\nBest model: {best_name} ({best_score:.4f})")
    best_model = candidates[best_name]
    best_model.fit(X_train_s, y_train)
    y_pred = best_model.predict(X_test_s)
    y_proba = best_model.predict_proba(X_test_s)[:, 1]

    print("\n--- Holdout evaluation ---")
    print(f"Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC        : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"MCC        : {matthews_corrcoef(y_test, y_pred):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Sensitivity: {tp/(tp+fn):.4f}")
    print(f"Specificity: {tn/(tn+fp):.4f}")
    print("\n", classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    final_scaler = StandardScaler().fit(X)
    final_model = candidates[best_name]
    final_model.fit(final_scaler.transform(X), y)
    joblib.dump(final_model, "final_model_propy.joblib")
    joblib.dump(final_scaler, "final_scaler_propy.joblib")
    joblib.dump(list(X.columns), "feature_columns_propy.joblib")
    print("Saved final_model_propy.joblib")


if __name__ == "__main__":
    main()
