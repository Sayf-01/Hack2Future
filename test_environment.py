"""
STEP 0 (participants) - Confirm your Python environment works BEFORE
touching the real data.

Runs a tiny end-to-end pipeline (load CSV -> hand-built features -> train/test
split -> LogisticRegression -> print metrics) on the fake demo data in
participant_datasets/demo/. If this prints two accuracy/AUC blocks without
errors, your environment (pandas, numpy, scikit-learn) and file paths are
fine and you're ready to work with the real datasets.

The demo sequences are SYNTHETIC (see make_demo_data.py) - they carry no
signal about the real anticancer/hemolytic tasks, so scoring well here says
nothing about your final model. It only proves the pipeline runs.

Usage:
    python test_environment.py
"""

import os
import re
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

DEMO_DIR = os.path.join("participant_datasets", "demo")
ANTICANCER_DEMO = os.path.join(DEMO_DIR, "anticancer_demo.csv")
HEMOLYTIC_DEMO = os.path.join(DEMO_DIR, "hemolytic_demo.csv")

AA = list("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)


def extract_features(seq: str) -> dict:
    """Minimal hand-built features: length, net charge, AA composition."""
    seq = clean_sequence(seq)
    if len(seq) == 0:
        seq = "A"
    n = len(seq)
    feats = {
        "length": n,
        "net_charge": (seq.count("K") + seq.count("R")) - (seq.count("D") + seq.count("E")),
    }
    for a in AA:
        feats[f"aac_{a}"] = seq.count(a) / n
    return feats


def build_feature_matrix(sequences) -> pd.DataFrame:
    return pd.DataFrame([extract_features(s) for s in sequences])


def run_pipeline(name: str, path: str) -> None:
    print(f"\n--- {name} ---")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run make_demo_data.py first (organizer step) "
              f"or check you're in the right directory.")
        sys.exit(1)

    df = pd.read_csv(path)
    assert "sequence" in df.columns and "label" in df.columns, f"{path} missing sequence/label columns"
    print(f"Loaded {len(df)} sequences")

    X = build_feature_matrix(df["sequence"])
    y = df["label"].values
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"Accuracy   : {acc:.4f}")
    print(f"AUC        : {auc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")


def main():
    print("Testing environment on fake/demo data (synthetic - no real signal)...")
    run_pipeline("Anticancer demo", ANTICANCER_DEMO)
    run_pipeline("Hemolytic demo", HEMOLYTIC_DEMO)
    print("\nIf both blocks printed metrics with no errors, your environment is ready.")
    print("Now go work with the real data in participant_datasets/initial/.")


if __name__ == "__main__":
    main()
