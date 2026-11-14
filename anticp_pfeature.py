"""
not super reliable, pfeature used to be a command line library but now its a package
"""

import re
import sys
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

# Path is relative to the folder you run the script from (the top-level
# challenge folder). Swap in hemolytic_train.csv to train the hemolytic model.
TRAIN_PATH = "participant_datasets/initial/anticancer_train.csv"

PFEATURE_AAC = None
PFEATURE_DPC = None
PFEATURE_MODE = None

try:
    # Newer pip package style
    from Pfeature.pfeature import aac_wp, dpc_wp
    PFEATURE_AAC, PFEATURE_DPC = aac_wp, dpc_wp
    PFEATURE_MODE = "Pfeature.pfeature (file-based wp functions)"
except ImportError:
    pass

if PFEATURE_MODE is None:
    try:
        # Alternate lowercase package style
        from pfeature.pfeature import aac_wp, dpc_wp
        PFEATURE_AAC, PFEATURE_DPC = aac_wp, dpc_wp
        PFEATURE_MODE = "pfeature.pfeature (file-based wp functions)"
    except ImportError:
        pass

if PFEATURE_MODE is None:
    print(
        "import failed"
    )

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)


def build_feature_matrix_pfeature_file_based(df: pd.DataFrame) -> pd.DataFrame:
    import tempfile, os
    seqs = [clean_sequence(s) or "A" for s in df["sequence"]]
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        with open(fasta_path, "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">seq{i}\n{s}\n")

        aac_out = os.path.join(tmpdir, "aac.csv")
        dpc_out = os.path.join(tmpdir, "dpc.csv")
        PFEATURE_AAC(fasta_path, aac_out)
        PFEATURE_DPC(fasta_path, dpc_out)

        aac_df = pd.read_csv(aac_out)
        dpc_df = pd.read_csv(dpc_out)

    aac_df.columns = [f"aac_{c}" for c in aac_df.columns]
    dpc_df.columns = [f"dpc_{c}" for c in dpc_df.columns]

    X = pd.concat([aac_df.reset_index(drop=True), dpc_df.reset_index(drop=True)], axis=1)
    X["length"] = [len(s) for s in seqs]
    X["net_charge"] = [s.count('K') + s.count('R') - s.count('D') - s.count('E') for s in seqs]
    return X.fillna(0.0)


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
    if PFEATURE_MODE is None:
        print("Pfeature import failed - aborting. See message above for fallback options.")
        sys.exit(1)

    print(f"Using Pfeature import mode: {PFEATURE_MODE}")
    print("Loading data...")
    df = load_data(TRAIN_PATH)
    print(f"Loaded {len(df)} sequences")

    print("Extracting Pfeature features (AAC + DPC)...")
    X = build_feature_matrix_pfeature_file_based(df)
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
    joblib.dump(final_model, "final_model_pfeature.joblib")
    joblib.dump(final_scaler, "final_scaler_pfeature.joblib")
    joblib.dump(list(X.columns), "feature_columns_pfeature.joblib")
    print("Saved final_model_pfeature.joblib")


if __name__ == "__main__":
    main()
