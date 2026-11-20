"""Inference: saved models + a file of peptide sequences -> predictions CSV.

    python scripts/predict.py --input examples/sample_peptides.csv \
                              --output predictions.csv

Nothing is trained here. This script loads the models saved by
``scripts/train.py``, rebuilds exactly the feature columns each model was
trained on, and writes one output row per input sequence.

Output columns, in this order:

    sequence, anticancer_probability, anticancer_prediction,
    hemolytic_probability, hemolytic_prediction, verdict

Two details that are easy to get wrong:
  * the key is ``sequence``, not ``id`` — the input file may carry both, the
    output carries only the sequence
  * the sequence is written out exactly as it appeared in the input, not
    upper-cased or stripped, so the output still joins against the input
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from peptide_classifier import features  # noqa: E402
from peptide_classifier.load import load  # noqa: E402

T0 = time.time()

COLUMNS = ["sequence", "anticancer_probability", "anticancer_prediction",
           "hemolytic_probability", "hemolytic_prediction", "verdict"]

# (anticancer, hemolytic) -> verdict. Spelled exactly as the organizers spell it.
VERDICT = {
    (1, 0): "Selective",
    (1, 1): "Works but toxic",
    (0, 1): "Toxic and doesn't work",
    (0, 0): "Does nothing",
}


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def build_matrix(seqs, feature_set, columns):
    """Rebuild the exact feature columns a saved model was trained on."""
    parts, groups = [], feature_set.split("+")
    if "esm" in groups:
        from pep.esm import embed
        E = embed(seqs)
        parts.append(pd.DataFrame(E, columns=[f"esm_{i}" for i in range(E.shape[1])]))
    for g in ("aac", "dpc", "phys"):
        if g in groups:
            parts.append(features.build(seqs, (g,)))
    X = pd.concat(parts, axis=1)
    missing = [c for c in columns if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    if missing:
        log(f"WARNING: {len(missing)} expected columns were missing and filled with 0")
    return X[columns].fillna(0).to_numpy(dtype=np.float32)


def score(model_dir, seqs, use_tuned=True):
    f = Path(model_dir) / "best_model.joblib"
    if not f.exists():
        sys.exit(f"MISSING {f}\nTrain and save that model first — this script never trains.")
    saved = joblib.load(f)
    X = build_matrix(seqs, saved["features"], saved["columns"])
    p = saved["model"].predict_proba(X)[:, 1]
    thr = saved["threshold"] if use_tuned else 0.5
    log(f"{f.parent.name}: {saved['features']} | threshold {thr:.3f} | "
        f"CV accuracy {saved.get('cv_accuracy', float('nan')):.3f} | "
        f"{(p >= thr).mean():.1%} positive")
    return p, thr


def main():
    ap = argparse.ArgumentParser(
        description="Score peptide sequences with the saved anticancer and hemolytic models.")
    ap.add_argument("--input", required=True,
                    help="CSV/TSV/Excel/FASTA holding the peptide sequences to score")
    ap.add_argument("--output", default=str(ROOT / "predictions.csv"),
                    help="where to write the predictions CSV")
    ap.add_argument("--anticancer", default=str(ROOT / "models" / "anticancer"),
                    help="directory holding the anticancer best_model.joblib")
    ap.add_argument("--hemolytic", default=str(ROOT / "models" / "hemolytic"),
                    help="directory holding the hemolytic best_model.joblib")
    ap.add_argument("--seq-col", help="name of the sequence column (auto-detected if omitted)")
    ap.add_argument("--half", action="store_true",
                    help="force a 0.5 cutoff instead of each model's saved threshold")
    ap.add_argument("--template", help="a CSV whose header the output must match, e.g. "
                                       "examples/output_schema.csv")
    a = ap.parse_args()

    if not Path(a.input).exists():
        sys.exit(f"Input file not found: {a.input}")
    try:
        df, _, info = load(a.input, a.seq_col)
    except ValueError as e:
        sys.exit(f"Could not read sequences from {a.input}: {e}")
    if len(df) == 0:
        sys.exit(f"No sequences found in {a.input}")
    log(f"{len(df)} sequences from column '{info['seq_col']}' of {a.input}")
    seqs = df["sequence"].tolist()

    pa, ta = score(a.anticancer, seqs, not a.half)
    ph, th = score(a.hemolytic, seqs, not a.half)
    ca = (pa >= ta).astype(int)
    ch = (ph >= th).astype(int)

    sub = pd.DataFrame({
        "sequence": df["sequence_raw"],          # exactly as it appeared in the input
        "anticancer_probability": np.round(pa, 6),
        "anticancer_prediction": ca,
        "hemolytic_probability": np.round(ph, 6),
        "hemolytic_prediction": ch,
        "verdict": [VERDICT[(int(x), int(y))] for x, y in zip(ca, ch)],
    })[COLUMNS]

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)

    counts = sub["verdict"].value_counts()
    log(f"wrote {out}  ({len(sub)} rows)")
    for v in VERDICT.values():
        log(f"  {v:24s} {counts.get(v, 0):5d}  ({counts.get(v, 0) / len(sub):6.1%})")

    # A problem means the file is malformed and the script exits non-zero.
    # A warning is worth a look but never blocks the write.
    problems, warnings = [], []
    if list(sub.columns) != COLUMNS:
        problems.append(f"column order is {list(sub.columns)}")
    if len(sub) != len(df):
        problems.append(f"{len(sub)} rows out for {len(df)} rows in")
    if sub["sequence"].isna().any() or (sub["sequence"].astype(str).str.len() == 0).any():
        problems.append("empty sequence cell")
    if sub[["anticancer_probability", "hemolytic_probability"]].isna().any().any():
        problems.append("NaN probabilities")
    for c in ("anticancer_probability", "hemolytic_probability"):
        if not sub[c].between(0, 1).all():
            problems.append(f"{c} outside 0-1")
    for c in ("anticancer_prediction", "hemolytic_prediction"):
        if not sub[c].isin([0, 1]).all():
            problems.append(f"{c} is not strictly 0/1")
    if not sub["verdict"].isin(list(VERDICT.values())).all():
        problems.append("a verdict string is not one of the four allowed values")
    if counts.get("Selective", 0) == len(sub):
        problems.append("every row is Selective — both models have collapsed to one class")

    d = int(sub["sequence"].duplicated().sum())
    if d:
        warnings.append(f"{d} repeated sequences in the input — kept, one output row per input row")
    for c, lo in (("anticancer_prediction", 0.02), ("hemolytic_prediction", 0.02)):
        if sub[c].mean() < lo or sub[c].mean() > 1 - lo:
            warnings.append(f"{c} is {sub[c].mean():.1%} positive — one class is nearly never predicted")
    if a.template:
        tmpl = pd.read_csv(a.template)
        if list(tmpl.columns) != COLUMNS:
            problems.append(f"TEMPLATE disagrees: template columns are {list(tmpl.columns)}")
        else:
            log(f"format matches {a.template}")
    print()
    for w in warnings:
        print(f"  warning: {w}")
    if problems:
        print("FAILED — the output is malformed:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    log("format checks passed")
    print(sub.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
