"""Final-round script: saved models + the test sequences -> submission CSV.

    python predict.py --input final_test_sequences.csv --output submissions/final.csv

NOTHING is trained here. It loads the models saved by run.py, rebuilds exactly
the feature columns each model was trained on, and writes the submission.

Submission columns, in this order (the judging script is strict):

    sequence, anticancer_probability, anticancer_prediction,
    hemolytic_probability, hemolytic_prediction, verdict

Two details that are easy to get wrong and fatal if you do:
  * the key is `sequence`, not `id` — the test file has both, the submission
    carries only the sequence
  * the sequence is written out EXACTLY as the organizers wrote it, not
    upper-cased or stripped, so it still matches their file on join
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pep import features  # noqa: E402
from pep.load import load  # noqa: E402

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="final_test_sequences.csv")
    ap.add_argument("--output", default="submissions/final.csv")
    ap.add_argument("--anticancer", default="models/anticancer", help="dir holding best_model.joblib")
    ap.add_argument("--hemolytic", default="models/hemolytic")
    ap.add_argument("--seq-col")
    ap.add_argument("--half", action="store_true", help="use 0.5 instead of the tuned thresholds")
    ap.add_argument("--template", help="submission_template.csv, to verify the format against")
    a = ap.parse_args()

    df, _, info = load(a.input, a.seq_col)
    log(f"{len(df)} sequences from column '{info['seq_col']}' of {a.input}")
    seqs = df["sequence"].tolist()

    pa, ta = score(a.anticancer, seqs, not a.half)
    ph, th = score(a.hemolytic, seqs, not a.half)
    ca = (pa >= ta).astype(int)
    ch = (ph >= th).astype(int)

    sub = pd.DataFrame({
        "sequence": df["sequence_raw"],          # exactly as the organizers wrote it
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

    # FATAL = the judging script would reject or misread the file.
    # WARNING = worth a look, but never a reason to block a submission at 15:30.
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
        problems.append("every row is Selective — the shortcut the organizers call out")

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
        print("FATAL — do not submit this file:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    log("format checks passed — safe to submit")
    print(sub.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
