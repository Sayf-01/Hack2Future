"""One command, whole pipeline.

    python run.py --train data/train.csv [--test data/test.csv] [--quick] [--no-esm]

Outputs land in out/: data_report.txt, cv_results.csv, predictions.csv,
bio_boxplots.png, bio_class_comparison.csv, shap_summary.png,
shap_importance.csv, roc.png, umap_esm.png, umap3d_esm.csv, best_model.joblib
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
from pep import cv, explain, features, leak  # noqa: E402
from pep.load import load, report  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test")
    ap.add_argument("--seq-col")
    ap.add_argument("--label-col")
    ap.add_argument("--id-col")
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2 models, no dipeptides: first score in ~a minute")
    ap.add_argument("--sim-threshold", type=float, default=0.6, help="3-mer Jaccard for near-duplicate groups")
    ap.add_argument("--no-groups", action="store_true", help="plain stratified CV (to show the leakage gap)")
    ap.add_argument("--target", help="name for this model in the submission columns")
    ap.add_argument("--rank-by", default="accuracy", choices=["accuracy", "MCC", "AUC"],
                    help="metric used to pick the winning model and tune its threshold")
    ap.add_argument("--out", default=str(ROOT / "out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # 1. Load and describe ------------------------------------------------
    tr, _, info = load(a.train, a.seq_col, a.label_col, a.id_col)
    log(f"train: seq column '{info['seq_col']}', label column '{info['label_col']}', map {info['label_map']}")
    if "label" not in tr.columns:
        sys.exit("No label column found in train. Pass --label-col.")
    rep = [report(tr, "train")]
    te = None
    if a.test:
        te, _, tinfo = load(a.test, a.seq_col or info["seq_col"], a.label_col, a.id_col)
        rep.append(report(te, "test"))
    print("\n".join(rep))

    # 2. Leakage check ------------------------------------------------------
    seqs = tr["sequence"].tolist()
    y = tr["label"].to_numpy()
    groups = leak.similarity_groups(seqs, a.sim_threshold)
    msg = (f"near-duplicate clusters in train (3-mer Jaccard >= {a.sim_threshold}): "
           f"{len(np.unique(groups))} clusters for {len(seqs)} sequences, "
           f"largest {np.bincount(groups).max()}")
    log(msg)
    rep.append(msg)
    if te is not None:
        s = leak.summary(leak.cross_max_similarity(te["sequence"].tolist(), seqs))
        log(s)
        rep.append(s)
    (out / "data_report.txt").write_text("\n".join(rep), encoding="utf-8")
    if a.no_groups:
        groups = None

    # 3. Features -----------------------------------------------------------
    all_seqs = seqs + (te["sequence"].tolist() if te is not None else [])
    n = len(seqs)
    phys = features.build(all_seqs, ("phys",))
    aac = features.build(all_seqs, ("aac",))
    sets = {"phys": phys, "aac+phys": pd.concat([aac, phys], axis=1)}
    if not a.quick:
        dpc = features.build(all_seqs, ("dpc",))
        sets["aac+dpc+phys"] = pd.concat([aac, dpc, phys], axis=1)
    log(f"hand-made features built: {', '.join(f'{k} {v.shape[1]}' for k, v in sets.items())}")
    E = None
    if not a.no_esm:
        from pep.esm import embed
        E = embed(all_seqs)
        Edf = pd.DataFrame(E, columns=[f"esm_{i}" for i in range(E.shape[1])])
        sets["esm"] = Edf
        sets["esm+phys"] = pd.concat([Edf, phys], axis=1)
        log(f"ESM-2 embeddings: {E.shape}")

    # 4. Cross-validation ---------------------------------------------------
    which = ("logreg", "lgbm") if a.quick else ("logreg", "rf", "xgb", "lgbm")
    results, oofs = [], {}
    for fs, X in sets.items():
        r, o = cv.run_cv(X.iloc[:n].to_numpy(), y, cv.make_models(y, which), groups)
        r.insert(0, "features", fs)
        results.append(r)
        for m, p in o.items():
            oofs[(fs, m)] = p
        best = r.sort_values(a.rank_by).iloc[-1]
        log(f"CV {fs:13s} best {best['model']:6s}  acc {best['accuracy']:.3f}  "
            f"MCC {best['MCC']:.3f}  AUC {best['AUC']:.3f}")
    # Ranked on accuracy because that is the organizers' primary metric
    # (averaged over the two classifiers). AUC breaks ties.
    res = pd.concat(results).sort_values([a.rank_by, "AUC"], ascending=False).reset_index(drop=True)
    res.to_csv(out / "cv_results.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print("\n" + res.to_string(index=False) + "\n")

    top = res.iloc[0]
    fs, mname = top["features"], top["model"]
    metric = a.rank_by if a.rank_by in ("accuracy", "MCC") else "accuracy"
    thr, thr_score = cv.best_threshold(y, oofs[(fs, mname)], metric)
    tuned = cv.metrics(y, oofs[(fs, mname)], thr)
    log(f"BEST: {fs} + {mname}  accuracy {top['accuracy']:.3f}  MCC {top['MCC']:.3f}  AUC {top['AUC']:.3f} (at 0.5)")
    log(f"  tuned threshold {thr:.3f} -> {metric} {thr_score:.3f} | "
        f"sens {tuned['sensitivity']:.3f}  spec {tuned['specificity']:.3f}  (slightly optimistic)")
    if min(tuned["sensitivity"], tuned["specificity"]) < 0.3:
        log("  WARNING: one class is being nearly ignored at this threshold. "
            "Good average accuracy with a collapsed class is the shortcut the organizers call out.")

    # 5. Fit on all train, predict test -------------------------------------
    X = sets[fs]
    model = cv.make_models(y, (mname,))[mname].fit(X.iloc[:n].to_numpy(), y)
    joblib.dump({"model": model, "features": fs, "columns": list(X.columns), "threshold": thr,
                 "target": a.target or info["label_col"] or out.name,
                 "cv_accuracy": float(top["accuracy"]), "cv_mcc": float(top["MCC"]),
                 "cv_auc": float(top["AUC"]), "cv_sensitivity": float(tuned["sensitivity"]),
                 "cv_specificity": float(tuned["specificity"]), "tuned_metric": metric},
                out / "best_model.joblib")
    log(f"saved {out / 'best_model.joblib'}")
    if te is not None:
        p = model.predict_proba(X.iloc[n:].to_numpy())[:, 1]
        pred = te[["id", "sequence"]].copy()
        pred["prob"] = p
        pred["pred_0.5"] = (p >= 0.5).astype(int)
        pred["pred_tuned"] = (p >= thr).astype(int)
        pred.to_csv(out / "predictions.csv", index=False)
        log(f"predictions.csv written ({len(pred)} rows, {pred['pred_0.5'].mean():.1%} predicted positive)")
        if "label" in te.columns:
            m = cv.metrics(te["label"].to_numpy(), p)
            log("TEST (labelled): " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))

    # 6. Explain --------------------------------------------------------------
    bio = explain.class_comparison(phys.iloc[:n].reset_index(drop=True), y, out)
    log("top separating properties: " + ", ".join(
        f"{r.feature} ({'+' if r['diff'] > 0 else '-'})" for _, r in bio.head(6).iterrows()))
    Xe = sets["aac+phys"].iloc[:n]
    lg = cv.make_models(y, ("lgbm",))["lgbm"].fit(Xe, y)
    imp = explain.shap_summary(lg, Xe, out)
    log("SHAP top 8 (aac+phys LightGBM): " + ", ".join(imp["feature"].head(8)))
    curves = {}
    for f in sets:
        sub = res[res["features"] == f].iloc[0]
        curves[f"{f}/{sub['model']}"] = oofs[(f, sub["model"])]
    explain.roc_plot(y, curves, out)
    if E is not None:
        explain.umap_plot(E[:n], y, out)
    log(f"done. outputs in {out}")


if __name__ == "__main__":
    main()
