"""Plots and the biology story: class comparisons, SHAP, ROC, UMAP."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve

KEY = ["length", "charge_pH7", "hydrophobicity", "hydrophobic_moment", "frac_KR", "boman", "pI", "frac_aromatic"]


def class_comparison(phys_df, y, out_dir):
    """Which biological properties actually separate the classes? (Mann-Whitney U per feature)"""
    y = np.asarray(y)
    rows = []
    for c in phys_df.columns:
        a, b = phys_df[c][y == 1], phys_df[c][y == 0]
        try:
            p = mannwhitneyu(a, b).pvalue
        except ValueError:
            p = np.nan
        rows.append({"feature": c, "median_pos": a.median(), "median_neg": b.median(),
                     "diff": a.median() - b.median(), "p_value": p})
    t = pd.DataFrame(rows).sort_values("p_value")
    t.to_csv(out_dir / "bio_class_comparison.csv", index=False)

    cols = [c for c in KEY if c in phys_df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), cols):
        data = [phys_df[c][y == 0].dropna(), phys_df[c][y == 1].dropna()]
        ax.boxplot(data, tick_labels=["negative (0)", "positive (1)"], showfliers=False)
        pv = t.set_index("feature").loc[c, "p_value"]
        ax.set_title(f"{c}\n" + ("p < 1e-300" if pv == 0 else f"p = {pv:.1e}"))
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Physico-chemical properties by class")
    fig.tight_layout()
    fig.savefig(out_dir / "bio_boxplots.png", dpi=130)
    plt.close(fig)
    return t


def shap_summary(model, X, out_dir, max_rows=2000, top=20):
    import shap
    Xs = X.sample(min(len(X), max_rows), random_state=0)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    imp = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(sv).mean(0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "shap_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(sv, Xs, max_display=top, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close("all")
    return imp


def roc_plot(y, curves, out_dir):
    """curves: {label: oof probabilities}"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC, out-of-fold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=130)
    plt.close(fig)


def umap_plot(E, y, out_dir, seed=42):
    import umap
    y = np.asarray(y)
    r2 = umap.UMAP(n_components=2, random_state=seed).fit_transform(E)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, col in ((0, "#4c78a8"), (1, "#e45756")):
        m = y == lab
        ax.scatter(r2[m, 0], r2[m, 1], s=6, alpha=0.6, c=col, label=f"label {lab}")
    ax.set_title("ESM-2 embeddings (UMAP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "umap_esm.png", dpi=130)
    plt.close(fig)
    r3 = umap.UMAP(n_components=3, random_state=seed).fit_transform(E)
    pd.DataFrame({"x": r3[:, 0], "y": r3[:, 1], "z": r3[:, 2], "label": y}).to_csv(
        out_dir / "umap3d_esm.csv", index=False)
