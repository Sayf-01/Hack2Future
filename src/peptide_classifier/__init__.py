"""Peptide bioactivity classification.

Predicts anticancer activity and hemolytic toxicity from an amino acid sequence.

Modules:
    load      reading sequence datasets in whatever shape they arrive in
    features  hand-built, explainable features (AAC, dipeptides, physicochemical)
    leak      near-duplicate detection and similarity-based clustering
    cv        models, metrics and similarity-grouped cross-validation
    explain   class comparisons, SHAP, ROC and UMAP plots
    esm       optional protein language model embeddings (not used by the
              shipped models - see docs/methodology.md)
"""

__version__ = "1.0.0"
