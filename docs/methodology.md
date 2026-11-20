# Methodology

How the two classifiers were built, validated and frozen, and why each choice was
made. Every number here comes from the artifacts in `models/` and `experiments/`.

## The data

Two separate datasets, provided by the Hack2Future organizers, from different
sources. Each classifier is trained only on its own dataset.

| Dataset | Sequences | Class balance | Positive label means |
|---|---|---|---|
| `anticancer_train.csv` | 964 | exactly 50/50 | peptide kills cancer cells |
| `hemolytic_train.csv` | 618 | exactly 50/50 | peptide damages red blood cells |

30 sequences appear in both datasets. That is expected: a peptide can be both
active against cancer cells and toxic to red blood cells. The datasets are never
merged.

## Features

All features are hand-built and explainable. Nothing is learned from an external
pre-trained predictor.

**Amino acid composition (AAC), 20 columns.** The fraction of the peptide that is
each of the 20 standard amino acids.

**Dipeptide composition (DPC), 400 columns.** The fraction of adjacent residue
pairs that are each of the 400 possible pairs. This captures local sequence order
that AAC throws away.

**Physicochemical properties (phys), 18 columns.** Computed with the `peptides`
library: length, net charge at pH 7, charge per residue, isoelectric point,
Kyte-Doolittle hydrophobicity, hydrophobic moment, Boman index, aliphatic index,
instability index, molecular weight, and eight group fractions (cationic K/R,
anionic D/E, hydrophobic, aromatic, cysteine, proline, glycine, polar).

The biology behind the choice: cancer cell membranes carry more negative charge
than healthy cell membranes, so positively charged peptides are drawn to them and
disrupt them. Charge, hydrophobicity and amphipathicity are therefore the
properties expected to carry signal, and the hydrophobic moment measures
amphipathicity directly — how strongly the greasy residues cluster on one face of
a helix while the charged ones sit on the other.

### Which feature sets won

| Model | Feature set | Columns | Why |
|---|---|---|---|
| Anticancer | `aac+dpc+phys` | 438 | dipeptides were the single biggest gain |
| Hemolytic | `aac+phys` | 38 | dipeptides added nothing (0.937 either way) |

Dipeptide composition moved anticancer accuracy from 0.702 (phys alone) to 0.721
(aac+phys) to 0.739 (aac+dpc+phys). A shuffled-dipeptide control confirmed the
gain is real rather than an artifact of the extra columns: real sequences scored
0.739, sequences with their residues shuffled scored 0.711. Sequence order
carries genuine signal.

## Validation: similarity-grouped cross-validation

This is the methodological core of the project.

Peptide datasets are full of near-duplicates — the same peptide with one or two
substitutions, deposited separately. The largest near-duplicate cluster in the
anticancer set is **42 sequences**; in the hemolytic set it is 16. Under a plain
random split, a peptide lands in the training fold and its near-twin lands in the
test fold, so the model is rewarded for recognizing rather than generalizing, and
the reported accuracy is inflated.

`src/peptide_classifier/leak.py` clusters sequences by 3-mer Jaccard overlap at a
0.6 threshold, taking connected components of the "similar above threshold" graph.
`StratifiedGroupKFold` then keeps every cluster entirely inside one fold.

**What the protocol is worth, measured.** Same models, same features, same four
fold seeds; the only variable is grouped folds versus a plain random split:

| Target | Grouped | Plain random split | Difference |
|---|---|---|---|
| Anticancer | 0.7251 | 0.7516 | +2.65 pts |
| Hemolytic | 0.9438 | 0.9490 | +0.53 pts |
| **Average** | **0.8345** | **0.8503** | **+1.6 pts** |

A random-split number on this data carries roughly 1.6 points of near-duplicate
leakage. All headline numbers in this project are the stricter grouped figures.

## Models

Four algorithms were compared per target on identical folds: logistic regression,
random forest, XGBoost and LightGBM.

| Target | Winner | Accuracy | AUC | MCC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Anticancer | RandomForest | 0.739 | 0.811 | 0.483 | 0.662 | 0.815 |
| Hemolytic | LightGBM | 0.937 | 0.980 | 0.874 | 0.926 | 0.948 |

**The anticancer choice is stable.** A nested evaluation with four outer folds,
group-aware throughout, ran the entire model-selection procedure inside each
training fold. All four folds independently picked `aac+dpc+phys / RandomForest`.

**The hemolytic choice is arbitrary.** The same four folds picked four different
winners (RF, logistic regression, XGBoost, LightGBM), all within about two points
of each other. The signal in that dataset is strong enough that the algorithm
barely matters, so no time was spent tuning it.

## Thresholds: both left at 0.5

Both training sets are exactly 50/50, which makes 0.5 the principled default.

Threshold tuning on out-of-fold predictions moved the anticancer cutoff to 0.555
for +0.2% accuracy, while dropping sensitivity from 0.66 to 0.59. That is a gain
far below the fold-to-fold noise (±0.08 on that setup), bought by half-ignoring
one class.

`best_threshold()` in `cv.py` therefore only leaves 0.5 when tuning gains more
than 1%, and among thresholds statistically tied with the best it prefers the one
with the most balanced sensitivity and specificity.

## What was tested and rejected

**Pre-trained protein language model embeddings (ESM-2).** Tested properly
against the winning feature set — 320-dimensional mean-pooled embeddings from
`facebook/esm2_t6_8M_UR50D` concatenated with `aac+dpc+phys`, giving 758 features,
grouped folds, four seeds:

| Fold seed | Hand-built only | ESM + hand-built | Delta |
|---|---|---|---|
| 42 | 0.7386 | 0.7147 | −0.0239 |
| 7 | 0.7220 | 0.7230 | +0.0010 |
| 13 | 0.7230 | 0.7095 | −0.0135 |
| 2026 | 0.7168 | 0.7158 | −0.0010 |

Mean **−0.0093**, winning 1 of 4, with AUC dropping from 0.811 to 0.788. 320
dense dimensions dilute 438 informative ones. The embedding code remains in
`src/peptide_classifier/esm.py` and is reachable through `--esm`, but it is not
part of the shipped models.

**Feature scaling.** Feature ranges before scaling span five orders of magnitude,
so scaling looks like it should matter. For the shipped models it is exactly a
no-op: random forest scored 0.738589 both with and without `StandardScaler`,
identical to six decimal places. A tree asks whether a feature is above a
threshold, and rescaling moves that threshold by the same factor, so the same
peptides fall on the same side of every split. Both shipped models are
tree-based. The same test gave an SVM +7.5 points — and the scaled SVM still
only reached 0.709, below the random forest's 0.739.

**Feature selection.** `SelectKBest` at k = 50, 100 and 200 all scored below the
full 438 columns, monotonically. Dipeptide columns are individually weak and
jointly useful, which is exactly what a tree ensemble exploits and a univariate
filter discards. A random forest already selects features at every split.

**Support vector machines.** Lost at every setting: best RBF 0.7085, linear
0.6131. On 438 sparse compositional features, distance-based margins are the
wrong tool.

**Ensembling.** All 11 combinations of RF, XGBoost, LightGBM and logistic
regression scored below random forest alone (best ensemble 0.724 versus 0.739).
The forest is much stronger than the others, so averaging drags it down.

The full record is in [experiments.md](experiments.md).

## On the 100% training accuracy

The anticancer random forest scores 1.0000 on its own training data against 0.7385
held out. That gap is not diagnostic here: with `min_samples_leaf=1`, every tree
grows until its leaves are pure, so training accuracy is 1.00 by construction for
any random forest, good or bad.

The correct test is whether constraining the model helps. It does not:
leaf=1 → 0.739, leaf=2 → 0.722, leaf=3 → 0.709, leaf=5 → 0.709. Every
simplification made held-out accuracy worse, so the complexity is justified.

A learning curve on the anticancer set (25% → 0.685, 50% → 0.716, 75% → 0.714,
100% → 0.729) is still rising but flattening. Doubling the data from half to all
of it buys about 1.4 points, which confirms both that the final model should be
fit on all 964 peptides and that the limit is problem difficulty rather than model
choice.
