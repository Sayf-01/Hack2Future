# Peptide Bioactivity Classification

Machine learning models that predict, from an amino acid sequence alone, whether
a peptide kills cancer cells and whether it also destroys healthy red blood
cells.

Built for **Hack2Future 2026** (Track 01, Biology) and cleaned up afterwards into
a reproducible project.

---

## Overview

Some short peptides kill cancer cells. Cancer cell membranes carry more negative
charge than healthy ones, so positively charged peptides are drawn to them and
tear holes in the membrane. That makes them a promising starting point for drugs
that do not rely on the mechanisms tumours commonly evolve resistance to.

The problem is that the same property makes a peptide dangerous. A peptide that
punches holes in a cancer cell membrane may do the same to red blood cells, which
is called hemolysis. A peptide is only useful if it does the first thing *without*
doing the second.

So this is deliberately **two classifiers, not one**:

| Model | Question | Positive class |
|---|---|---|
| Anticancer | Does it kill cancer cells? | active |
| Hemolytic | Does it destroy red blood cells? | toxic |

Crossing the two answers gives the result a researcher actually wants — which
peptides are worth synthesising and testing in a lab:

| Anticancer | Hemolytic | Verdict |
|---|---|---|
| yes | no | **Selective** — the one worth pursuing |
| yes | yes | Works but toxic |
| no | yes | Toxic and doesn't work |
| no | no | Does nothing |

Wet-lab screening of a peptide library is slow and expensive. A model that reads
a sequence string and ranks candidates in milliseconds lets a lab spend its
bench time on the sequences most likely to be worth it.

## Key results

Validation is **5-fold cross-validation with similarity-grouped folds** on the
organizers' training data (964 anticancer and 618 hemolytic sequences, both
exactly balanced).

| Model | Algorithm | Features | Accuracy | AUC | MCC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Anticancer | RandomForest | 438 (AAC + DPC + physicochemical) | 0.739 | 0.811 | 0.483 | 0.662 | 0.815 |
| Hemolytic | LightGBM | 38 (AAC + physicochemical) | 0.937 | 0.980 | 0.874 | 0.926 | 0.948 |
| **Average** | | | **0.838** | | | | |

**How to read these numbers honestly:**

- They are **cross-validation figures on training data, not blind-test results**,
  and not an official competition score. The organizers' final test set has no
  public labels, so no accuracy can be computed on it from this side.
- They are **single-seed** figures. Across four fold seeds the means are 0.725
  (anticancer) and 0.944 (hemolytic). The anticancer headline is the luckiest of
  four; the hemolytic headline is the unluckiest. The two biases roughly cancel.
- An independent **nested held-out evaluation**, where model selection happened
  entirely inside the training folds, gives **0.829** average — about one point
  of selection optimism, inside the noise.

The spread matters more than the point estimate: realistic anticancer performance
is **0.72–0.74**, not 0.739 exactly.

### The result worth more than the accuracy

Peptide datasets are full of near-duplicates — the same peptide with one or two
substitutions. The largest near-duplicate cluster in the anticancer set is **42
sequences**. Under a plain random split, a peptide trains on its own near-twin
and the score comes out inflated.

This project clusters sequences by 3-mer Jaccard similarity and keeps every
cluster inside a single fold. Measuring what that protocol is worth, changing
nothing else:

| Target | Grouped folds | Plain random split | Difference |
|---|---|---|---|
| Anticancer | 0.725 | 0.752 | +2.65 pts |
| Hemolytic | 0.944 | 0.949 | +0.53 pts |
| **Average** | **0.835** | **0.850** | **+1.6 pts** |

The same models score **85.0%** under the looser protocol. Roughly 1.6 points of
any random-split number on this data is near-duplicate leakage rather than
predictive skill. Every headline figure above is the stricter one.

## Technical approach

**Features** — all hand-built and explainable, no external pre-trained predictor:

- **Amino acid composition (AAC)**, 20 columns — the fraction of each residue.
- **Dipeptide composition (DPC)**, 400 columns — the fraction of each adjacent
  residue pair, capturing local sequence order that AAC discards.
- **Physicochemical descriptors**, 18 columns — length, net charge at pH 7,
  isoelectric point, hydrophobicity, hydrophobic moment (amphipathicity), Boman
  index, instability and aliphatic indices, and group fractions for cationic,
  anionic, hydrophobic, aromatic and polar residues.

**Model selection** — four algorithms (logistic regression, random forest,
XGBoost, LightGBM) compared per target across several feature sets on identical
similarity-grouped folds, ranked on accuracy with AUC breaking ties. The winner
is refit on all available data and saved together with the exact column list it
was trained on, so inference cannot drift from training.

**Feature sets differ by target on purpose.** Dipeptides moved anticancer
accuracy from 0.702 → 0.721 → 0.739, and a shuffled-sequence control confirmed
the gain comes from real sequence order. On hemolytic they added nothing, so that
model keeps the smaller 38-feature set.

**Thresholds are both 0.5.** Both datasets are exactly balanced. Tuning bought
0.2% accuracy while dropping sensitivity from 0.66 to 0.59 — a gain far below
fold-to-fold noise, paid for by half-ignoring a class.

**What was tried and rejected:** pre-trained protein language model embeddings
(ESM-2, mean −0.009 across four seeds), SVMs (best 0.709), feature selection
(worse monotonically), ensembling (all 11 combinations below random forest alone),
and feature scaling (exactly zero change for trees). Thirteen attempts in total,
all recorded in [docs/experiments.md](docs/experiments.md).

## Pipeline

```
   peptide sequence
   "KWKLFKKIGAVLKVL"
           │
           ▼
   ┌───────────────┐
   │ preprocessing │   uppercase, strip non-standard letters,
   └───────────────┘   keep the raw string for the output join
           │
           ▼
   ┌───────────────────────────────────────┐
   │ feature extraction                    │
   │   AAC (20) · DPC (400) · phys (18)    │
   └───────────────────────────────────────┘
           │
           ├──────────────────────┬──────────────────────┐
           ▼                      ▼                      │
   ┌────────────────┐     ┌────────────────┐             │
   │  anticancer    │     │   hemolytic    │             │
   │  RandomForest  │     │   LightGBM     │             │
   │  438 features  │     │   38 features  │             │
   └────────────────┘     └────────────────┘             │
           │                      │                      │
           ▼                      ▼                      │
     probability             probability                 │
           │                      │                      │
           └──────────┬───────────┘                      │
                      ▼                                  │
            threshold at 0.5                             │
                      │                                  │
                      ▼                                  ▼
          ┌─────────────────────────────────────────────────┐
          │ verdict                                         │
          │   Selective · Works but toxic ·                 │
          │   Toxic and doesn't work · Does nothing         │
          └─────────────────────────────────────────────────┘
```

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Sayf-01/Hack2Future.git
cd Hack2Future
pip install -r requirements.txt
```

## Usage

Score a file of peptide sequences with the saved models:

```bash
python scripts/predict.py --input examples/sample_peptides.csv \
                          --output predictions.csv
```

**Input** — a CSV, TSV, Excel file or FASTA containing a column of sequences. The
sequence column is detected automatically (override with `--seq-col`); any other
columns are ignored.

```csv
id,sequence
peptide_001,KRPFPKKWMFKAFVLRF
peptide_002,GGWDSIYFYIMGPMVLMWAIAY
```

**Output** — one row per input sequence, in the same order:

```csv
sequence,anticancer_probability,anticancer_prediction,hemolytic_probability,hemolytic_prediction,verdict
KRPFPKKWMFKAFVLRF,0.676,1,0.999989,1,Works but toxic
GGWDSIYFYIMGPMVLMWAIAY,0.568,1,0.003139,0,Selective
```

(Those are the real first two rows from the command above. The sample sequences
are synthetic, so the predictions are an illustration of the output format rather
than a biological result.)

The script validates its own output before writing — column order, row count,
probabilities inside [0, 1], predictions strictly 0/1, verdict strings from the
allowed set, and each verdict consistent with its own two predictions — and exits
non-zero if any check fails.

Useful flags: `--seq-col` to name the sequence column explicitly, `--template` to
check the header against a reference schema, `--half` to force a 0.5 cutoff
regardless of what each bundle stores.

## Reproducibility

**Inference** uses the committed model artifacts in `models/`, so it reproduces
exactly with no retraining. The submission made during the competition is kept at
`submissions/final.csv`, and the current code regenerates it byte-for-byte from
the same inputs.

**Retraining** both models from the provided data takes under a minute each:

```bash
python scripts/train.py --train participant_datasets/initial/anticancer_train.csv \
                        --target anticancer --out models/anticancer

python scripts/train.py --train participant_datasets/initial/hemolytic_train.csv \
                        --target hemolytic  --out models/hemolytic
```

Each run writes `best_model.joblib` alongside `cv_results.csv`, `data_report.txt`,
SHAP importances and diagnostic plots. All random seeds are fixed at 42.

**Tests:**

```bash
pip install -r requirements-dev.txt
pytest
```

43 tests covering feature shapes and determinism, near-duplicate clustering,
dataset loading, model artifact integrity, and the prediction script end to end.
The model tests assert the exact figures quoted in this README against the saved
bundles, so the documentation cannot drift away from the artifacts.

**Reproduce the experiments** behind the claims above:

```bash
python experiments/exp_anticancer.py     # 13 candidates vs the baseline
python experiments/exp_seeds.py          # is a small win real across fold seeds?
python experiments/exp_protocol_esm.py   # grouped vs random split; ESM-2
```

## Limitations

- **Cross-validation accuracy is not biological efficacy.** These models predict
  labels in a curated dataset. A peptide predicted Selective is a candidate for
  lab testing, nothing more.
- **Near-duplicate leakage is reduced, not eliminated.** 3-mer Jaccard similarity
  is a cheap proxy for sequence relatedness, not an alignment. Peptides related
  below the 0.6 threshold can still share folds.
- **The datasets are small** — 964 and 618 sequences. The learning curve is still
  rising, so more labelled data would likely help more than a better model.
- **Anticancer performance (~0.74) is near this dataset's ceiling.** Thirteen
  unrelated attempts converged there. The limit appears to be the problem and the
  data, not the choice of algorithm. One plausible cause is label heterogeneity:
  "anticancer" aggregates assays across different cell lines, and splitting by
  cell line would need assay metadata that was not provided.
- **No external validation.** The models have never been tested against an
  independent dataset from a different source.
- **Hemolytic performance may be optimistic** relative to the anticancer model
  simply because hemolysis has a simpler, more direct physicochemical mechanism.

## Repository layout

```
src/peptide_classifier/   feature extraction, similarity clustering, CV, plots
scripts/                  train.py and predict.py — the entry points
models/                   saved model bundles and their training reports
tests/                    pytest suite
examples/                 a small sample input and the output schema
experiments/              the scripts behind the claims in docs/
docs/                     methodology and the full experiment record
participant_datasets/     training data as provided by the organizers
submissions/final.csv     the competition submission, kept as a record
```

## Team and acknowledgements

Built at **Hack2Future 2026**, Track 01 (Biology), by a team of two:

- **[Sayf Al Safadi](https://github.com/Sayf-01)** — modelling, the independent
  second feature set in `experiments/sayf_baseline/`, nested held-out validation,
  and the ensembling, hyperparameter and scaling experiments.
- **[Dany Abou Saleh](https://github.com/XxitsdanyxX)** — the feature and
  validation pipeline in `src/`, similarity-grouped cross-validation, the
  prediction and verification path, and the protocol and ESM-2 experiments.

Thanks to the Hack2Future organizers and mentors for the challenge, the datasets
and the biological framing.

**Data:** the contents of `participant_datasets/` were provided by the
organizers for use in the competition and are included here so the results can be
reproduced. They are not this team's data to license. The organizers' final test
sequences are deliberately not committed.

**On tooling:** AI coding assistants were used during the event for writing,
debugging and documenting code, which the competition rules permitted. The
models, the feature design and the validation methodology are the team's own, and
every number in this repository is reproducible from the scripts and artifacts
here.

## Further reading

- [docs/methodology.md](docs/methodology.md) — how the models were built and
  validated, and why each choice was made.
- [docs/experiments.md](docs/experiments.md) — every attempt to beat the models,
  including the ones that failed.
