# Experiment record

Every attempt made to beat the frozen models during the competition, and the
result of each. Nothing in this record changed the shipped models.

The scripts that produced the numbers below are in `experiments/`, and their raw
output is in the `.csv` files alongside them.

## Every attempt to improve the anticancer model

Anticancer accuracy (0.739) was the weak half of the average, so nearly all
optimization effort went there. Hemolytic sat at 0.937 with little headroom.

| # | Attempt | Result |
|---|---|---|
| 1 | 12 feature/algorithm combinations | `rf + aac+dpc+phys` won |
| 2 | ESM-2 embeddings alone | 0.722 < 0.739 |
| 3 | Ensembling, 11 combinations | all below RF alone (best 0.724) |
| 4 | RF hyperparameters, 10 variants | current settings won |
| 5 | Feature scaling | exactly 0.000 change (trees) |
| 6 | Threshold tuning | 0.5 optimal; tuning collapses a class |
| 7 | Independent second feature set | ~0.74, same ballpark |
| 8 | Feature selection (`SelectKBest`) | worse, monotonically (k=300 → 0.728, k=50 → 0.702) |
| 9 | Terminal N/C-region features | 0.7334 < 0.7386 |
| 10 | Shuffled-dipeptide control | real 0.739 vs shuffled 0.711 — order carries signal |
| 11 | SVM, RBF and linear, every setting | best 0.7085, worst 0.6131 |
| 12 | RF-importance feature selection | +0.0052 on one seed, noise across four |
| 13 | ESM-2 concatenated with the winning set | mean −0.0093, 1 win of 4 |

Thirteen unrelated routes converging on roughly 0.74 makes that number a property
of the dataset rather than of any modelling choice. The AUC of 0.811 against an
accuracy of 0.739 says the model ranks peptides well and the problem is simply
hard near the decision boundary.

### The one apparent improvement, and why it was rejected

RF-importance-based feature selection gained **+0.0052** over plain random forest
on fold seed 42 — enough to look like a real improvement. Re-running the identical
comparison across four fold seeds:

| Fold seed | Plain RF | RF-select + RF | Delta |
|---|---|---|---|
| 42 | 0.7386 | 0.7438 | +0.0052 |
| 7 | 0.7220 | 0.7210 | −0.0010 |
| 13 | 0.7230 | 0.7292 | +0.0062 |
| 2026 | 0.7168 | 0.7158 | −0.0010 |

Mean **+0.0023**, winning 2 of 4. That is noise, and it was not adopted.

The same sweep is also the reason the headline 0.739 should be read with care:
plain RF scored 0.7386 / 0.7220 / 0.7230 / 0.7168 across those four splits, mean
**0.7251**, spread 0.022. The 0.739 figure is the luckiest of four.

Reproduce with:

```bash
python experiments/exp_seeds.py
```

### Apples-to-apples, proven

Every candidate in `exp_anticancer.py` ran on the same similarity clusters and the
same `StratifiedGroupKFold(5, shuffle=True, random_state=42)` folds as the saved
model, and every scaler and selector sat inside the pipeline so it was fit on
training folds only.

The proof the comparison is valid: plain random forest re-scored **0.7386** on
those folds, matching the saved model's recorded CV accuracy to four decimal
places. Without reproducing the baseline exactly, no delta is attributable.

```bash
python experiments/exp_anticancer.py
```

## Investigating another team's higher score

During the event another team reported 74% anticancer and 95% hemolytic (average
84.5%) using transformers, against this project's 73.9 / 93.7 (average 83.8).

Rather than assume their model was better, the obvious alternative hypothesis was
tested: that the gap was in how the two teams measured. The same frozen models
were re-scored under a plain random split instead of similarity-grouped folds,
changing nothing else:

| Target | Model | Grouped | Plain random split | Protocol is worth |
|---|---|---|---|---|
| Anticancer | RF | 0.7251 | **0.7516** | +2.65 pts |
| Hemolytic | LightGBM | 0.9438 | **0.9490** | +0.53 pts |
| **Average** | | **0.8345** | **0.8503** | **+1.6 pts** |

The same models score **85.0%** on the protocol most teams were likely using —
level with or slightly ahead of the 84.5% report, with nothing changed. The
grouped number was not a worse model; it was a stricter measurement.

The experiment took about a minute. Chasing the model instead would have meant
spending the last hour of the event rebuilding something that was already
competitive.

```bash
python experiments/exp_protocol_esm.py
```

This also prompted the first proper test of ESM-2 **against the winning feature
set** rather than on its own — see [methodology.md](methodology.md#what-was-tested-and-rejected).
That test is what turned "pre-trained embeddings do not help here" from an
overstated claim into a supported one.

## Two corrections made to the project's own reported numbers

Both were found by re-checking claims against the saved artifacts.

**Hemolytic 0.9369 is pessimistic, not optimistic.** Seed 42 is the *worst* of
four fold seeds for hemolytic; the four-seed mean is 0.9438. The anticancer
headline leans the other way (0.7386 is the best of four, mean 0.7251). The two
biases roughly cancel, which is why the 0.838 average holds up against the 0.8345
four-seed mean.

**An early draft of the project summary was wrong about the models.** It claimed
random forest gave the strongest results for both targets and listed dipeptides as
a feature of both. Checked against the saved bundles, the hemolytic model is
LightGBM on `aac+phys` with no dipeptides. Anything stated about a model is
verifiable by loading its `best_model.joblib`, and should be.

## Independent reproduction

The two team members built separate feature sets independently. Sayf's set
(charge density, hydrophobic moment, aromaticity, AAC — no dipeptides, 32
features) scored ~0.743 on a random split and ~0.696 cluster-split: the same
ballpark from a different direction. That code is kept at
`experiments/sayf_baseline/`.

Two unrelated feature sets landing on the same number is strong evidence the
result is real and close to the dataset's ceiling, rather than an artifact of one
person's choices.

A nested held-out evaluation run independently reached the same place:

| | Reported (grouped CV) | Nested held-out | Spread |
|---|---|---|---|
| Anticancer | 0.739 | 0.727 | ±0.024 |
| Hemolytic | 0.937 | 0.930 | ±0.020 |
| **Average** | **0.838** | **0.829** | |

About one point of optimism from model selection, inside the noise.

## What would actually be better, given more time

- **More labelled peptides.** The learning curve is still rising.
- **A properly fine-tuned protein language model.** This needs far more than 964
  examples, and a GPU; the event machine was CPU-only.
- **Splitting the anticancer label by cell line**, if the heterogeneity
  hypothesis is right. This needs assay metadata that was not provided.
