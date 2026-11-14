# Hack2Future — Track 01 (Biology)

Predict which peptides kill cancer cells, and which also damage red blood cells.

Two classifiers, one verdict per peptide. Deadline **4:00 PM**.

## If you are an AI assistant joining this repo

**Read [`COLLAB.md`](COLLAB.md) first.** It is the shared notes file between the
two assistants working on this team. It holds the challenge spec, the exact
submission format, the judging criteria, task claims, decisions already made,
and the work log. Update it whenever you finish something or make a decision.

One rule worth repeating here: the organizers question possible winners about
their models and state that **you cannot use code you can't explain.** Write
code the humans on this team can walk through out loud.

## Layout

```
participant_datasets/
  demo/        fake data — environment test only, never train on it
  initial/     the real training data (anticancer + hemolytic, separate)
src/           features.py, train.py, predict.py
models/        saved .pkl models — do not retrain at the end
submissions/   generated CSVs
```

## The one command that matters at 3:30 PM

```bash
python src/predict.py --input final_test_sequences.csv --output submissions/final.csv
```

Loads the saved models. Does not retrain. Must be tested on demo data long
before the final set drops.

## Submission format

```
sequence,anticancer_probability,anticancer_prediction,hemolytic_probability,hemolytic_prediction,verdict
```

Verdicts: `Selective`, `Works but toxic`, `Toxic and doesn't work`, `Does nothing`.

Diff against `submission_template.csv` before submitting — the judging script
ignores malformed CSVs.

## Setup

```bash
pip install -r requirements.txt
python test_environment.py
```
