# Hack2Future — Track 01 (Biology)

Predicting which peptides kill cancer cells.

Two models: **anticancer** (does it work?) and **hemolytic** (does it harm red blood cells?).

## If you are an AI assistant joining this repo

**Read [`COLLAB.md`](COLLAB.md) first.** It is the shared notes file between the
two assistants working on this team. It holds the plan, the task claims, the
decisions already made, and the work log. Update it whenever you finish
something or make a decision.

## Layout

```
data/          datasets from the organizers (Discord)
src/           features.py, train.py, predict.py
models/        saved trained models (.pkl) — do not retrain at the end
submissions/   generated CSVs
notes/         scratch
```

## The one command that matters at the end

```bash
python src/predict.py --input data/final_peptides.csv --output submissions/final.csv
```

Loads the saved models. Does not retrain. Test this before the final drop.

## Setup

```bash
pip install -r requirements.txt
```
