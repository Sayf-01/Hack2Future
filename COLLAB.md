# COLLAB.md — shared notes between both AI assistants

**This file is how the two AI assistants on this team stay in sync.**
There is no live connection between them. This file is the only channel.

## Rules (both assistants follow these)

1. **Read this file first.** Before doing any work, read this whole file so you
   know what the other side has already done.
2. **Write here when you finish something.** Add a line under "Work log" with
   the time, who did it, and what changed. Keep it short.
3. **Claim before you build.** Before starting a task, move it to "In progress"
   with your name on it. Don't start a task someone already claimed.
4. **Never edit the other person's files** without saying so here first.
5. **Pull before you push.** `git pull --rebase` then push. If there's a
   conflict in this file, keep BOTH sides' lines.
6. **Push often.** Small commits. Don't sit on 2 hours of uncommitted work.

## Who is who

| Side | Human | AI | Branch |
|---|---|---|---|
| A | Sayf | Claude | `sayf` |
| B | (teammate — fill in) | (their AI) | `teammate` |

Merge into `main` when a piece is working.

---

## The challenge (Track 01 — Biology)

Predict which peptides kill cancer cells.

- A peptide is a short string of letters (amino acid sequence, e.g. `KWKLFKKIGAVLKVL`).
- The organizers give real lab-tested data: some peptides kill cancer cells, some don't.
- Pipeline: **build features → train a classifier → output a probability.**
- **Two models required:**
  1. `anticancer` — does it kill cancer cells?
  2. `hemolytic` — does it damage red blood cells? (this is the safety/side-effect model)
- Scoring goal: **be right about both classes**, not just the positives. So a model
  that says "yes" to everything scores badly. Optimize a balanced metric
  (MCC / balanced accuracy / F1), not raw accuracy.

### Hard logistics

- **~650 new peptides drop 30 minutes before the deadline.**
- **Deadline: 4:00 PM. Upload via the Google Form.**
- Only the **last CSV submitted** counts.
- Organizers' instruction: **save your trained models, do NOT retrain** at the end.
- Have **one prediction script** already tested on the demo data, so the final run is
  just: load models → read new CSV → write predictions CSV.
- Submit early and often. A mediocre CSV in hand beats a great one that misses 4:00.

**All datasets / intro material / submission details come through Discord.** Whoever
gets them, commit them to `data/` immediately and note it below.

---

## Plan

Rough split for a team of 2. Adjust freely — just update this file.

**Person A (Sayf + Claude)** — modeling
- Feature builder (`src/features.py`)
- Train the two models (`src/train.py`), save to `models/`
- Cross-validation, pick the threshold

**Person B (teammate + their AI)** — pipeline & safety net
- `src/predict.py`: load saved models + a CSV of sequences → write submission CSV
- Verify the output format matches exactly what the organizers asked for
- Dry-run the whole final sequence on demo data **before** the 650 drop
- Slides / short writeup if the track asks for one

**Both, at T-40 min:** stop building. Freeze. Wait for the drop with
`predict.py` tested and models saved on disk.

### Checklist

- [ ] Dataset from Discord committed to `data/`
- [ ] Exact submission format confirmed (column names, order, probability vs label)
- [ ] `features.py` works on a sample
- [ ] Anticancer model trained + saved to `models/`
- [ ] Hemolytic model trained + saved to `models/`
- [ ] CV scores written below
- [ ] `predict.py` tested end-to-end on demo data
- [ ] One practice submission uploaded to the Google Form
- [ ] Final CSV uploaded before 4:00 PM

---

## Decisions

Append decisions here so neither side redoes the argument.

- **Features:** start simple and fast — amino acid composition (20 counts),
  dipeptide composition if time, length, net charge, hydrophobicity (GRAVY),
  aromaticity, isoelectric point. These are cheap to compute and work well on
  short peptides.
- **Model:** start with gradient boosting (LightGBM / XGBoost) or RandomForest +
  logistic regression baseline. Simple, fast, no GPU, no fine-tuning risk.
- **Validation:** stratified K-fold. Report balanced metric, not accuracy.
- **Saving:** `joblib.dump` to `models/anticancer.pkl` and `models/hemolytic.pkl`.
  Save the feature pipeline WITH the model so predict.py can't drift.

---

## Work log

Newest at the bottom. Format: `HH:MM — who — what`

- 11:40 — Claude (A) — Created repo structure and this file.

---

## Blockers / questions

- [ ] Need the dataset from Discord.
- [ ] Need the exact submission CSV format.
