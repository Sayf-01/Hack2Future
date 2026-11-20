"""End-to-end: run scripts/predict.py the way the README documents it.

This is the test that would have caught a broken refactor during the competition,
so it runs the real script as a subprocess rather than importing pieces of it.
"""
import subprocess
import sys

import pandas as pd
import pytest

COLUMNS = ["sequence", "anticancer_probability", "anticancer_prediction",
           "hemolytic_probability", "hemolytic_prediction", "verdict"]

VERDICTS = {"Selective", "Works but toxic", "Toxic and doesn't work", "Does nothing"}


@pytest.fixture(scope="module")
def predictions(root, tmp_path_factory):
    """Run the documented command once; every test below reads its output."""
    out = tmp_path_factory.mktemp("predict") / "predictions.csv"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "predict.py"),
         "--input", str(root / "examples" / "sample_peptides.csv"),
         "--output", str(out),
         "--template", str(root / "examples" / "output_schema.csv")],
        cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, f"predict.py failed:\n{result.stdout}\n{result.stderr}"
    assert out.exists(), "predict.py exited cleanly but wrote no file"
    return pd.read_csv(out)


def test_output_schema(predictions):
    """Column names and their order are part of the contract."""
    assert list(predictions.columns) == COLUMNS


def test_one_row_per_input_sequence(predictions, root):
    sample = pd.read_csv(root / "examples" / "sample_peptides.csv")
    assert len(predictions) == len(sample)


def test_sequences_are_returned_unchanged(predictions, root):
    """The output must still join against the input, position by position."""
    sample = pd.read_csv(root / "examples" / "sample_peptides.csv")
    assert predictions["sequence"].tolist() == sample["sequence"].tolist()


def test_probabilities_are_in_range(predictions):
    for col in ("anticancer_probability", "hemolytic_probability"):
        assert predictions[col].between(0, 1).all(), f"{col} outside [0, 1]"
        assert predictions[col].notna().all()


def test_predictions_are_strictly_binary(predictions):
    for col in ("anticancer_prediction", "hemolytic_prediction"):
        assert predictions[col].isin([0, 1]).all()


def test_verdicts_are_from_the_allowed_set(predictions):
    assert set(predictions["verdict"]) <= VERDICTS


def test_verdict_agrees_with_its_own_two_predictions(predictions):
    """The verdict is derived from the two labels, so it cannot disagree."""
    expected = {
        (1, 0): "Selective",
        (1, 1): "Works but toxic",
        (0, 1): "Toxic and doesn't work",
        (0, 0): "Does nothing",
    }
    for _, row in predictions.iterrows():
        key = (int(row["anticancer_prediction"]), int(row["hemolytic_prediction"]))
        assert row["verdict"] == expected[key], f"verdict disagrees with labels for {key}"


def test_missing_input_fails_cleanly(root, tmp_path):
    """A bad path must give a readable error, not a traceback."""
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "predict.py"),
         "--input", str(tmp_path / "does_not_exist.csv"),
         "--output", str(tmp_path / "out.csv")],
        cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "not found" in (result.stdout + result.stderr).lower()
    assert "Traceback" not in result.stderr


def test_runs_from_any_working_directory(root, tmp_path):
    """Paths resolve from the project root, not the caller's cwd."""
    out = tmp_path / "elsewhere.csv"
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "predict.py"),
         "--input", str(root / "examples" / "sample_peptides.csv"),
         "--output", str(out)],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, f"failed when run from {tmp_path}:\n{result.stderr}"
    assert out.exists()
