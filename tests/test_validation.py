"""Near-duplicate clustering and dataset loading.

The grouped-CV result is the main methodological claim in this project, and it
rests entirely on leak.similarity_groups putting near-identical peptides in the
same cluster. These tests pin that behaviour down.
"""
import numpy as np
import pandas as pd
import pytest

from peptide_classifier import leak
from peptide_classifier.load import load, report


def test_identical_sequences_land_in_one_cluster():
    groups = leak.similarity_groups(["KWKLFKKIGAVLKVL"] * 4, threshold=0.6)
    assert len(np.unique(groups)) == 1


def test_near_duplicates_are_grouped_together():
    """One substitution must not be enough to escape the cluster."""
    seqs = ["KWKLFKKIGAVLKVL",    # original
            "KWKLFKKIGAVLKVA",    # one substitution at the end
            "SSGNEENKKEEDSDEI"]   # unrelated
    groups = leak.similarity_groups(seqs, threshold=0.6)
    assert groups[0] == groups[1], "near-duplicates were split across clusters"
    assert groups[2] != groups[0], "an unrelated peptide was absorbed into the cluster"


def test_unrelated_sequences_stay_separate():
    seqs = ["KKKKKKKKKKKK", "DDDDDDDDDDDD", "WWWWWWWWWWWW"]
    assert len(np.unique(leak.similarity_groups(seqs, threshold=0.6))) == 3


def test_cross_similarity_is_one_for_an_exact_match():
    sim = leak.cross_max_similarity(["KWKLFKKIGAVLKVL"], ["KWKLFKKIGAVLKVL", "AAAA"])
    assert sim[0] == pytest.approx(1.0)


def test_real_training_data_has_the_documented_clusters(root):
    """docs/methodology.md states the largest cluster is 42 sequences."""
    path = root / "participant_datasets" / "initial" / "anticancer_train.csv"
    if not path.exists():
        pytest.skip("training data not present")
    df, _, _ = load(str(path))
    groups = leak.similarity_groups(df["sequence"].tolist(), threshold=0.6)
    assert np.bincount(groups).max() == 42


def test_load_preserves_the_raw_sequence(tmp_path):
    """The raw string is what the output file carries, so it must survive intact."""
    f = tmp_path / "seqs.csv"
    pd.DataFrame({"sequence": ["  kwkl  ", "KWKLFKKIGAVLKVL"]}).to_csv(f, index=False)
    df, _, _ = load(str(f))
    assert df["sequence_raw"].iloc[0] == "  kwkl  "
    assert df["sequence"].iloc[0] == "KWKL", "load() must normalise case and whitespace"


def test_load_detects_the_sequence_column(tmp_path):
    """Column names vary between files, so the sequence column is detected."""
    f = tmp_path / "odd_names.csv"
    pd.DataFrame({"peptide_id": ["a", "b"],
                  "aa_string": ["KWKLFKKIGAVLKVL", "SSGNEENKKEEDSDEI"]}).to_csv(f, index=False)
    _, _, info = load(str(f))
    assert info["seq_col"] == "aa_string"


def test_load_reads_fasta(tmp_path):
    f = tmp_path / "seqs.fasta"
    f.write_text(">seq1\nKWKLFKKIGAVLKVL\n>seq2\nSSGNEENKKEEDSDEI\n")
    df, _, _ = load(str(f))
    assert len(df) == 2
    assert df["sequence"].tolist() == ["KWKLFKKIGAVLKVL", "SSGNEENKKEEDSDEI"]


def test_report_mentions_class_balance(root):
    path = root / "participant_datasets" / "initial" / "hemolytic_train.csv"
    if not path.exists():
        pytest.skip("training data not present")
    df, _, _ = load(str(path))
    text = report(df, "hemolytic")
    assert "class balance" in text
    assert "618 sequences" in text
