"""Feature extraction must be deterministic, correctly shaped and order-preserving.

If any of this breaks, the saved models silently receive the wrong columns and
produce confident nonsense, so these are the tests that matter most.
"""
import numpy as np
import pytest

from peptide_classifier import features
from peptide_classifier.load import clean_seq


def test_feature_group_widths(peptides):
    """Each group has a fixed, known width: 20 + 400 + 18 = 438."""
    assert features.build(peptides, ("aac",)).shape == (len(peptides), 20)
    assert features.build(peptides, ("dpc",)).shape == (len(peptides), 400)
    assert features.build(peptides, ("phys",)).shape == (len(peptides), 18)
    assert features.build(peptides, ("aac", "dpc", "phys")).shape == (len(peptides), 438)


def test_composition_sums_to_one(peptides):
    """Amino acid composition is a set of fractions, so each row sums to 1."""
    aac = features.build(peptides, ("aac",))
    np.testing.assert_allclose(aac.sum(axis=1), 1.0, rtol=1e-5)


def test_dipeptide_counts_are_fractions_of_adjacent_pairs():
    """A peptide of length n has n-1 adjacent pairs, so its DPC row sums to 1."""
    dpc = features.build(["KWKLFKKIGAVLKVL"], ("dpc",))
    np.testing.assert_allclose(dpc.sum(axis=1), 1.0, rtol=1e-5)
    # AAAA is three identical AA pairs, all in one column
    aaaa = features.build(["AAAA"], ("dpc",))
    assert aaaa["dpc_AA"].iloc[0] == pytest.approx(1.0)


def test_charge_follows_the_biology():
    """Net charge must rise with K/R content and fall with D/E content."""
    f = features.build(["KKKKRRRR", "DDDDEEEE"], ("phys",))
    assert f["charge_pH7"].iloc[0] > 5
    assert f["charge_pH7"].iloc[1] < -5
    assert f["frac_KR"].iloc[0] == pytest.approx(1.0)
    assert f["frac_DE"].iloc[1] == pytest.approx(1.0)


def test_row_order_is_preserved(peptides):
    """Row i of the matrix must be the features of sequence i, always."""
    full = features.build(peptides, ("aac", "phys"))
    for i, seq in enumerate(peptides):
        one = features.build([seq], ("aac", "phys"))
        np.testing.assert_allclose(full.iloc[i].to_numpy(), one.iloc[0].to_numpy(), rtol=1e-5)


def test_deterministic(peptides):
    """The same input must give byte-identical features every time."""
    a = features.build(peptides, ("aac", "dpc", "phys"))
    b = features.build(peptides, ("aac", "dpc", "phys"))
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    assert list(a.columns) == list(b.columns)


def test_no_nans_survive(peptides):
    """NaNs reaching a model are silent corruption; build() must fill them."""
    f = features.build(peptides + ["X", "", "ZZZ"], ("aac", "dpc", "phys"))
    assert not f.isna().any().any()


@pytest.mark.parametrize("raw,expected", [
    ("KWKL", "KWKL"),
    ("KW-KL", "KWKL"),
    ("KWXKL", "KWKL"),     # X is not one of the 20 standard letters
    ("", ""),
])
def test_sequence_cleaning(raw, expected):
    """clean_seq keeps only the 20 standard letters.

    It deliberately does not handle case: normalising to upper case happens once,
    in load(), so that the raw string can be preserved for the output file.
    """
    assert clean_seq(raw) == expected
