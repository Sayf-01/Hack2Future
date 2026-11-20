"""
Turn a peptide sequence (a string) into numbers a model can use.

Written by Sayf Al Safadi as an independent second feature set, separate from
``src/peptide_classifier/features.py``. 32 features: charge, greasiness, shape
and amino acid composition, with no dipeptides.

Design rule for this file: every feature here must be explainable in one
sentence, so there is deliberately nothing clever in it.

Why these features?  The biology:
cancer cell membranes carry MORE NEGATIVE charge than healthy cells, so
POSITIVELY charged peptides get pulled in and punch holes. So the features
that should matter most are charge, size, and how the peptide arranges its
water-loving vs water-avoiding parts.
"""

import re
import math
from collections import Counter

import pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")

# Eisenberg consensus hydrophobicity scale — positive = water-avoiding
# (this is the same table the organizers put in build_your_own_features_example.py)
HYDROPHOBICITY = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08,
}

POSITIVE = "KR"       # lysine, arginine — carry + charge at body pH
NEGATIVE = "DE"       # aspartate, glutamate — carry - charge
HYDROPHOBIC = "AILMFVWC"
AROMATIC = "FWY"


def clean_sequence(seq) -> str:
    """
    Keep only the 20 standard amino acid letters, uppercase.

    The training data is already clean, but the final test file is generated
    separately and we never get to inspect it before scoring ~650 rows. This
    guard means one odd character can't crash the run at 3:35 PM.
    """
    seq = str(seq).strip().upper()
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """
    How 'two-faced' (amphipathic) the peptide is.

    Plain English: many membrane-punching peptides coil into a helix where all
    the greasy amino acids end up on one side and all the charged ones on the
    other. That lets one face grip the membrane while the other faces the water.
    This number measures how strongly that split exists.

    How it's computed: walk along the sequence placing each residue 100 degrees
    further around a circle (the turn angle of an alpha helix), treat each one
    as an arrow whose length is its hydrophobicity, and add the arrows up. If
    the greasy residues cluster on one side, the arrows reinforce and the total
    is large. If they're spread evenly, they cancel out and the total is small.
    """
    if not seq:
        return 0.0
    rad = math.radians(angle_deg)
    x = sum(HYDROPHOBICITY.get(a, 0.0) * math.cos(i * rad) for i, a in enumerate(seq))
    y = sum(HYDROPHOBICITY.get(a, 0.0) * math.sin(i * rad) for i, a in enumerate(seq))
    return math.sqrt(x * x + y * y) / len(seq)


def extract_features(seq: str) -> dict:
    """One sequence -> one flat dict of {feature_name: number}."""
    seq = clean_sequence(seq)
    if not seq:
        seq = "A"  # guard so a fully-invalid row still produces a valid row

    n = len(seq)
    counts = Counter(seq)

    n_pos = sum(counts.get(a, 0) for a in POSITIVE)
    n_neg = sum(counts.get(a, 0) for a in NEGATIVE)
    net_charge = n_pos - n_neg

    feats = {
        # --- size ---
        "length": n,

        # --- charge: the core of the mechanism ---
        "net_charge": net_charge,
        "charge_density": net_charge / n,       # charge per residue, so short and long peptides compare fairly
        "frac_positive": n_pos / n,             # fraction that is K or R
        "frac_negative": n_neg / n,             # fraction that is D or E

        # --- greasiness: needed to insert into a membrane ---
        "avg_hydrophobicity": sum(HYDROPHOBICITY.get(a, 0.0) for a in seq) / n,
        "frac_hydrophobic": sum(counts.get(a, 0) for a in HYDROPHOBIC) / n,
        "frac_aromatic": sum(counts.get(a, 0) for a in AROMATIC) / n,

        # --- shape: greasy on one face, charged on the other ---
        "hydrophobic_moment": hydrophobic_moment(seq),

        # --- structure-breakers worth their own column ---
        "frac_G": counts.get("G", 0) / n,       # glycine: very flexible, breaks up helices
        "frac_P": counts.get("P", 0) / n,       # proline: puts a kink in a helix
        "frac_C": counts.get("C", 0) / n,       # cysteine: forms bridges that lock the shape
    }

    # --- amino acid composition: what fraction of the peptide is each letter ---
    # 20 more columns. Lets the model find patterns we didn't think to name.
    for a in AA:
        feats[f"aac_{a}"] = counts.get(a, 0) / n

    return feats


def build_feature_matrix(sequences) -> pd.DataFrame:
    """
    THE CONTRACT FUNCTION (see COLLAB.md).

    Takes any iterable of raw sequence strings. Returns a DataFrame with one
    row per input sequence, in the same order. Cleans internally. Never raises.
    """
    return pd.DataFrame([extract_features(s) for s in sequences])


if __name__ == "__main__":
    demo = [
        "KWKLFKKIGAVLKVL",   # classic positively-charged membrane puncher
        "SSGNNNNKKEEDSDEI",  # mixed charge, from the challenge doc example
        "YSL",               # a very short one from the real training data
    ]
    df = build_feature_matrix(demo)
    pd.set_option("display.width", 200)
    print(df[["length", "net_charge", "charge_density", "frac_positive",
              "avg_hydrophobicity", "hydrophobic_moment"]].to_string(index=False))
    print(f"\n{df.shape[1]} features per sequence")
