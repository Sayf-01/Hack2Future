"""Hand-made, explainable peptide features.

aac   : amino-acid composition, 20 fractions
dpc   : dipeptide composition, 400 fractions (pairs of neighbours)
phys  : physico-chemical descriptors the biology story hangs on:
        length, net charge, pI, hydrophobicity, hydrophobic moment
        (amphipathicity), Boman index, and a few group fractions.
"""
import numpy as np
import pandas as pd
import peptides

from .load import AA, clean_seq

IDX = {a: i for i, a in enumerate(AA)}
DIPEP = [a + b for a in AA for b in AA]

GROUPS = {
    "frac_KR": "KR",            # cationic: drawn to negatively charged bacterial membranes
    "frac_DE": "DE",            # anionic
    "frac_hydrophobic": "AILMFVW",
    "frac_aromatic": "FWY",
    "frac_C": "C",              # disulfide bridges (defensins)
    "frac_P": "P",              # helix breaker
    "frac_G": "G",
    "frac_polar": "STNQ",
}


def aac(seqs):
    M = np.zeros((len(seqs), 20), dtype=np.float32)
    for r, s in enumerate(seqs):
        for ch in s:
            M[r, IDX[ch]] += 1
        if s:
            M[r] /= len(s)
    return pd.DataFrame(M, columns=[f"aac_{a}" for a in AA])


def dpc(seqs):
    M = np.zeros((len(seqs), 400), dtype=np.float32)
    for r, s in enumerate(seqs):
        for a, b in zip(s, s[1:]):
            M[r, IDX[a] * 20 + IDX[b]] += 1
        if len(s) > 1:
            M[r] /= len(s) - 1
    return pd.DataFrame(M, columns=[f"dpc_{d}" for d in DIPEP])


def _phys_one(s):
    n = len(s)
    if n == 0:
        return {}
    p = peptides.Peptide(s)
    d = {
        "length": n,
        "charge_pH7": p.charge(pH=7.0),
        "charge_per_res": p.charge(pH=7.0) / n,
        "pI": p.isoelectric_point(),
        "hydrophobicity": p.hydrophobicity(),           # Kyte-Doolittle, mean
        "boman": p.boman(),                              # protein-binding potential
        "aliphatic_index": p.aliphatic_index(),
        "instability": p.instability_index(),
        "mol_weight": p.molecular_weight(),
    }
    # Hydrophobic moment at 100 deg = alpha helix. High = one face oily, one face charged.
    w = min(11, n)
    try:
        d["hydrophobic_moment"] = p.hydrophobic_moment(angle=100, window=w)
    except Exception:
        d["hydrophobic_moment"] = np.nan
    for name, letters in GROUPS.items():
        d[name] = sum(s.count(ch) for ch in letters) / n
    return d


def phys(seqs):
    return pd.DataFrame([_phys_one(s) for s in seqs]).astype(np.float32)


BUILDERS = {"aac": aac, "dpc": dpc, "phys": phys}


def build(seqs, groups=("aac", "phys")):
    seqs = [clean_seq(s) for s in seqs]
    parts = [BUILDERS[g](seqs) for g in groups]
    X = pd.concat(parts, axis=1)
    return X.fillna(X.median(numeric_only=True)).fillna(0)
