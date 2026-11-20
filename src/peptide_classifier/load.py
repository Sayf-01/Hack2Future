"""Load a peptide dataset in whatever shape the organizers hand us.

Handles CSV/TSV/TXT (separator sniffed), Excel, Parquet and FASTA.
Guesses the sequence column (the one made of amino-acid letters) and the
label column (the first two-valued column, preferring names like 'label').
"""
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)
AMBIG = set("XBZUOJ")
POS_WORDS = {"1", "pos", "positive", "amp", "active", "yes", "true", "y", "t"}


def read_fasta(path):
    recs, head, buf = [], None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if head is not None:
                recs.append((head, "".join(buf)))
            head, buf = line[1:], []
        else:
            buf.append(line)
    if head is not None:
        recs.append((head, "".join(buf)))
    df = pd.DataFrame(recs, columns=["id", "sequence"])
    # Labels are often baked into the header: ">seq1|1", ">pos_12", "label=0"
    lab = df["id"].str.extract(r"(?:\||label[=_:]|_)([01])\s*$", expand=False)
    if lab.notna().mean() > 0.9:
        df["label"] = lab.astype(int)
    else:
        word = df["id"].str.lower().str.extract(r"(pos|neg|positive|negative|non)", expand=False)
        if word.notna().mean() > 0.9:
            df["label"] = word.str.startswith("pos").astype(int)
    return df


def read_any(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".fa", ".fasta", ".faa", ".fas", ".fna"):
        return read_fasta(p)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if ext == ".parquet":
        return pd.read_parquet(p)
    if ext == ".json":
        return pd.read_json(p)
    # Sniff the first line: a FASTA without the extension
    with open(p, encoding="utf-8", errors="replace") as f:
        if f.readline().startswith(">"):
            return read_fasta(p)
    if ext == ".csv":
        return pd.read_csv(p)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(p, sep="	")
    return pd.read_csv(p, sep=None, engine="python")


def guess_seq_col(df):
    best, best_frac = None, 0.0
    for c in df.columns:
        if not (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object):
            continue
        v = df[c].dropna().astype(str).str.strip().str.upper().head(1000)
        if v.empty:
            continue
        frac = v.map(lambda s: len(s) >= 2 and set(s) <= (AA_SET | AMBIG)).mean()
        # prefer longer strings when tie (a column of single letters is not sequences)
        if frac > best_frac or (frac == best_frac and best is not None
                                and v.str.len().mean() > df[best].astype(str).str.len().mean()):
            best, best_frac = c, frac
    return best if best_frac > 0.8 else None


def guess_label_col(df, seq_col):
    names = ("label", "labels", "y", "target", "class", "activity", "is_amp", "amp", "active")
    pref = [c for c in df.columns if str(c).lower() in names]
    rest = [c for c in df.columns if c not in pref and c != seq_col]
    for c in pref + rest:
        if df[c].dropna().nunique() == 2:
            return c
    return None


def to_binary(series):
    vals = sorted(series.dropna().unique(), key=lambda x: str(x))
    if set(map(str, vals)) <= {"0", "1", "0.0", "1.0"}:
        return series.astype(float).astype(int), {str(v): int(float(v)) for v in vals}
    if set(map(str, vals)) <= {"-1", "1", "-1.0", "1.0"}:
        return (series.astype(float) > 0).astype(int), {str(v): int(float(v) > 0) for v in vals}
    pos = [v for v in vals if str(v).strip().lower() in POS_WORDS]
    if len(pos) == 1:
        mapping = {v: int(v == pos[0]) for v in vals}
    else:
        mapping = {vals[0]: 0, vals[1]: 1}
    return series.map(mapping).astype(int), mapping


def load(path, seq_col=None, label_col=None, id_col=None):
    """Returns (df with columns id, sequence[, label], info dict)."""
    raw = read_any(path)
    if seq_col not in raw.columns:
        seq_col = guess_seq_col(raw)
    if seq_col is None:
        raise ValueError(f"No sequence column found. Columns: {list(raw.columns)}. Pass --seq-col.")
    if label_col is None:
        label_col = guess_label_col(raw, seq_col)
    out = pd.DataFrame({
        "id": raw[id_col].astype(str) if id_col else (raw["id"].astype(str) if "id" in raw.columns else raw.index.astype(str)),
        # sequence_raw is the string EXACTLY as the organizers wrote it. The
        # submission keys on this, so it must never be normalised.
        "sequence_raw": raw[seq_col].astype(str),
        "sequence": raw[seq_col].astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True),
    })
    info = {"path": str(path), "columns": list(raw.columns), "seq_col": seq_col,
            "label_col": label_col, "label_map": None}
    if label_col is not None and label_col in raw.columns:
        out["label"], info["label_map"] = to_binary(raw[label_col])
    return out, raw, info


def report(df, name="data"):
    """Plain-text summary: size, balance, lengths, odd letters, duplicates."""
    lines = [f"=== {name}: {len(df)} sequences ==="]
    L = df["sequence"].str.len()
    lines.append(f"length: min {L.min()}  median {int(L.median())}  mean {L.mean():.1f}  max {L.max()}")
    odd = df["sequence"].map(lambda s: sorted(set(s) - AA_SET))
    n_odd = (odd.str.len() > 0).sum()
    if n_odd:
        letters = sorted(set().union(*odd))
        lines.append(f"non-standard letters in {n_odd} sequences: {letters}")
    if "label" in df.columns:
        vc = df["label"].value_counts().sort_index()
        lines.append("class balance: " + ", ".join(f"{k}: {v} ({v/len(df):.1%})" for k, v in vc.items()))
        for k, g in df.groupby("label"):
            gl = g["sequence"].str.len()
            lines.append(f"  label {k}: median length {int(gl.median())}, mean {gl.mean():.1f}")
    dup = df["sequence"].duplicated(keep=False)
    if dup.any():
        lines.append(f"exact duplicate sequences: {dup.sum()} rows ({df.loc[dup,'sequence'].nunique()} unique)")
        if "label" in df.columns:
            conflict = df[dup].groupby("sequence")["label"].nunique()
            lines.append(f"  duplicates with CONFLICTING labels: {(conflict > 1).sum()}")
    else:
        lines.append("exact duplicate sequences: none")
    return "\n".join(lines)


def clean_seq(s):
    """Keep only the 20 standard letters (features and ESM need that)."""
    return "".join(ch for ch in s if ch in AA_SET)
