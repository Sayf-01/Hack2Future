"""Near-duplicate detection (the classic peptide-benchmark trap).

Similarity = Jaccard overlap of 3-letter chunks (3-mers). Fast, no alignment
needed, and good enough to catch "same peptide with 1-2 mutations".

- similarity_groups(): clusters of near-duplicates inside the training set,
  used for grouped CV so a peptide and its twin never sit on both sides of a fold.
- cross_max_similarity(): for each test peptide, how close is its nearest
  training peptide. A big share above ~0.8 means the test score is inflated.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .load import clean_seq


def _kmer_matrix(seqs, k=3, vocab=None):
    seqs = [clean_seq(s) for s in seqs]
    vocab = {} if vocab is None else vocab
    rows, cols = [], []
    for r, s in enumerate(seqs):
        km = {s[i:i + k] for i in range(len(s) - k + 1)} or {s}
        for m in km:
            c = vocab.setdefault(m, len(vocab))
            rows.append(r)
            cols.append(c)
    return rows, cols, vocab


def _pair(a_seqs, b_seqs, k=3):
    ra, ca, vocab = _kmer_matrix(a_seqs, k)
    rb, cb, vocab = _kmer_matrix(b_seqs, k, vocab)
    n = len(vocab)
    A = sparse.csr_matrix((np.ones(len(ra), np.float32), (ra, ca)), shape=(len(a_seqs), n))
    B = sparse.csr_matrix((np.ones(len(rb), np.float32), (rb, cb)), shape=(len(b_seqs), n))
    return A, B


def _jaccard_rows(A, B, chunk=2000):
    """Yields (row_offset, dense jaccard block A[chunk] x B)."""
    na = np.asarray(A.sum(1)).ravel()
    nb = np.asarray(B.sum(1)).ravel()
    BT = B.T.tocsc()
    for i in range(0, A.shape[0], chunk):
        inter = (A[i:i + chunk] @ BT).toarray()
        union = na[i:i + chunk, None] + nb[None, :] - inter
        yield i, inter / np.maximum(union, 1)


def cross_max_similarity(test_seqs, train_seqs, k=3):
    A, B = _pair(test_seqs, train_seqs, k)
    best = np.zeros(A.shape[0], np.float32)
    for i, J in _jaccard_rows(A, B):
        best[i:i + J.shape[0]] = J.max(1)
    return best


def similarity_groups(seqs, threshold=0.6, k=3):
    """Connected components of the 'similar above threshold' graph."""
    A, _ = _pair(seqs, seqs[:1], k)
    rows, cols = [], []
    for i, J in _jaccard_rows(A, A):
        r, c = np.nonzero(J >= threshold)
        rows.append(r + i)
        cols.append(c)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    G = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(seqs), len(seqs)))
    _, labels = connected_components(G, directed=False)
    return labels


def summary(sim, name="test vs train"):
    return (f"{name}: nearest-neighbour 3-mer similarity  median {np.median(sim):.2f}  |  "
            f">=0.8: {(sim >= 0.8).mean():.1%}   >=0.6: {(sim >= 0.6).mean():.1%}   "
            f"exact (1.0): {(sim >= 0.999).mean():.1%}")
