"""ESM-2 protein language model embeddings (offline, CPU, cached to disk).

Each sequence becomes a fixed-size vector (320 numbers for the 8M model):
the mean of the model's per-residue vectors. The model learned from ~65M
natural proteins, so the vector already "knows" a lot of biochemistry.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

from .load import clean_seq

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never touch venue wifi
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

DEFAULT_MODEL = "facebook/esm2_t6_8M_UR50D"
CACHE = Path(__file__).resolve().parent.parent / "cache"


def embed(seqs, model_name=DEFAULT_MODEL, batch_size=32, max_len=1022, threads=8, verbose=True):
    import torch
    from transformers import AutoModel, AutoTokenizer

    seqs = [clean_seq(s)[:max_len] or "X" for s in seqs]
    key = hashlib.sha1((model_name + "\n" + "\n".join(seqs)).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"esm_{model_name.split('/')[-1]}_{key}.npy"
    if f.exists():
        if verbose:
            print(f"[esm] cache hit {f.name}")
        return np.load(f)

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    order = np.argsort([len(s) for s in seqs])        # batch similar lengths: less padding, less RAM
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(seqs), batch_size):
            idx = order[i:i + batch_size]
            batch = [seqs[j] for j in idx]
            enc = tok(batch, return_tensors="pt", padding=True)
            h = model(**enc).last_hidden_state
            for k, j in enumerate(idx):
                L = len(batch[k])
                out[j] = h[k, 1:L + 1].mean(0).numpy()  # skip <cls> and <eos>
            if verbose and (i // batch_size) % 20 == 0:
                print(f"[esm] {min(i + batch_size, len(seqs))}/{len(seqs)}")
    np.save(f, out)
    return out
