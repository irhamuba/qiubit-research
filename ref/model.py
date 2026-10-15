"""Load llama2.c checkpoint + tokenizer (stories260K) into numpy."""
import struct
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


def load_checkpoint(path=WEIGHTS / "stories260K.bin"):
    raw = Path(path).read_bytes()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len = struct.unpack("7i", raw[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    head = dim // n_heads
    kv_dim = n_kv * head
    f = np.frombuffer(raw[28:], dtype=np.float32)
    off = 0

    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        out = f[off:off + n].reshape(shape)
        off += n
        return out

    w = {
        "tok": take(vocab, dim),
        "rms_att": take(n_layers, dim),
        "wq": take(n_layers, dim, dim),
        "wk": take(n_layers, kv_dim, dim),
        "wv": take(n_layers, kv_dim, dim),
        "wo": take(n_layers, dim, dim),
        "rms_ffn": take(n_layers, dim),
        "w1": take(n_layers, hidden, dim),
        "w2": take(n_layers, dim, hidden),
        "w3": take(n_layers, hidden, dim),
        "rms_final": take(dim),
    }
    off += seq_len * head  # skip legacy freq_cis_real + freq_cis_imag
    w["cls"] = w["tok"] if shared else take(vocab, dim)
    cfg = dict(dim=dim, hidden=hidden, n_layers=n_layers, n_heads=n_heads,
               n_kv=n_kv, vocab=vocab, seq_len=seq_len, head=head, kv_dim=kv_dim)
    return cfg, w


class Tokenizer:
    def __init__(self, path=WEIGHTS / "tok512.bin", vocab=512):
        raw = Path(path).read_bytes()
        off = 4  # max_token_length
        self.vocab, self.scores = [], []
        for _ in range(vocab):
            score, n = struct.unpack_from("fi", raw, off)
            off += 8
            self.vocab.append(raw[off:off + n])
            self.scores.append(score)
            off += n
        self.lookup = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, text, bos=True):
        """BPE merge by best score, same as llama2.c run.c."""
        tokens = [1] if bos else []
        if text:
            tokens.append(self.lookup[b" "])  # dummy prefix
        for ch in text.encode():
            b = bytes([ch])
            tokens.append(self.lookup[b] if b in self.lookup else ch + 3)
        while True:
            best, best_i, best_id = -1e10, -1, -1
            for i in range(len(tokens) - 1):
                merged = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                tid = self.lookup.get(merged)
                if tid is not None and self.scores[tid] > best:
                    best, best_i, best_id = self.scores[tid], i, tid
            if best_i < 0:
                return tokens
            tokens[best_i:best_i + 2] = [best_id]

    def decode(self, prev, tok):
        piece = self.vocab[tok]
        if prev == 1 and piece.startswith(b" "):
            piece = piece[1:]
        if piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        return piece.decode("utf-8", errors="replace")
