"""int8 per-row quantization in the layout the Octra VM consumes via load_int8_q16.

Each row: (scale_raw, bytes). VM value = int8 * scale_raw, interpreted as Q16.
RoPE: llama2.c rotates interleaved pairs (2j, 2j+1); the VM rotates half-split
pairs (j, j+half). Permuting wq/wk rows inside each head maps one onto the other.
"""
import numpy as np

from model import load_checkpoint

SCALE = 1 << 16


def quant_rows(m):
    m = np.atleast_2d(m).astype(np.float64)
    rows = []
    for r in m:
        scale = max(1, round(float(np.abs(r).max()) / 127 * SCALE))
        q = np.clip(np.round(r * SCALE / scale), -127, 127).astype(np.int8)
        rows.append((scale, q.tobytes()))
    return rows


def rope_perm(n_rows, head):
    per_head = list(range(0, head, 2)) + list(range(1, head, 2))
    return [h * head + i for h in range(n_rows // head) for i in per_head]


def q16_vec(v):
    return [int(round(float(x) * SCALE)) for x in v]


def quantize():
    """Matrices -> per-row int8; RMSNorm gammas -> exact Q16 ints (prefixed g_)."""
    cfg, w = load_checkpoint()
    hd = cfg["head"]
    qperm, kperm = rope_perm(cfg["dim"], hd), rope_perm(cfg["kv_dim"], hd)
    t = {"tok": quant_rows(w["tok"]), "g_rms_final": q16_vec(w["rms_final"])}
    for l in range(cfg["n_layers"]):
        t[f"g_rms_att{l}"] = q16_vec(w["rms_att"][l])
        t[f"wq{l}"] = quant_rows(w["wq"][l][qperm])
        t[f"wk{l}"] = quant_rows(w["wk"][l][kperm])
        t[f"wv{l}"] = quant_rows(w["wv"][l])
        t[f"wo{l}"] = quant_rows(w["wo"][l])
        t[f"g_rms_ffn{l}"] = q16_vec(w["rms_ffn"][l])
        t[f"w1{l}"] = quant_rows(w["w1"][l])
        t[f"w2{l}"] = quant_rows(w["w2"][l])
        t[f"w3{l}"] = quant_rows(w["w3"][l])
    return cfg, t


def int8_row(row):
    return [int(v) for v in np.frombuffer(row[1], np.int8)]
