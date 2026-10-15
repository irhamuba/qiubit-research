"""Q16 forward pass using only VM-equivalent ops. Must match the AML program bit for bit.

Matrices are loaded with scale 65536 (value = int8 as Q16 integer), multiplied, then
each output is scaled by its row scale via elemwise_mul_q16. Embedding rows are loaded
directly with their own scale (load_int8_q16 with offset).
"""
import sys

import q16
from model import Tokenizer
from quant import int8_row, quantize

ROPE_BASE = 10000 * q16.SCALE


class IntModel:
    def __init__(self):
        self.cfg, t = quantize()
        self.g = {k: v for k, v in t.items() if k.startswith("g_")}
        self.q = {k: [int8_row(r) for r in rows] for k, rows in t.items() if not k.startswith("g_")}
        self.s = {k: [r[0] for r in rows] for k, rows in t.items() if not k.startswith("g_")}
        self.kc = [[] for _ in range(self.cfg["n_layers"])]
        self.vc = [[] for _ in range(self.cfg["n_layers"])]

    def lin(self, name, x):
        y = q16.matvec([[v * q16.SCALE for v in r] for r in self.q[name]], x)
        return [q16.tmul(a, s) for a, s in zip(y, self.s[name])]

    def forward(self, token, pos):
        c, g = self.cfg, self.g
        hd = c["head"]
        x = [v * self.s["tok"][token] for v in self.q["tok"][token]]
        for l in range(c["n_layers"]):
            xb = q16.rms(x, g[f"g_rms_att{l}"])
            q = self.lin(f"wq{l}", xb)
            k = self.lin(f"wk{l}", xb)
            v = self.lin(f"wv{l}", xb)
            q = [y for h in range(c["n_heads"]) for y in q16.rope(q[h*hd:(h+1)*hd], ROPE_BASE, pos)]
            k = [y for h in range(c["n_kv"]) for y in q16.rope(k[h*hd:(h+1)*hd], ROPE_BASE, pos)]
            self.kc[l] += k
            self.vc[l] += v
            att = q16.attention(q, self.kc[l], self.vc[l], pos + 1, c["n_heads"], c["n_kv"], hd)
            x = [a + b for a, b in zip(x, self.lin(f"wo{l}", att))]
            xb = q16.rms(x, g[f"g_rms_ffn{l}"])
            h1 = [q16.silu(y) for y in self.lin(f"w1{l}", xb)]
            h3 = self.lin(f"w3{l}", xb)
            hh = [q16.tmul(a, b) for a, b in zip(h1, h3)]
            x = [a + b for a, b in zip(x, self.lin(f"w2{l}", hh))]
        x = q16.rms(x, g["g_rms_final"])
        return self.lin("tok", x)


def generate(model, tok, prompt, steps):
    tokens = tok.encode(prompt)
    out, ids, cur = [], [], tokens[0]
    for pos in range(steps):
        logits = model.forward(cur, pos)
        nxt = tokens[pos + 1] if pos + 1 < len(tokens) else max(range(len(logits)), key=logits.__getitem__)
        if nxt == 1:
            break
        out.append(tok.decode(cur, nxt))
        ids.append(nxt)
        cur = nxt
    return "".join(out), ids


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    text, ids = generate(IntModel(), Tokenizer(), prompt, steps)
    print(text)
    print(",".join(map(str, ids)))
