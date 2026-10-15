"""Float32 forward pass, mirrors llama2.c run.c. Ground truth for the int port."""
import sys

import numpy as np

from model import Tokenizer, load_checkpoint


def rmsnorm(x, g):
    return g * x / np.sqrt(np.mean(x * x) + 1e-5)


def rope(v, pos, head):
    out = v.copy()
    for i in range(0, len(v), 2):
        freq = 1.0 / 10000.0 ** ((i % head) / head)
        c, s = np.cos(pos * freq), np.sin(pos * freq)
        out[i], out[i + 1] = v[i] * c - v[i + 1] * s, v[i] * s + v[i + 1] * c
    return out


class FloatModel:
    def __init__(self):
        self.cfg, self.w = load_checkpoint()
        c = self.cfg
        self.kc = np.zeros((c["n_layers"], c["seq_len"], c["kv_dim"]), np.float32)
        self.vc = np.zeros_like(self.kc)

    def forward(self, token, pos):
        c, w = self.cfg, self.w
        hd, rep = c["head"], c["n_heads"] // c["n_kv"]
        x = w["tok"][token].copy()
        for l in range(c["n_layers"]):
            xb = rmsnorm(x, w["rms_att"][l])
            q = rope(w["wq"][l] @ xb, pos, hd)
            self.kc[l, pos] = rope(w["wk"][l] @ xb, pos, hd)
            self.vc[l, pos] = w["wv"][l] @ xb
            att_out = np.zeros(c["dim"], np.float32)
            for h in range(c["n_heads"]):
                kvh = h // rep
                qh = q[h * hd:(h + 1) * hd]
                k = self.kc[l, :pos + 1, kvh * hd:(kvh + 1) * hd]
                v = self.vc[l, :pos + 1, kvh * hd:(kvh + 1) * hd]
                a = k @ qh / np.sqrt(hd)
                a = np.exp(a - a.max())
                att_out[h * hd:(h + 1) * hd] = (a / a.sum()) @ v
            x = x + w["wo"][l] @ att_out
            xb = rmsnorm(x, w["rms_ffn"][l])
            h1, h3 = w["w1"][l] @ xb, w["w3"][l] @ xb
            x = x + w["w2"][l] @ (h1 / (1 + np.exp(-h1)) * h3)
        return w["cls"] @ rmsnorm(x, w["rms_final"])


def generate(model, tok, prompt, steps):
    tokens = tok.encode(prompt)
    out, cur = [], tokens[0]
    for pos in range(steps):
        logits = model.forward(cur, pos)
        nxt = tokens[pos + 1] if pos + 1 < len(tokens) else int(np.argmax(logits))
        if nxt == 1:
            break
        out.append(tok.decode(cur, nxt))
        cur = nxt
    return "".join(out), logits


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    text, _ = generate(FloatModel(), Tokenizer(), prompt, steps)
    print(text)
