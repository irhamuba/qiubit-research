// Parity check vs the Python tokenizer: node web/tokenizer.test.mjs
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createTokenizer, packTokens } from "./tokenizer.js";

const vocab = JSON.parse(readFileSync(new URL("./vocab.json", import.meta.url)));
const tok = createTokenizer(vocab);
const cases = ["", "Once upon a time, Tom", "Lily saw a big, red ball!", "café naïve 123", "The dog\nran."];
const py = JSON.parse(execFileSync("python3", ["-c", `
import json, sys
sys.path.insert(0, "ref")
from model import Tokenizer
from export import pack_tokens
t = Tokenizer()
cases = json.loads(sys.argv[1])
out = []
for c in cases:
    ids = t.encode(c)
    raw, prev = b"", 1
    for i in ids[1:]:
        piece = t.vocab[i]
        if prev == 1 and piece.startswith(b" "):
            piece = piece[1:]
        if piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])
        raw += piece; prev = i
    out.append({"ids": ids, "text": raw.decode("utf-8"), "packed": pack_tokens(ids)})
print(json.dumps(out))
`, JSON.stringify(cases)]).toString());

cases.forEach((c, i) => {
  const ids = tok.encode(c);
  const text = tok.decodeEach(ids.slice(1)).join("");
  const ok = JSON.stringify(ids) === JSON.stringify(py[i].ids) && text === py[i].text
    && packTokens(ids) === py[i].packed;
  console.log(ok ? "ok  " : "FAIL", JSON.stringify(c), ids.join(","));
  if (!ok) process.exitCode = 1;
});
