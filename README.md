<p align="center"><img src="web/icon-128.png" width="72" alt="qiubit"></p>

# qiubit: a Llama 2 model running inside the Octra VM

This repository ports [stories260K](https://huggingface.co/karpathy/tinyllamas), a 260K-parameter
Llama 2 model, to an AML program. The program runs transformer inference inside the Octra VM in
Q16 fixed point. The model is live on Octra devnet, and its output matches a float32 reference
bit for bit. The `web/` folder is a chat-style page that lets anyone use the model with no wallet.

- Paper: [`docs/paper.pdf`](docs/paper.pdf)
- Live program (v2): `oct4LetvjEP7fzTyvaUBSu7aDKBtM1peufyCBmSDvos5HTR` on
  [devnet](https://devnet.octrascan.io/program.html?addr=oct4LetvjEP7fzTyvaUBSu7aDKBtM1peufyCBmSDvos5HTR)

## What it can do

| | |
|---|---|
| Model | dim 64, 5 layers, 8 heads / 4 KV heads, vocab 512; continues simple English children's stories |
| Weights on-chain | 83 tensors (int8 rows + Q16 scales), uploaded with `set_tensor` |
| Inference | `generate(prompt_b64, n, steps)` as a view call: no wallet, no fee |
| Limit | 9 new tokens with no prompt, or 10 positions per call shared between prompt and reply |
| Verified | `code_hash` on-chain equals SHA-256 of the bytecode compiled from `aml/` |
| Not | private (plaintext compute) or recorded on-chain (view calls run on one node) |

The paper explains these limits. In short, the node compiler rejects `matmul_q16`, so
every matrix row costs one `vecdot_q16`. The node then caps a call at roughly 31–35 thousand
vector operations.

## Layout

```
aml/       tinystories_deployed.aml (v1), tinystories_v2.aml (v2), tinystories.aml (current = v2)
ref/       Python: float32 reference, bit-exact Q16 emulator, quantizer, exporter,
           local VM runner, deploy/upload/evidence scripts for devnet
weights/   stories260K.bin, tok512.bin (MIT), tensors.json (exported upload payload)
web/       qiubit chat UI: static HTML/CSS/JS, in-browser tokenizer, calls the devnet RPC
docs/      paper.tex / paper.pdf
build/     raw on-chain evidence: deploy, upload, and generate results for v1 and v2
```

## Reproduce

Requirements: Python 3 with numpy, and Node 18+ for the tokenizer test. The VM steps also need
Octra's [`amlc`](https://github.com/octra-labs/amlc) built at `tools/amlc`, and the devnet steps
need [`webcli`](https://github.com/octra-labs/webcli) running at `127.0.0.1:8420`.

```sh
cd ref
python3 float_ref.py 60              # float32 reference (matches the model card)
python3 int_ref.py 60                # Q16 emulator: same text
python3 export.py                    # -> weights/tensors.json
LIMIT=2000000 python3 run_local.py 9 "Once upon a time, Tom"   # local VM, asserts bit-exact

# devnet: webcli running with the owner wallet loaded, OCTRA_PIN set
python3 deploy.py ../aml/tinystories.aml
python3 upload.py <program_address>
python3 evidence.py <program_address> ../aml/tinystories.aml 9,10
python3 onchain.py generate <program_address> 9 "Once upon a time, Tom"
```

`node web/tokenizer.test.mjs` (from the repo root) checks the browser tokenizer against Python.

## Website

`web/` is fully static. Run it locally with `cd web && python3 -m http.server`. To deploy on
Vercel, set the project root to `web/` with framework preset "Other" and no build step;
`web/vercel.json` sets the security headers. See [`web/README.md`](web/README.md).

## Licenses

The weights, tokenizer, and `llama2.c` reference are MIT (Andrej Karpathy); see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The model was trained on TinyStories
(CDLA-Sharing-1.0). Octra's `amlc` (BSD-3-Clause) and `webcli` (GPL-2.0-or-later) are used as
external tools and are not included.
