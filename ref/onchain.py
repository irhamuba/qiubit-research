"""Talk to the deployed TinyStories program through a running webcli.

usage:
  python3 onchain.py check <addr>                 # every set_tensor receipt succeeded
  python3 onchain.py generate <addr> [steps] [prompt]
"""
import json
import sys
from pathlib import Path

import octra_api as api
from export import pack_tokens
from int_ref import IntModel, generate
from model import Tokenizer

STATE = Path(__file__).resolve().parent.parent / "build/upload.json"
CALLER = "oct41fQ5hpaarA4ZuNpy3DJxvSRq9RwFs7TMKSTHZRkce5p"


def view(addr, method, params):
    """contract_call straight to the RPC (webcli's view proxy times out after 15 s)."""
    return api.rpc("contract_call", [addr, method, params, CALLER])


def check():
    sent = json.loads(STATE.read_text())
    bad = {}
    for name, h in sent.items():
        r = api.get(f"/api/contract/receipt?hash={h}")
        if not r.get("success"):
            bad[name] = r
    print(f"{len(sent) - len(bad)}/{len(sent)} receipts ok")
    for name, r in bad.items():
        print(name, str(r)[:200])


def run(addr, steps, prompt):
    tok = Tokenizer()
    ids = tok.encode(prompt)
    r = view(addr, "generate", [pack_tokens(ids), len(ids), steps])
    print(json.dumps(r)[:1000])
    got = str((r.get("result") or {}).get("result", ""))
    text, want = generate(IntModel(), tok, prompt, steps)
    want = ",".join(map(str, want))
    print("on-chain :", got)
    print("reference:", want)
    print("MATCH" if got == want else "MISMATCH", "|", text)


if __name__ == "__main__":
    cmd, addr = sys.argv[1], sys.argv[2]
    if cmd == "check":
        check()
    else:
        run(addr, int(sys.argv[3]) if len(sys.argv) > 3 else 5,
            sys.argv[4] if len(sys.argv) > 4 else "")
