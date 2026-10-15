"""Collect on-chain evidence for a deployed program into build/evidence.json.

usage: python3 evidence.py <addr> <source.aml> <steps,...>
Checks code_hash against the node's compile of <source.aml>, then runs generate for each
step count (no prompt) plus one prompted run, comparing every result with int_ref.py.
"""
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import octra_api as api
from export import pack_tokens
from int_ref import IntModel, generate
from model import Tokenizer
from onchain import view

ROOT = Path(__file__).resolve().parent.parent


def code_hash_of(src):
    for _ in range(10):
        c = api.post("/api/contract/compile-aml", {"source": src, "program": True})
        if "bytecode" in c:
            return hashlib.sha256(base64.b64decode(c["bytecode"])).hexdigest()
        time.sleep(5)
    return None


def run(tok, addr, prompt, steps):
    ids = tok.encode(prompt)
    t = time.time()
    r = view(addr, "generate", [pack_tokens(ids), len(ids), steps])
    dt = round(time.time() - t, 1)
    got = (r.get("result") or {}).get("result")
    text, want = generate(IntModel(), tok, prompt, steps)
    want = ",".join(map(str, want))
    row = {"prompt": prompt, "steps": steps, "seconds": dt, "response": r,
           "reference": want, "match": got == want, "text": text}
    print(steps, repr(prompt), "MATCH" if got == want else f"-> {str(r)[:120]}", f"{dt}s")
    return row


def main():
    addr, source, steps = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    tok = Tokenizer()
    contract = api.rpc("vm_contract", [addr])
    local_hash = code_hash_of(source.read_text())
    ev = {"time_unix": int(time.time()), "address": addr, "source": source.name,
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "contract": contract, "bytecode_sha256_from_source": local_hash,
          "code_hash_match": (contract.get("result") or {}).get("code_hash") == local_hash,
          "runs": []}
    print("code_hash match:", ev["code_hash_match"])
    for s in map(int, steps.split(",")):
        ev["runs"].append(run(tok, addr, "", s))
    ev["runs"].append(run(tok, addr, "Once upon a time, Tom", max(map(int, steps.split(",")))))
    out = ROOT / "build/evidence.json"
    out.write_text(json.dumps(ev, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
