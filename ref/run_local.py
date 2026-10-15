"""Run aml/tinystories.aml in the local amlc VM and assert it matches int_ref.py bit for bit.

usage: [LIMIT=effort] python3 run_local.py [steps] [prompt]
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from export import pack_tokens, storage_key
from int_ref import IntModel, generate
from model import WEIGHTS, Tokenizer

ROOT = Path(__file__).resolve().parent.parent
AMLC = ROOT / "tools/amlc/_build/install/default/bin/amlc"
SOURCE = ROOT / "aml/tinystories.aml"
LIMIT = int(os.environ.get("LIMIT", 1_000_000_000_000))


def run_vm(prompt_ids, steps):
    tensors = json.loads((WEIGHTS / "tensors.json").read_text())
    cmd = [str(AMLC), "run", str(SOURCE), "--method", "generate",
           "--arg", pack_tokens(prompt_ids), "--arg", str(len(prompt_ids)), "--arg", str(steps),
           "--limit", str(LIMIT), "--step-cap", str(LIMIT), "--view"]
    for name, data in tensors.items():
        cmd += ["--set", f"{storage_key(name)}={data}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    tok = Tokenizer()
    code, out, err = run_vm(tok.encode(prompt), steps)
    print(out[-3000:], err[-3000:], sep="\n")
    if code != 0:
        sys.exit(code)
    text, ids = generate(IntModel(), tok, prompt, steps)
    want = ",".join(map(str, ids))
    m = re.search(r"result = text:(\S+)", out)
    got = m.group(1) if m else ""
    if got.startswith("size:"):  # amlc prints long strings as size:<n>:sha256:<hex>
        want = f"size:{len(want)}:sha256:{hashlib.sha256(want.encode()).hexdigest()}"
    assert got == want, f"VM {got} != reference {want}"
    print("MATCH:", text)


if __name__ == "__main__":
    main()
