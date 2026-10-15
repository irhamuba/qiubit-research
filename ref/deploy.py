"""Compile aml/tinystories.aml on the node and deploy it. Needs webcli + OCTRA_PIN.

usage: python3 deploy.py [source.aml]   -> prints and saves build/deploy.json
"""
import json
import sys
import time
from pathlib import Path

import octra_api as api

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "aml/tinystories.aml").read_text()
    for _ in range(10):
        c = api.post("/api/contract/compile-aml", {"source": src, "program": True})
        if "bytecode" in c:
            break
        print("compile:", str(c)[:200])
        time.sleep(5)
    else:
        sys.exit("compile failed")
    nonce = api.account_nonce() + 1
    r = api.send("/api/contract/deploy", {
        "bytecode": c["bytecode"], "deploy_payload": c["deploy_payload"],
        "ou": c["program_min_ou"]}, nonce)
    r.update(nonce=nonce, ou=c["program_min_ou"], instructions=c["instructions"])
    (ROOT / "build/deploy.json").write_text(json.dumps(r, indent=1))
    print(r)


if __name__ == "__main__":
    main()
