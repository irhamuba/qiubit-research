"""Upload weights/tensors.json to a deployed TinyStories program via set_tensor.

Needs a running webcli (tools/webcli/octra_wallet) with the owner wallet loaded and
OCTRA_PIN set. Progress is kept in build/upload.json so reruns skip sent tensors.

usage: python3 upload.py <program_address> [ou]
"""
import json
import sys
from pathlib import Path

import octra_api as api
from model import WEIGHTS

STATE = Path(__file__).resolve().parent.parent / "build/upload.json"  # per-address: move aside before a new deploy


def main():
    addr = sys.argv[1]
    ou = sys.argv[2] if len(sys.argv) > 2 else "1000"
    tensors = json.loads((WEIGHTS / "tensors.json").read_text())
    sent = json.loads(STATE.read_text()) if STATE.exists() else {}
    nonce = api.account_nonce()
    for name, data in tensors.items():
        if name in sent:
            continue
        nonce += 1
        r = api.send("/api/program/call", {
            "address": addr, "method": "set_tensor",
            "params": [name, data], "ou": ou}, nonce)
        sent[name] = r["tx_hash"]
        STATE.write_text(json.dumps(sent, indent=1))
        print(f"{len(sent)}/{len(tensors)} {name} ({len(data)} chars) {r['tx_hash']}")


if __name__ == "__main__":
    main()
