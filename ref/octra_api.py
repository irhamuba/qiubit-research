"""Minimal client for a locally running webcli (tools/webcli/octra_wallet) HTTP API.

PIN comes from the OCTRA_PIN environment variable; the private key never touches this code.
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

BASE = os.environ.get("WEBCLI", "http://127.0.0.1:8420")
HEADERS = {"Content-Type": "application/json", "Origin": BASE}


def _req(method, path, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace"), "status": e.code}


def get(path):
    return _req("GET", path)


def post(path, body, pin=False):
    if pin:
        body = {**body, "pin": os.environ["OCTRA_PIN"]}
    return _req("POST", path, body)


RPC = os.environ.get("OCTRA_RPC", "https://devnet.octrascan.io/rpc")
# devnet RPC fans out to nodes that lag each other; these replies are worth retrying
TRANSIENT = ("bytecode not found", "capacity", "busy", "error code: 5")


def rpc(method, params, tries=10, timeout=600):
    """JSON-RPC straight to the node. curl, because the RPC edge returns 403 to urllib."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    out = ""
    for _ in range(tries):
        out = subprocess.run(["curl", "-s", "-m", str(timeout), "-X", "POST", RPC,
                              "-H", "Content-Type: application/json", "-d", body],
                             capture_output=True, text=True).stdout
        if out and not any(t in out for t in TRANSIENT):
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                pass
        time.sleep(3)
    return {"error": out[:300]}


def send(path, body, nonce):
    """Submit a signed tx through webcli and make sure it used `nonce`.

    webcli reads the account nonce from whichever node answers; a lagging node makes it sign
    with a nonce that is already spent. Wait until /api/balance reports nonce - 1, submit,
    and resubmit if the node recorded a different nonce.
    """
    for _ in range(40):
        if get("/api/balance").get("nonce") != nonce - 1:
            time.sleep(3)
            continue
        r = post(path, body, pin=True)
        h = r.get("tx_hash")
        if not h:
            time.sleep(3)
            continue
        got = None
        for _ in range(10):  # a lagging node may not know the tx yet
            got = (rpc("octra_transaction", [h]).get("result") or {}).get("nonce")
            if got is not None:
                break
            time.sleep(3)
        if got == nonce:
            return r
        time.sleep(3)
    raise RuntimeError(f"could not submit with nonce {nonce}")


def account_nonce(reads=6):
    """Highest nonce reported over several reads (lagging nodes report stale values)."""
    return max(get("/api/balance").get("nonce", 0) for _ in range(reads))
