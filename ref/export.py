"""Export quantized stories260K as the tensor map the AML program reads (state wt).

Output: weights/tensors.json  {name: data}
  <mat>     base64 int8 rows concatenated (row-major)
  s_<mat>   per-row Q16 scales, int24 split into 3 int8 planes (see pack_int24)
  g_<vec>   Q16 RMSNorm gammas, same int24 packing
The node's typed Program profile has no parse_ints, so every input is base64 int8.
On chain each entry is uploaded with set_tensor(name, data).
"""
import base64
import json

from model import WEIGHTS
from quant import quantize


def pack_int24(vals):
    """v = a*65536 + b*256 + c with a, b, c signed int8; planes a|b|c concatenated, base64.

    The program rebuilds v with three load_int8_q16 calls (scales 65536, 256, 1) and two adds.
    """
    planes = ([], [], [])
    for v in vals:
        c = (v + 128) % 256 - 128
        v1 = (v - c) // 256
        b = (v1 + 128) % 256 - 128
        a = (v1 - b) // 256
        assert -128 <= a <= 127, f"value {v} exceeds int24"
        for plane, x in zip(planes, (a, b, c)):
            plane.append(x & 0xFF)
    return base64.b64encode(bytes(planes[0] + planes[1] + planes[2])).decode()


def pack_tokens(ids):
    """token ids (< 32768) as two int8 planes: id = hi*256 + lo."""
    hi, lo = [], []
    for t in ids:
        l = (t + 128) % 256 - 128
        hi.append(((t - l) // 256) & 0xFF)
        lo.append(l & 0xFF)
    return base64.b64encode(bytes(hi + lo)).decode()


def export():
    _, t = quantize()
    out = {}
    for name, val in t.items():
        if name.startswith("g_"):
            out[name] = pack_int24(val)
        else:
            out[name] = base64.b64encode(b"".join(r[1] for r in val)).decode()
            out[f"s_{name}"] = pack_int24([r[0] for r in val])
    return out


def storage_key(name):
    """amlc lowers map[string] entries to @aml/map/<field>/<len>#<key>."""
    return f"@aml/map/wt/{len(name)}#{name}"


if __name__ == "__main__":
    tensors = export()
    path = WEIGHTS / "tensors.json"
    path.write_text(json.dumps(tensors))
    size = sum(len(v) for v in tensors.values())
    print(f"{len(tensors)} entries, {size} chars -> {path}")
