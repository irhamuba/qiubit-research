"""Bit-exact Python port of amlc lib/vm/machine/fixed_q16.ml (Zarith semantics).

Z.div / Z.rem truncate toward zero; Z.shift_right floors (same as Python >>).
"""
from math import isqrt

FRAC = 16
SCALE = 1 << FRAC
MIN_EXP = -16 * SCALE
MAX_EXP = 8 * SCALE
MAX_RAW = (1 << 63) - 1
PI, HALF_PI, TWO_PI, LN_TWO = 205887, 102944, 411775, 45426


def zdiv(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def zrem(a: int, b: int) -> int:
    return a - b * zdiv(a, b)


def tmul(a: int, b: int) -> int:
    return (a * b) >> FRAC


def in_range(v: int) -> bool:
    return abs(v) <= MAX_RAW


def round16(v: int) -> int:
    half = 1 << (FRAC - 1)
    return (v + half) >> FRAC if v >= 0 else -((-v + half) >> FRAC)


def round_div(n: int, d: int) -> int:
    half = d >> 1
    return zdiv(n + half, d) if n >= 0 else -zdiv(-n + half, d)


def exp(v: int) -> int:
    v = min(max(v, MIN_EXP), MAX_EXP)
    e = round_div(v, LN_TWO)
    r = v - e * LN_TWO
    term = total = SCALE
    for i in range(1, 13):
        term = zdiv(tmul(term, r), i)
        total += term
    if e >= 0:
        return min(total << e, MAX_RAW)
    s = -e
    return (total + (1 << (s - 1))) >> s


def inv_sqrt(v: int) -> int:
    root = isqrt(v * SCALE)
    return zdiv(SCALE * SCALE, root)


def ln(v: int) -> int:
    e = 0
    while v >= 2 * SCALE:
        v >>= 1
        e += 1
    while v < SCALE:
        v <<= 1
        e -= 1
    z = zdiv((v - SCALE) * SCALE, v + SCALE)
    z2 = tmul(z, z)
    power, acc = z, 0
    for i in range(16):
        acc += zdiv(power, 2 * i + 1)
        power = tmul(power, z2)
    return 2 * acc + e * LN_TWO


def _sin0(x):
    sq = tmul(x, x); cu = tmul(sq, x); fi = tmul(cu, sq); se = tmul(fi, sq)
    return (x - zdiv(cu, 6)) + (zdiv(fi, 120) - zdiv(se, 5040))


def _cos0(x):
    sq = tmul(x, x); fo = tmul(sq, sq); si = tmul(fo, sq)
    return (SCALE - zdiv(sq, 2)) + (zdiv(fo, 24) - zdiv(si, 720))


def sin_cos(angle):
    r = zrem(angle, TWO_PI)
    a = r + TWO_PI if r < 0 else r
    if a <= HALF_PI:
        return _sin0(a), _cos0(a)
    if a <= PI:
        x = PI - a
        return _sin0(x), -_cos0(x)
    if a <= PI + HALF_PI:
        x = a - PI
        return -_sin0(x), -_cos0(x)
    x = TWO_PI - a
    return -_sin0(x), _cos0(x)


def silu(v: int) -> int:
    return tmul(v, zdiv(SCALE * SCALE, SCALE + exp(-v)))


def rope(vals, base, pos):
    n = len(vals)
    half = n // 2
    lb = ln(base)
    out = list(vals)
    for i in range(half):
        freq = exp(-tmul(zdiv(2 * i * SCALE, n), lb))
        s, c = sin_cos(pos * freq)
        out[i] = tmul(vals[i], c) - tmul(vals[i + half], s)
        out[i + half] = tmul(vals[i], s) + tmul(vals[i + half], c)
    return out


def rms(vals, gamma):
    n = len(vals)
    mean_sq = zdiv(sum(v * v for v in vals), n * SCALE)
    inv = inv_sqrt(mean_sq + 1)
    return [tmul(g, tmul(v, inv)) for v, g in zip(vals, gamma)]


def matvec(w_rows, x):
    """MATMUL_Q16 with rhs = column vector: round16 of exact sum."""
    return [round16(sum(a * b for a, b in zip(row, x))) for row in w_rows]


def attention(q, k, v, total, qh, kh, hd):
    inv_sqrt_dim = inv_sqrt(hd * SCALE)
    group = qh // kh
    out = []
    for h in range(qh):
        kvh = h // group
        qo = h * hd
        scores = []
        for t in range(total):
            ko = (t * kh + kvh) * hd
            d = sum(tmul(q[qo + i], k[ko + i]) for i in range(hd))
            scores.append(tmul(d, inv_sqrt_dim))
        m = max(scores)
        w = [exp(s - m) for s in scores]
        tot = sum(w)
        p = [zdiv(x * SCALE, tot) for x in w]
        for d in range(hd):
            out.append(sum(tmul(p[t], v[(t * kh + kvh) * hd + d]) for t in range(total)))
    return out
