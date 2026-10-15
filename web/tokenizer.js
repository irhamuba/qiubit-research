// tok512 BPE tokenizer, byte-exact port of ref/model.py (llama2.c run.c).
// Pieces are latin1 strings: one char per byte, so merges compare raw bytes.

const BOS = 1;

function utf8ToLatin1(text) {
  return Array.from(new TextEncoder().encode(text), (b) => String.fromCharCode(b)).join("");
}

export function createTokenizer({ pieces, scores }) {
  const lookup = new Map(pieces.map((p, i) => [p, i]));

  function encode(text) {
    const tokens = [BOS];
    if (text) tokens.push(lookup.get(" "));
    for (const ch of utf8ToLatin1(text)) {
      tokens.push(lookup.has(ch) ? lookup.get(ch) : ch.charCodeAt(0) + 3);
    }
    for (;;) {
      let best = -1e10, bestI = -1, bestId = -1;
      for (let i = 0; i < tokens.length - 1; i++) {
        const id = lookup.get(pieces[tokens[i]] + pieces[tokens[i + 1]]);
        if (id !== undefined && scores[id] > best) {
          best = scores[id]; bestI = i; bestId = id;
        }
      }
      if (bestI < 0) return tokens;
      tokens.splice(bestI, 2, bestId);
    }
  }

  // bytes of one token as it follows `prev` (strip the space after BOS, expand <0xXX>)
  function pieceBytes(prev, tok) {
    let piece = pieces[tok];
    if (prev === BOS && piece.startsWith(" ")) piece = piece.slice(1);
    const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
    if (m) return [parseInt(m[1], 16)];
    return Array.from(piece, (c) => c.charCodeAt(0));
  }

  // text of each token in `ids` (which continue after BOS); multi-byte chars stay intact
  function decodeEach(ids) {
    const dec = new TextDecoder("utf-8");
    let prev = BOS;
    return ids.map((tok) => {
      const text = dec.decode(new Uint8Array(pieceBytes(prev, tok)), { stream: true });
      prev = tok;
      return text;
    });
  }

  return { encode, decodeEach };
}

// prompt ids -> base64 of hi|lo int8 planes (ref/export.py pack_tokens)
export function packTokens(ids) {
  const out = new Uint8Array(ids.length * 2);
  ids.forEach((t, i) => {
    const lo = ((t + 128) % 256) - 128;
    out[i] = ((t - lo) / 256) & 0xff;
    out[ids.length + i] = lo & 0xff;
  });
  return btoa(String.fromCharCode(...out));
}
