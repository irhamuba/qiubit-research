# qiubit chat UI (web)

A static page that calls the `generate` view on the TinyStories program on Octra devnet
(`oct4LetvjEP7fzTyvaUBSu7aDKBtM1peufyCBmSDvos5HTR`) straight from the browser. It has no backend and needs no
wallet or OCT. The devnet RPC sends `access-control-allow-origin: *`, and a view call accepts any
well-formed caller address.

## Files

- `index.html`, `style.css`, `app.js`, `icon-128.png`, `favicon.ico`: the page
- `tokenizer.js` + `vocab.json`: tok512 BPE tokenizer, byte-exact with `ref/model.py`
- `tokenizer.test.mjs`: parity check against Python (`node web/tokenizer.test.mjs` from the repo root)
- `vercel.json`: security headers (CSP, HSTS, frame-ancestors, and others)

## Run locally

```sh
cd web && python3 -m http.server 8765
# open http://127.0.0.1:8765
```

## Deploy to Vercel

Set the project root to `web/`. No build step is needed (framework preset "Other"), and the output directory is `.`.

## Budget rule

A call passes on devnet when it makes at most 31,608 `vecdot_q16` calls. Each position costs
3,000 rows, and each newly generated token adds 512 rows for the classifier. `app.js` uses this to
cap new tokens: 9 with no prompt, and 1 after a 9-token prompt. The RPC sometimes replies with `429`,
`Program view busy`, or `bytecode not found`, so the page retries up to 8 times.
