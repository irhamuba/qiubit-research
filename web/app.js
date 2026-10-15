import { createTokenizer, packTokens } from "./tokenizer.js";

const RPC = "https://devnet.octrascan.io/rpc";
const PROGRAM = "oct4LetvjEP7fzTyvaUBSu7aDKBtM1peufyCBmSDvos5HTR";
const CODE_HASH = "84ad4f9bfc4d95a937d31a13a850e2bf74fb0c70e191ce58572b6fefa74d06c9";
const EXPLORER = `https://devnet.octrascan.io/program.html?addr=${PROGRAM}`;
// view calls need a caller address but no signature; any well-formed address works
const CALLER = "oct1111111111111111111111111111111111111111111";

// Measured on devnet: a call passes with <= 31,608 vecdot_q16 calls. Every position runs
// 3,000 rows through the 5 layers; generated positions add the 512-row classifier.
const ROWS_PER_POSITION = 3000;
const ROWS_CLASSIFIER = 512;
const ROW_BUDGET = 31608;
const CELLS = 10;
const MAX_TRIES = 8;
const TRANSIENT = /busy|capacity|bytecode not found|rate|timeout/i;

const $ = (id) => document.getElementById(id);
const els = {
  thread: $("thread"), empty: $("empty"), form: $("composer"), prompt: $("prompt"),
  cells: $("cells"), note: $("budget-note"), count: $("new-tokens"),
  minus: $("minus"), plus: $("plus"), run: $("run"),
};

let tokenizer = null;
let wanted = 3;
let busy = false;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Prompt ids and the number of new tokens that fit the on-chain budget. */
function plan() {
  const ids = tokenizer ? tokenizer.encode(els.prompt.value.trim()) : [1];
  const given = ids.length - 1;
  const limit = Math.floor((ROW_BUDGET - ROWS_PER_POSITION * given) / (ROWS_PER_POSITION + ROWS_CLASSIFIER));
  const fresh = Math.max(0, Math.min(wanted, limit));
  return { ids, given, limit, fresh };
}

function renderComposer() {
  const { given, limit, fresh } = plan();
  els.cells.replaceChildren(...Array.from({ length: CELLS }, (_, i) => {
    const c = el("span", "cell");
    if (i < given) c.classList.add(given > CELLS ? "over" : "prompt-cell");
    else if (i < given + fresh) c.classList.add("new-cell");
    return c;
  }));
  els.count.value = `+${fresh} tokens`;
  els.minus.disabled = busy || fresh <= 1;
  els.plus.disabled = busy || fresh >= limit;
  els.run.disabled = busy || !tokenizer || limit < 1;
  const over = limit < 1;
  els.note.textContent = over ? "Prompt too long" : `${given} prompt + ${fresh} new / ${CELLS}`;
  els.note.classList.toggle("warn", over);
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${els.prompt.scrollHeight}px`;
}

async function rpcOnce(params) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 120000);
  try {
    const res = await fetch(RPC, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "contract_call", params }),
      signal: ctrl.signal,
    });
    if (res.status === 429 || res.status >= 500) return { transient: `HTTP ${res.status}` };
    const body = await res.json();
    if (body.error) {
      const msg = `${body.error.message || ""} ${body.error.data || ""}`.trim();
      return TRANSIENT.test(msg) ? { transient: msg } : { error: msg };
    }
    return { result: body.result?.result ?? "" };
  } catch (err) {
    return { transient: err instanceof Error ? err.message : "network" };
  } finally {
    clearTimeout(timer);
  }
}

/** contract_call with retries: devnet RPC fans out to nodes that lag or are busy. */
async function generate(ids, steps, onAttempt) {
  const params = [PROGRAM, "generate", [packTokens(ids), ids.length, steps], CALLER];
  let last = "";
  for (let attempt = 1; attempt <= MAX_TRIES; attempt++) {
    onAttempt(attempt);
    const r = await rpcOnce(params);
    if (!r.transient) return { ...r, attempt };
    last = r.transient;
    await new Promise((ok) => setTimeout(ok, 1500 * attempt));
  }
  return { error: `The RPC stayed busy (${last}). Try again in a moment.`, attempt: MAX_TRIES };
}

function scrollToEnd() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function addUserMessage(text) {
  const row = el("div", "msg user");
  row.append(el("div", "bubble", text || "(no prompt)"));
  els.thread.append(row);
}

function addBotMessage() {
  const row = el("div", "msg bot");
  const answer = el("div", "answer");
  const dots = el("div", "thinking");
  dots.append(el("span"), el("span"), el("span"));
  const progress = el("p", "progress", "Octra node is computing…");
  answer.append(dots, progress);
  row.append(el("div", "avatar"), answer);
  els.thread.append(row);
  return { answer, progress };
}

function renderAnswer(answer, outIds, given, info) {
  const texts = tokenizer.decodeEach(outIds);
  const p = el("p", "answer-text");
  texts.forEach((text, i) => {
    const span = el("span", `tok ${i < given ? "given" : "made"}`, text);
    span.title = `token ${outIds[i]}`;
    span.style.animationDelay = `${(i - given > 0 ? i - given : 0) * 90}ms`;
    p.append(span);
  });

  const details = el("dl", "details");
  details.hidden = true;
  const rows = [
    ["program", PROGRAM], ["method", "contract_call → generate"], ["positions", String(info.steps)],
    ["token ids", outIds.join(", ")], ["attempts", String(info.attempt)],
  ];
  rows.forEach(([k, v]) => details.append(el("dt", "", k), el("dd", "", v)));

  const copy = el("button", "meta-btn", "Copy");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(texts.join(""));
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Copy failed";
    }
    setTimeout(() => { copy.textContent = "Copy"; }, 1500);
  });
  const toggle = el("button", "meta-btn", "On-chain details");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", () => {
    details.hidden = !details.hidden;
    toggle.setAttribute("aria-expanded", String(!details.hidden));
  });
  const link = el("a", "meta-btn", "Explorer ↗");
  link.href = EXPLORER;
  link.target = "_blank";
  link.rel = "noopener";

  const meta = el("div", "meta");
  meta.append(copy, toggle, link, el("span", "meta-stat", `${(info.ms / 1000).toFixed(1)} s`));
  const made = outIds.length - given;
  const children = [p, meta, details];
  if (made < info.fresh) children.splice(1, 0, el("p", "progress", `The model stopped early (${made} new tokens).`));
  answer.replaceChildren(...children);
}

function renderError(answer, message) {
  answer.classList.add("error");
  answer.replaceChildren(el("p", "answer-text", message));
}

async function send(text) {
  els.prompt.value = text;
  const { ids, given, fresh } = plan();
  if (busy || fresh < 1) return;
  busy = true;
  els.empty.hidden = true;
  addUserMessage(text);
  els.prompt.value = "";
  renderComposer();
  const { answer, progress } = addBotMessage();
  scrollToEnd();

  const steps = given + fresh;
  const t0 = performance.now();
  const r = await generate(ids, steps, (n) => {
    progress.textContent = n === 1 ? "Octra node is computing…" : `Node busy, retrying (${n}/${MAX_TRIES})…`;
  });
  const ms = performance.now() - t0;

  if (r.error !== undefined) {
    renderError(answer, /revert/i.test(r.error) ? "The node rejected the call: compute budget exceeded. Ask for fewer new tokens." : r.error);
  } else {
    const outIds = r.result.split(",").filter(Boolean).map(Number);
    if (outIds.some((n) => !Number.isInteger(n) || n < 0 || n > 511)) {
      renderError(answer, "Could not read the node's reply.");
    } else {
      renderAnswer(answer, outIds, given, { steps, ms, attempt: r.attempt, fresh });
    }
  }
  busy = false;
  renderComposer();
  scrollToEnd();
  els.prompt.focus();
}

function init() {
  $("program-link").textContent = PROGRAM;
  $("program-link").href = EXPLORER;
  $("hash-full").textContent = CODE_HASH;

  const info = $("info");
  $("info-toggle").addEventListener("click", (e) => {
    info.hidden = !info.hidden;
    e.currentTarget.setAttribute("aria-expanded", String(!info.hidden));
  });
  $("new-chat").addEventListener("click", () => {
    if (busy) return;
    els.thread.replaceChildren(els.empty);
    els.empty.hidden = false;
    els.prompt.value = "";
    renderComposer();
  });
  $("suggestions").addEventListener("click", (e) => {
    if (e.target instanceof HTMLButtonElement) send(e.target.textContent);
  });
  els.form.addEventListener("submit", (e) => { e.preventDefault(); send(els.prompt.value.trim()); });
  els.prompt.addEventListener("input", renderComposer);
  els.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); els.form.requestSubmit(); }
  });
  els.minus.addEventListener("click", () => { wanted = Math.max(1, plan().fresh - 1); renderComposer(); });
  els.plus.addEventListener("click", () => { wanted = plan().fresh + 1; renderComposer(); });
  renderComposer();

  fetch("vocab.json")
    .then((res) => res.json())
    .then((vocab) => { tokenizer = createTokenizer(vocab); renderComposer(); els.prompt.focus(); })
    .catch(() => { els.note.textContent = "Failed to load the tokenizer"; els.note.classList.add("warn"); });
}

init();
