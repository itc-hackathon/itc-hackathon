"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const api = async (path, body) => {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(path + " -> " + res.status);
  return res.json();
};

/* ============================ DEMO TABS ============================ */
const DEMO_META = [
  { num: "01", label: "Long-horizon memory" },
  { num: "02", label: "Personalization" },
  { num: "03", label: "Self-improving skills" },
];
let activeDemo = 0; // open on the live long-horizon memory demo

function renderTabs() {
  const row = $("demoTabs");
  row.innerHTML = "";
  DEMO_META.forEach((d, i) => {
    const b = el("button", "tab" + (i === activeDemo ? " on" : ""));
    b.innerHTML = `<span class="tnum">${d.num}</span>${d.label}`;
    b.onclick = () => {
      activeDemo = i;
      renderTabs();
      [0, 1, 2].forEach((j) => ($(`demo-${j}`).style.display = j === i ? "" : "none"));
    };
    row.appendChild(b);
  });
}

/* ============================ DEMO 1: MEMORY (live, SSE) ============================ */
const M1 = {
  scenario: "apollo_migration",
  size: "medium",
  running: false,
  es: null,
  meta: null,
  scenarios: [],
  sizes: [],
};
const M1_SIZE_LABEL = { small: "Small", medium: "Medium", large: "Large" };
const M1_SCEN_LABEL = {
  apollo_migration: "Project log",
  trip_planning: "Trip planning",
  research_assistant: "Research notes",
};

function m1RenderControls() {
  const st = $("m1ScenarioTabs");
  st.innerHTML = "";
  M1.scenarios.forEach((name) => {
    const b = el("button", "btn-sm" + (name === M1.scenario ? " on" : ""), M1_SCEN_LABEL[name] || name);
    b.onclick = () => { if (!M1.running) { M1.scenario = name; m1RenderControls(); m1Reset(); } };
    st.appendChild(b);
  });
  const sz = $("m1Size");
  sz.innerHTML = "";
  M1.sizes.forEach((s) => {
    const turns = M1.meta && M1.meta.turns_per_size ? M1.meta.turns_per_size[s] : null;
    const b = el("button", "sizeseg" + (s === M1.size ? " on" : ""), M1_SIZE_LABEL[s] || s);
    if (turns) b.title = turns + " turns";
    b.onclick = () => { if (!M1.running) { M1.size = s; m1RenderControls(); m1Reset(); } };
    sz.appendChild(b);
  });
}

const m1mk = (cls, tag, text) => {
  const e = el("div", cls);
  if (tag) e.appendChild(el("span", "mtag", tag));
  if (text != null) e.appendChild(document.createTextNode(text));
  return e;
};

function m1Fills(nap, md, van) {
  const rows = [
    ["NapLoRA", "#2f6ae0", nap],
    ["Markdown .md", "#c79a3a", md],
    ["Vanilla (raw)", "#94908a", van],
  ];
  const box = $("m1Fills");
  box.innerHTML = "";
  rows.forEach(([name, color, m]) => {
    const pct = m ? m.fill_pct : 0;
    const tok = m ? m.context_tokens : 0;
    const over = m && m.overflow;
    const row = el("div", "fillrow");
    row.innerHTML =
      '<div class="filllabel"><span class="dot" style="background:' + color + '"></span>' + name + "</div>" +
      '<div class="filltrack"><div class="fillbar" style="width:' + Math.min(100, pct) + "%;background:" + (over ? "#c2554d" : color) + '"></div></div>' +
      '<div class="fillval" style="' + (over ? "color:#c2554d;font-weight:600" : "") + '">' + tok.toLocaleString() + " tok · " + pct + "%" + (over ? " ⚠ overflow" : "") + "</div>";
    box.appendChild(row);
  });
}

function m1CostCards(methods) {
  const order = [
    ["napora", "NapLoRA", "#2f6ae0"],
    ["markdown", "Markdown .md", "#c79a3a"],
    ["vanilla", "Vanilla", "#94908a"],
  ];
  const box = $("m1Cost");
  box.innerHTML = "";
  order.forEach(([k, name, color]) => {
    const m = methods[k];
    const c = el("div", "costcard");
    c.innerHTML =
      '<div class="cname"><span class="dot" style="background:' + color + '"></span>' + name + "</div>" +
      '<div class="cbig" style="color:' + color + '">' + m.kv_mb.toLocaleString() + ' <span style="font-size:13px;color:#9a9890">MB</span></div>' +
      '<div class="csub">' + m.prompt_tokens.toLocaleString() + " prompt tokens</div>";
    box.appendChild(c);
  });
}

const M1_FEEDS = ["m1Hay", "m1NapCtx", "m1MdCtx", "m1NapLog", "m1MdLog", "m1NapResp", "m1MdResp"];

function m1Reset() {
  if (M1.es) { M1.es.close(); M1.es = null; }
  M1.running = false;
  M1_FEEDS.forEach((id) => ($(id).innerHTML = ""));
  $("m1Hay").appendChild(el("div", "empty", "press Run — the full conversation streams in here"));
  $("m1NapCtx").appendChild(el("div", "mentry idle", "the prompt context will appear here"));
  $("m1MdCtx").appendChild(el("div", "mentry idle", "the .md note text will appear here"));
  $("m1NapLog").appendChild(el("div", "mentry idle", "nap / evict events appear here"));
  $("m1MdLog").appendChild(el("div", "mentry idle", "summarization events appear here"));
  $("m1NapResp").appendChild(el("div", "empty", "answers appear after the trajectory"));
  $("m1MdResp").appendChild(el("div", "empty", "answers appear after the trajectory"));
  $("m1Progress").textContent = "";
  $("m1NapStat").textContent = "0 adapters";
  $("m1MdStat").textContent = "0 notes";
  ["m1NapCtxTok", "m1MdCtxTok", "m1HayTok", "m1WindowLbl"].forEach((id) => ($(id).textContent = ""));
  $("m1NapDot").classList.remove("live");
  $("m1MdDot").classList.remove("live");
  $("m1Run").textContent = "Run trajectory ▸";
  $("m1Run").disabled = false;
  m1Fills(null, null, null);
  $("m1Cost").innerHTML = "";
}

function m1OnMeta(f) {
  M1.meta = Object.assign(M1.meta || {}, f);
  $("m1Needle").textContent = f.probes.map((p) => p.needle).join("  ·  ");
  $("m1Questions").textContent = f.probes.map((p) => p.q).join("   ");
  $("m1WindowLbl").textContent = "window 8,192 tok · " + f.total_turns + " turns · nap every K=" + f.nap_k;
  M1_FEEDS.forEach((id) => ($(id).innerHTML = ""));
}

function m1NeedleAt(step) {
  const pos = (M1.meta && M1.meta.needle_positions) || [];
  return pos.includes(step - 1);
}

function m1OnTurn(f) {
  const isNeedle = m1NeedleAt(f.step);
  $("m1Progress").textContent = "streaming turn " + f.step + " / " + M1.meta.total_turns + " …";

  // --- full haystack: append the exact turn text ---
  const hay = $("m1Hay");
  if (hay.querySelector(".empty")) hay.innerHTML = "";
  const h = el("div", "hturn" + (isNeedle ? " needle" : ""));
  h.innerHTML =
    '<span class="hn">' + f.step + "</span>" +
    '<span class="hrole">' + (isNeedle ? "🔑 " : "") + f.role + "</span>" +
    '<span class="htext"></span>';
  h.querySelector(".htext").textContent = f.text;
  hay.appendChild(h);
  hay.scrollTop = hay.scrollHeight;
  $("m1HayTok").textContent = f.raw_tokens.toLocaleString() + " tokens so far";

  // --- NapLoRA: context (compression summary) + activity log ---
  const nap = f.napora;
  $("m1NapCtx").innerHTML =
    '<div class="napcompress">' +
    "<b>" + nap.context_tokens + "</b> tokens of conversation in prompt<br>" +
    '<span class="dim">' + f.raw_tokens.toLocaleString() + " tokens of raw text →</span> " +
    nap.segments + " LoRA adapters <b style=\"font-size:13px\">≈ " + nap.adapter_mb + " MB</b> weights<br>" +
    '<span class="dim">rank ' + nap.adapter_rank + " · all history lives in Δweights, not tokens</span>" +
    "</div>";
  if (nap.napped) {
    const log = $("m1NapLog");
    if (log.querySelector(".idle")) log.innerHTML = "";
    log.appendChild(m1mk("mentry evict", "↯ EVICT → WEIGHTS", nap.event));
    log.scrollTop = log.scrollHeight;
  }
  $("m1NapStat").textContent = nap.segments + " adapters · rank " + nap.adapter_rank;
  $("m1NapCtxTok").textContent = nap.context_tokens + " tok in prompt";

  // --- Markdown: context (full .md notes) + activity log ---
  const md = f.markdown;
  if (md.notes) {
    const feed = $("m1MdCtx");
    feed.innerHTML = "";
    md.notes.forEach((n) => {
      const line = el("div", "ctxline");
      line.textContent = "• " + n;
      feed.appendChild(line);
    });
    feed.scrollTop = feed.scrollHeight;
    const log = $("m1MdLog");
    if (log.querySelector(".idle")) log.innerHTML = "";
    log.appendChild(m1mk("mentry note", "✎ SUMMARIZE → .md", md.event));
    log.scrollTop = log.scrollHeight;
  }
  $("m1MdStat").textContent = md.notes_lines + " notes";
  $("m1MdCtxTok").textContent = md.context_tokens.toLocaleString() + " tok in prompt";

  m1Fills(nap, md, f.vanilla);
}

function m1OnQuery(f) {
  $("m1Progress").textContent = "querying — needles now live only in weights / .md notes …";
  const napFeed = $("m1NapResp"), mdFeed = $("m1MdResp");
  if (napFeed.querySelector(".empty")) napFeed.innerHTML = "";
  if (mdFeed.querySelector(".empty")) mdFeed.innerHTML = "";
  const row = (feed, m) => {
    const e = el("div", "rentry " + (m.hit ? "hit" : "miss"));
    e.innerHTML = '<div class="rq">Q: ' + f.query + "</div>";
    e.appendChild(el("span", "rtag", m.hit ? "✓ RECALLED  " : "✗ LOST  "));
    e.appendChild(document.createTextNode(m.answer));
    e.appendChild(el("div", "csub", m.prompt_tokens + " prompt tok · " + m.kv_mb + " MB KV"));
    feed.appendChild(e);
    feed.scrollTop = feed.scrollHeight;
  };
  row(napFeed, f.methods.napora);
  row(mdFeed, f.methods.markdown);
  m1CostCards(f.methods);
}

function m1Run() {
  if (M1.running) return;
  m1Reset();
  M1.running = true;
  $("m1Run").textContent = "Running…";
  $("m1Run").disabled = true;
  $("m1NapDot").classList.add("live");
  $("m1MdDot").classList.add("live");
  $("m1NapResp").innerHTML = "";
  $("m1MdResp").innerHTML = "";
  const url = "/api/memory/run?scenario=" + encodeURIComponent(M1.scenario) + "&size=" + encodeURIComponent(M1.size);
  const es = new EventSource(url);
  M1.es = es;
  es.onmessage = (ev) => {
    const f = JSON.parse(ev.data);
    if (f.type === "meta") m1OnMeta(f);
    else if (f.type === "turn") m1OnTurn(f);
    else if (f.type === "query") m1OnQuery(f);
    else if (f.type === "done") m1Done(false);
  };
  es.onerror = () => { m1Done(true); };
}

function m1Done(err) {
  if (M1.es) { M1.es.close(); M1.es = null; }
  M1.running = false;
  $("m1Run").textContent = "Run trajectory ▸";
  $("m1Run").disabled = false;
  $("m1NapDot").classList.remove("live");
  $("m1MdDot").classList.remove("live");
  $("m1Progress").textContent = err ? "stream ended" : "done — needles recalled from weights with an ~8-token prompt";
}

async function m1Init() {
  try {
    const meta = await (await fetch("/api/memory/meta")).json();
    M1.scenarios = meta.scenarios;
    M1.sizes = meta.sizes;
    M1.meta = { turns_per_size: meta.turns_per_size };
    if (!M1.scenarios.includes(M1.scenario)) M1.scenario = M1.scenarios[0];
  } catch (e) {
    M1.scenarios = ["apollo_migration"];
    M1.sizes = ["small", "medium", "large"];
  }
  m1RenderControls();
  m1Reset();
  $("m1Run").onclick = m1Run;
  $("m1Reset").onclick = () => { if (!M1.running) m1Reset(); };
}

/* ============================ DEMO 2: PERSONALIZATION (live) ============================ */
const UID = "demo";
const P = { phase: 0, facts: 0, adapterReady: false, busy: false };
const SUGGEST_TEACH = [
  "I just moved to Seattle and I work as an ICU nurse.",
  "I'm vegetarian, always hunting for good meatless recipes.",
  "I have a golden retriever named Biscuit who joins me on hikes.",
  "Please keep your answers short and to the point.",
];
const SUGGEST_PROBE = [
  "Where do I live?",
  "What should I make for dinner tonight?",
  "What's my dog's name?",
  "Plan a fun weekend for me.",
];
const SYM = { added: "+", changed: "~", removed: "-", none: "·" };

function pStepper() {
  const labels = ["Converse", "Internalize", "New session"];
  const box = $("pStepper");
  box.innerHTML = "";
  labels.forEach((label, idx) => {
    const reached = P.phase >= idx;
    const s = el("div", "step" + (reached ? (idx === 2 && P.phase >= 2 ? " done" : " on") : ""));
    s.appendChild(el("span", "sn", String(idx + 1)));
    s.appendChild(el("span", "sl", label));
    if (idx < labels.length - 1) s.appendChild(el("span", "bar"));
    box.appendChild(s);
  });
}
function pAddMsg(feedId, role, text, typing) {
  const feed = $(feedId);
  const empty = feed.querySelector(".empty");
  if (empty) empty.remove();
  const m = el("div", "msg " + role + (typing ? " typing" : ""), text);
  feed.appendChild(m);
  feed.scrollTop = feed.scrollHeight;
  return m;
}
function pRenderChips(profile) {
  const box = $("pChips");
  box.innerHTML = "";
  const entries = Object.entries(profile || {});
  if (!entries.length) { box.appendChild(el("span", "", "")); }
  entries.forEach(([cat, val]) => {
    box.appendChild(el("span", "chip", `${cat.replace(/_/g, " ")}: ${val}`));
  });
  P.facts = entries.length;
  $("pFactCount").textContent = `${entries.length} fact${entries.length === 1 ? "" : "s"}`;
}
function pAddDiff(diff) {
  const feed = $("pDiffFeed");
  const empty = feed.querySelector(".empty");
  if (empty) empty.remove();
  if (!diff.length) {
    const d = el("div", "diff none");
    d.appendChild(el("span", "sym", SYM.none));
    d.appendChild(el("span", "", "no durable preference in that turn"));
    feed.appendChild(d);
  }
  diff.forEach((e) => {
    const d = el("div", "diff " + e.kind);
    d.appendChild(el("span", "sym", SYM[e.kind]));
    let txt;
    if (e.kind === "removed") txt = `${e.category}: ${e.old}`;
    else if (e.kind === "changed") txt = `${e.category}: ${e.old} → ${e.new}`;
    else txt = `${e.category}: ${e.new}`;
    d.appendChild(el("span", "", txt));
    feed.appendChild(d);
  });
  feed.scrollTop = feed.scrollHeight;
}
async function pSendTeach(text) {
  if (P.busy || !text.trim()) return;
  P.busy = true;
  $("pInput").value = "";
  pAddMsg("pChat", "user", text);
  const typing = pAddMsg("pChat", "bot", "…", true);
  try {
    const r = await api("/api/personalization/observe", { uid: UID, message: text });
    typing.classList.remove("typing");
    typing.textContent = r.reply;
    pAddDiff(r.diff);
    pRenderChips(r.profile);
    if (P.phase === 0 && P.facts > 0) { P.phase = 0; }
    $("pRepersonalize").disabled = P.facts === 0;
  } catch (err) {
    typing.classList.remove("typing");
    typing.textContent = "[error: " + err.message + "]";
  } finally {
    P.busy = false;
  }
}
async function pRepersonalize() {
  if (P.busy || P.facts === 0) return;
  P.busy = true;
  const status = $("pAdapterStatus");
  status.style.display = "";
  status.style.background = "#ebf2fe";
  status.style.borderColor = "#cdddfb";
  status.style.color = "#1e50c0";
  status.innerHTML = "";
  status.appendChild(el("span", "spinner"));
  status.appendChild(el("span", "", "internalizing profile → forging LoRA…"));
  $("pRepersonalize").disabled = true;
  try {
    const r = await api("/api/personalization/repersonalize", { uid: UID });
    $("pAdapterName").textContent = r.name;
    status.style.background = "#eef6f0";
    status.style.borderColor = "#b5ddc6";
    status.style.color = "#2f7d57";
    status.innerHTML = `<span style="font-size:15px">↯</span><span><strong>${r.name}</strong> · r=16 · ${r.num_facts} facts written into the weights</span>`;
    P.adapterReady = true;
    P.phase = 2;
    pStepper();
    const ph2 = $("pPhase2");
    ph2.style.opacity = "1";
    ph2.style.pointerEvents = "auto";
    pAddMsg("pChat2", "bot", "Fresh session — none of the previous conversation is in my context. Ask me something.");
  } catch (err) {
    status.style.background = "#fbeaea";
    status.style.borderColor = "#e8b8b2";
    status.style.color = "#b04a42";
    status.textContent = "[error: " + err.message + "]";
  } finally {
    P.busy = false;
    $("pRepersonalize").disabled = false;
  }
}
async function pSendProbe(text) {
  if (P.busy || !text.trim() || !P.adapterReady) return;
  P.busy = true;
  $("pInput2").value = "";
  const adapter = $("pAdapterToggle").checked;
  pAddMsg("pChat2", "user", text);
  const typing = pAddMsg("pChat2", "bot", "…", true);
  try {
    const r = await api("/api/personalization/chat", { uid: UID, message: text, adapter });
    typing.classList.remove("typing");
    typing.textContent = r.reply;
    const tag = el("div", "", `↳ adapter ${adapter ? "ON · " + $("pAdapterName").textContent : "OFF · base model"}`);
    tag.style.cssText = "font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#9a9890;margin-top:5px";
    typing.appendChild(tag);
  } catch (err) {
    typing.classList.remove("typing");
    typing.textContent = "[error: " + err.message + "]";
  } finally {
    P.busy = false;
  }
}
async function pResetAll() {
  if (P.busy) return;
  P.busy = true;
  try { await api("/api/personalization/reset", { uid: UID }); } catch (e) {}
  P.phase = 0; P.facts = 0; P.adapterReady = false;
  $("pChat").innerHTML = "";
  pAddMsg("pChat", "bot", "Hi! Tell me about yourself and I'll learn your preferences as we talk.");
  $("pDiffFeed").innerHTML = "";
  $("pDiffFeed").appendChild(el("div", "empty", "preference diffs appear here per turn"));
  pRenderChips({});
  $("pAdapterStatus").style.display = "none";
  $("pRepersonalize").disabled = true;
  $("pChat2").innerHTML = "";
  $("pChat2").appendChild(el("div", "empty", "forge an adapter first ↑"));
  $("pAdapterToggle").checked = true;
  $("pAdapterLabel").textContent = "ON";
  $("pAdapterLabel").style.color = "#2f6ae0";
  const ph2 = $("pPhase2");
  ph2.style.opacity = ".5"; ph2.style.pointerEvents = "none";
  pStepper();
  P.busy = false;
}
function pInitSuggest() {
  const s1 = $("pSuggest");
  SUGGEST_TEACH.forEach((t) => {
    const b = el("button", "chip-btn", t.length > 42 ? t.slice(0, 40) + "…" : t);
    b.title = t;
    b.onclick = () => pSendTeach(t);
    s1.appendChild(b);
  });
  const s2 = $("pSuggest2");
  SUGGEST_PROBE.forEach((t) => {
    const b = el("button", "chip-btn", t);
    b.onclick = () => pSendProbe(t);
    s2.appendChild(b);
  });
}

/* ============================ DEMO 3: SKILLS (scripted) ============================ */
const D3 = {
  item: 0, phase: 0, view: "after",
  data: [
    {
      bench: "HellaSwag",
      question: "A woman is outside with a bucket and a dog. The dog is trying to avoid a bath. She…",
      choices: ["…rinses the bucket with soap and blow-dries the dog's head.", "…uses a hose to keep the dog away from the bucket.", "…gets the dog wet, and it runs off again before she finishes.", "…climbs into the bathtub with the dog."],
      vanillaPick: 0, correct: 2,
      reflection: "I keep choosing the ending with the most familiar words. This is physical-continuation reasoning — I should simulate the literal next event that follows from the actions, not match surface vocabulary.",
      prompt: "You are solving a physical commonsense continuation. Ignore lexical overlap. For each ending, simulate the literal next physical event and choose the most plausible continuation.",
    },
    {
      bench: "PIQA",
      question: "To keep sliced apple from browning before lunch, you should…",
      choices: ["…rub the slices with lemon juice.", "…store them in a warm, sunny spot.", "…wrap them loosely in foil only.", "…leave them open to the air."],
      vanillaPick: 3, correct: 0,
      reflection: "I defaulted to the lowest-effort action. This is physical-goal reasoning about oxidation — I should pick the action whose mechanism actually prevents browning (acid slows enzymatic oxidation).",
      prompt: "You are solving a physical-goal task. Identify the mechanism the goal depends on, then select the action whose mechanism achieves it.",
    },
  ],
};
function d3RenderTabs() {
  const row = $("d3Tabs");
  row.innerHTML = "";
  D3.data.forEach((d, i) => {
    const b = el("button", "btn-sm" + (i === D3.item ? " on" : ""), d.bench);
    b.onclick = () => { D3.item = i; D3.phase = 0; D3.view = "after"; d3Render(); };
    row.appendChild(b);
  });
}
function d3Render() {
  const it = D3.data[D3.item];
  const phase = D3.phase, solved = phase >= 3;
  d3RenderTabs();
  $("d3Bench").textContent = it.bench;
  $("d3Question").textContent = it.question;
  // stepper
  const labels = ["Attempt", "Reflect", "Internalize", "Succeed"];
  const steps = $("d3Steps");
  steps.innerHTML = "";
  labels.forEach((label, idx) => {
    const reached = phase >= idx + 1;
    const isLast = idx === labels.length - 1;
    const s = el("div", "step" + (reached ? (isLast && solved ? " done" : " on") : ""));
    s.appendChild(el("span", "sn", String(idx + 1)));
    s.appendChild(el("span", "sl", label));
    if (!isLast) s.appendChild(el("span", "bar"));
    steps.appendChild(s);
  });
  // choices
  let highlight = -1, wrong = false, right = false;
  if (phase >= 1 && phase < 3) { highlight = it.vanillaPick; wrong = true; }
  else if (solved) {
    if (D3.view === "before") { highlight = it.vanillaPick; wrong = true; }
    else { highlight = it.correct; right = true; }
  }
  const choices = $("d3Choices");
  choices.innerHTML = "";
  it.choices.forEach((text, idx) => {
    let cls = "choice", mk = String.fromCharCode(65 + idx);
    if (idx === highlight && wrong) { cls += " wrong"; mk = "✗"; }
    else if (idx === highlight && right) { cls += " right"; mk = "✓"; }
    const c = el("div", cls);
    c.appendChild(el("span", "mk", mk));
    c.appendChild(el("span", "", text));
    c.querySelector("span:last-child").style.flex = "1";
    choices.appendChild(c);
  });
  // right panel
  const R = $("d3Right");
  R.innerHTML = "";
  if (phase === 0) {
    R.appendChild(el("div", "empty", "The agent will attempt the task, then improve itself. Press Attempt ▸"));
    R.firstChild.style.minHeight = "160px";
  } else {
    if (phase >= 2) {
      const rf = el("div", "", "");
      rf.style.marginBottom = "18px";
      rf.appendChild(el("div", "eyebrow", "Agent self-reflection")).style.cssText = "font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#2f6ae0;margin-bottom:10px";
      const p = el("p", "", it.reflection);
      p.style.cssText = "font-family:'IBM Plex Serif',Georgia,serif;font-style:italic;font-size:15px;line-height:1.62;color:#3a3832;margin:0";
      rf.appendChild(p);
      R.appendChild(rf);
      const pw = el("div", "");
      pw.style.marginBottom = "18px";
      const lbl = el("div", "eyebrow", "Self-written task prompt");
      lbl.style.marginBottom = "10px";
      pw.appendChild(lbl);
      const code = el("div", "", it.prompt);
      code.style.cssText = "font-family:'IBM Plex Mono',monospace;font-size:13px;line-height:1.6;color:#26241f;background:#fff;border:1px solid #eceae3;border-radius:8px;padding:14px 15px";
      pw.appendChild(code);
      R.appendChild(pw);
    }
    if (phase >= 3) {
      const ad = el("div", "");
      ad.style.cssText = "display:flex;align-items:center;gap:11px;font-family:'IBM Plex Mono',monospace;font-size:13px;background:#ebf2fe;border:1px solid #cdddfb;border-radius:8px;padding:13px 15px;color:#1e50c0;animation:fadeUp .4s ease both";
      ad.innerHTML = `<span style="font-size:16px">↯</span><span>Text-to-LoRA → <strong>${it.bench}-skill.lora</strong> · r=8 · composed into weights</span>`;
      R.appendChild(ad);
    }
    const res = el("div", "");
    res.style.cssText = `margin-top:18px;display:flex;align-items:center;gap:10px;font-size:15px;font-weight:600;color:${phase >= 3 ? "#2f7d57" : "#b04a42"}`;
    res.innerHTML = `<span style="font-size:18px">${phase >= 3 ? "✓" : "✗"}</span>${phase >= 3 ? "Correct — skill internalized as a LoRA adapter." : "Incorrect — anchored on surface features."}`;
    R.appendChild(res);
  }
  // next button
  const nb = $("d3Next");
  nb.textContent = phase === 0 ? "Attempt ▸" : phase === 1 ? "Self-reflect ▸" : phase === 2 ? "Internalize (T2L) ▸" : "Solved ✓";
  nb.disabled = solved;
  // compare
  const cmp = $("d3Compare");
  cmp.style.display = solved ? "flex" : "none";
  $("d3Before").className = "seg" + (D3.view === "before" ? " seg-on" : "");
  $("d3After").className = "seg" + (D3.view === "after" ? " seg-on" : "");
  $("d3Before").style.color = D3.view === "before" ? "#b04a42" : "#8a877f";
  $("d3After").style.color = D3.view === "after" ? "#2f7d57" : "#8a877f";
  $("d3CompareNote").textContent = D3.view === "before" ? "Vanilla agent picked the wrong continuation." : "With the T2L adapter, the agent answers correctly.";
}

/* ============================ INIT ============================ */
async function init() {
  renderTabs();
  [0, 1, 2].forEach((j) => ($(`demo-${j}`).style.display = j === activeDemo ? "" : "none"));

  // demo 1 (live memory)
  await m1Init();

  // demo 2
  pInitSuggest();
  await pResetAll();
  $("pResetAll").onclick = pResetAll;
  $("pSend").onclick = () => pSendTeach($("pInput").value);
  $("pInput").addEventListener("keydown", (e) => { if (e.key === "Enter") pSendTeach($("pInput").value); });
  $("pRepersonalize").onclick = pRepersonalize;
  $("pSend2").onclick = () => pSendProbe($("pInput2").value);
  $("pInput2").addEventListener("keydown", (e) => { if (e.key === "Enter") pSendProbe($("pInput2").value); });
  $("pReset2").onclick = () => { $("pChat2").innerHTML = ""; pAddMsg("pChat2", "bot", "Cleared. Ask me something."); };
  $("pAdapterToggle").addEventListener("change", (e) => {
    const on = e.target.checked;
    $("pAdapterLabel").textContent = on ? "ON" : "OFF";
    $("pAdapterLabel").style.color = on ? "#2f6ae0" : "#9a9890";
  });

  // demo 3
  $("d3Next").onclick = () => { D3.phase = Math.min(D3.phase + 1, 3); d3Render(); };
  $("d3Reset").onclick = () => { D3.phase = 0; D3.view = "after"; d3Render(); };
  $("d3Before").onclick = () => { D3.view = "before"; d3Render(); };
  $("d3After").onclick = () => { D3.view = "after"; d3Render(); };
  d3Render();

  // health badge
  try {
    const h = await (await fetch("/api/health")).json();
    $("liveBadge").textContent = "live · D2L";
    $("modeBadge").textContent = "backend: Doc-to-LoRA" + (h.model_loaded ? " (loaded)" : " (lazy)");
  } catch (e) {
    $("modeBadge").textContent = "backend: unreachable";
  }
}
init();
