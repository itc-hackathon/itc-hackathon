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

/* ============================ DEMO 3: SKILLS (converse / document / internalize / converse again) ============================ */
const RC_SKILL_META = {
  physics: { color: "#2f6ae0", label: "Physics" },
  formatting: { color: "#9a5cc7", label: "Formatting" },
};
const RC = { phase: 0, busy: false, lastMessage: "", lastBaseReply: "", lastSkill: null };

function rcStepper() {
  const labels = ["Converse", "Document", "Internalize", "Converse again"];
  const box = $("rcStepper");
  box.innerHTML = "";
  const last = labels.length - 1;
  labels.forEach((label, idx) => {
    const reached = RC.phase >= idx;
    const s = el("div", "step" + (reached ? (idx === last && RC.phase >= last ? " done" : " on") : ""));
    s.appendChild(el("span", "sn", String(idx + 1)));
    s.appendChild(el("span", "sl", label));
    if (idx < last) s.appendChild(el("span", "bar"));
    box.appendChild(s);
  });
}

function rcAddMsg(feedId, role, text, typing) {
  const feed = $(feedId);
  const empty = feed.querySelector(".empty");
  if (empty) empty.remove();
  const m = el("div", "msg " + role + (typing ? " typing" : ""), text);
  feed.appendChild(m);
  feed.scrollTop = feed.scrollHeight;
  return m;
}

function rcTag(text, color) {
  const tag = el("div", "", text);
  tag.style.cssText = "font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:" + (color || "#9a9890") + ";margin-top:5px";
  return tag;
}

async function rcSend(text) {
  if (RC.busy || !text.trim()) return;
  RC.busy = true;
  $("rcInput").value = "";
  rcAddMsg("rcChat", "user", text);
  const typing = rcAddMsg("rcChat", "bot", "…", true);
  try {
    const r = await api("/api/skills/converse", { message: text });
    typing.classList.remove("typing");
    typing.textContent = r.reply;
    typing.appendChild(rcTag(`↳ base model · ${r.elapsed}s · ${r.prompt_tokens} prompt tok`));
    RC.lastMessage = text;
    RC.lastBaseReply = r.reply;
    $("rcClassifyBtn").disabled = false;
  } catch (err) {
    typing.classList.remove("typing");
    typing.textContent = "[error: " + err.message + "]";
  } finally {
    RC.busy = false;
  }
}

async function rcClassify() {
  if (RC.busy || !RC.lastMessage) return;
  RC.busy = true;
  $("rcClassifyBtn").disabled = true;
  const status = $("rcClassifyStatus");
  status.style.display = "";
  status.textContent = "classifying…";
  try {
    const r = await api("/api/skills/classify", { message: RC.lastMessage });
    const meta = RC_SKILL_META[r.skill] || { color: "#9a9890" };
    status.innerHTML =
      '<span class="dot" style="background:' + meta.color + ';width:7px;height:7px;display:inline-block;margin-right:6px"></span>routed → ' +
      r.label + " adapter · classified in " + r.classify_ms + "ms";
    const docBox = $("rcDoc");
    docBox.textContent = r.doc;
    docBox.style.display = "";
    RC.lastSkill = r.skill;
    RC.phase = Math.max(RC.phase, 1);
    rcStepper();
    $("rcInternalizeBtn").disabled = false;
  } catch (err) {
    status.textContent = "[error: " + err.message + "]";
  } finally {
    RC.busy = false;
    $("rcClassifyBtn").disabled = false;
  }
}

async function rcInternalize() {
  if (RC.busy || !RC.lastSkill) return;
  RC.busy = true;
  $("rcInternalizeBtn").disabled = true;
  const status = $("rcAdapterStatus");
  status.style.display = "";
  status.style.background = "#ebf2fe";
  status.style.borderColor = "#cdddfb";
  status.style.color = "#1e50c0";
  status.innerHTML = "";
  status.appendChild(el("span", "spinner"));
  status.appendChild(el("span", "", `internalizing ${RC.lastSkill} doc → forging LoRA…`));
  try {
    const r = await api("/api/skills/internalize", { skill: RC.lastSkill });
    status.style.background = "#eef6f0";
    status.style.borderColor = "#b5ddc6";
    status.style.color = "#2f7d57";
    status.innerHTML = `<span style="font-size:15px">↯</span><span><strong>${r.skill}.lora</strong> · ${r.cached ? "already cached" : r.elapsed + "s to forge"}</span>`;
    RC.phase = 2;
    rcStepper();
    const ph2 = $("rcPhase2");
    ph2.style.opacity = "1";
    ph2.style.pointerEvents = "auto";
    $("rcChat2").innerHTML = "";
    $("rcChat2").appendChild(el("div", "empty", "ask again to see the adapter answer ↑"));
  } catch (err) {
    status.style.background = "#fbeaea";
    status.style.borderColor = "#e8b8b2";
    status.style.color = "#b04a42";
    status.textContent = "[error: " + err.message + "]";
  } finally {
    RC.busy = false;
    $("rcInternalizeBtn").disabled = false;
  }
}

async function rcAskAgain() {
  if (RC.busy || RC.phase < 2) return;
  RC.busy = true;
  $("rcChat2").innerHTML = "";
  rcAddMsg("rcChat2", "user", RC.lastMessage);
  const before = rcAddMsg("rcChat2", "bot", RC.lastBaseReply);
  before.appendChild(rcTag("↳ before · base model"));
  const typing = rcAddMsg("rcChat2", "bot", "…", true);
  try {
    const r = await api("/api/skills/converse-again", { message: RC.lastMessage, skill: RC.lastSkill });
    typing.classList.remove("typing");
    typing.textContent = r.reply;
    typing.appendChild(rcTag(`↳ after · ${RC.lastSkill} adapter restored · ${r.elapsed}s · ${r.prompt_tokens} prompt tok`, "#2f6ae0"));
    RC.phase = 3;
    rcStepper();
  } catch (err) {
    typing.classList.remove("typing");
    typing.textContent = "[error: " + err.message + "]";
  } finally {
    RC.busy = false;
  }
}

function rcReset() {
  RC.phase = 0;
  RC.busy = false;
  RC.lastMessage = "";
  RC.lastBaseReply = "";
  RC.lastSkill = null;
  $("rcChat").innerHTML = "";
  rcAddMsg("rcChat", "bot", "Hi — ask me a physics word problem, or ask for structured output (JSON / YAML / protobuf / bullets). I'll answer cold first; then we'll classify, internalize, and ask again.");
  $("rcClassifyBtn").disabled = true;
  $("rcClassifyStatus").style.display = "none";
  $("rcDoc").style.display = "none";
  $("rcDoc").textContent = "";
  $("rcInternalizeBtn").disabled = true;
  $("rcAdapterStatus").style.display = "none";
  const ph2 = $("rcPhase2");
  ph2.style.opacity = ".5";
  ph2.style.pointerEvents = "none";
  $("rcChat2").innerHTML = "";
  $("rcChat2").appendChild(el("div", "empty", "internalize an adapter first ↑"));
  rcStepper();
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

  // demo 3 (converse / document / internalize / converse again)
  rcReset();
  $("rcSend").onclick = () => rcSend($("rcInput").value);
  $("rcInput").addEventListener("keydown", (e) => { if (e.key === "Enter") rcSend($("rcInput").value); });
  $("rcReset").onclick = () => { if (!RC.busy) rcReset(); };
  $("rcClassifyBtn").onclick = rcClassify;
  $("rcInternalizeBtn").onclick = rcInternalize;
  $("rcAskAgainBtn").onclick = rcAskAgain;

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
