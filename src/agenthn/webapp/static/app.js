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
let activeDemo = 1; // open on the live personalization demo

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

/* ============================ DEMO 1: MEMORY (scripted) ============================ */
const D1 = {
  scenario: 0,
  running: false,
  timer: null,
  data: [
    {
      label: "Support · 8 turns",
      needle: "Order A-2294 — the customer wants an exchange (not a refund) for a wrong-size jacket.",
      turns: [
        "My jacket (order A-2294) came in the wrong size — I'd like an exchange, not a refund.",
        "Also, do you ship to Canada?",
        "What's your return window?",
        "Can I change the email on my account?",
        "Do you have this jacket in green?",
        "What's the status of my other order, B-1180?",
        "Is there a student discount?",
        "Remind me — what did I originally want done with order A-2294?",
      ],
      agentAnswer: "You asked for an exchange on the wrong-size jacket (A-2294) — not a refund. I can start that now.",
      vanillaAnswer: "I'm sorry — the earlier details of this conversation are no longer in my context.",
    },
    {
      label: "Research · 8 turns",
      needle: "Hard constraint set first: training must fit in under 8GB of VRAM.",
      turns: [
        "Let's design the experiment. Hard rule: training has to fit in under 8GB of VRAM.",
        "What optimizer should we use?",
        "How many epochs is reasonable here?",
        "Should we use mixed precision?",
        "What learning-rate schedule do you suggest?",
        "Recommend a data-augmentation strategy.",
        "Which eval metrics should we report?",
        "Before we finalize — what hardware budget did we lock in at the start?",
      ],
      agentAnswer: "Under 8GB of VRAM for training — the hard constraint we set first. Keep batch size and precision within it.",
      vanillaAnswer: "The start of our discussion has dropped out of my context, so I can't recall that constraint.",
    },
    {
      label: "Coding · 8 turns",
      needle: "Team decision: every database column uses snake_case, never camelCase.",
      turns: [
        "Project rule before we start: every DB column must be snake_case, never camelCase.",
        "Scaffold the users table.",
        "Add an auth middleware.",
        "Write the migration for orders.",
        "Set up the test database.",
        "Add an index on the email column.",
        "Wire up the CI pipeline.",
        "Quick check — what naming convention did we agree on for DB columns?",
      ],
      agentAnswer: "snake_case for every database column — the rule we set at the start, so the new tables follow it.",
      vanillaAnswer: "That was early in our session and is no longer in my context window, so I can't say.",
    },
  ],
};
const K = 4;
const ENTRY_TAG = { user: "USER", note: "· ", d2l: "↯ D2L", sys: "⚠ OVERFLOW", pass: "✓ RECALLED", fail: "✗ LOST" };

function d1Entry(kind, text) {
  const e = el("div", "entry " + kind);
  if (ENTRY_TAG[kind] && kind !== "note") e.appendChild(el("span", "tag", ENTRY_TAG[kind]));
  e.appendChild(document.createTextNode(text));
  return e;
}
function d1RenderScenarioTabs() {
  const row = $("d1ScenarioTabs");
  row.innerHTML = "";
  D1.data.forEach((d, i) => {
    const b = el("button", "btn-sm" + (i === D1.scenario ? " on" : ""), d.label);
    b.onclick = () => { d1Set(i); };
    row.appendChild(b);
  });
}
function d1Reset() {
  if (D1.timer) clearTimeout(D1.timer);
  D1.timer = null;
  D1.running = false;
  $("d1Agent").innerHTML = "";
  $("d1Vanilla").innerHTML = "";
  $("d1Agent").appendChild(el("div", "empty", "press Run to start the trajectory"));
  $("d1Vanilla").appendChild(el("div", "empty", "press Run to start the trajectory"));
  $("d1Adapters").textContent = "0 adapters";
  d1Meters(600, 600, false);
  $("d1DotA").classList.remove("live");
  $("d1DotB").classList.remove("live");
  $("d1Run").textContent = "Run trajectory ▸";
  $("d1Run").disabled = false;
}
function d1Set(i) {
  D1.scenario = i;
  $("d1Needle").textContent = D1.data[i].needle;
  d1RenderScenarioTabs();
  d1Reset();
}
function d1Meters(aTok, vTok, overflow) {
  $("d1AgentTok").textContent = `${aTok.toLocaleString()} tok · ${Math.round(aTok * 0.12)} MB KV`;
  $("d1VanillaTok").textContent = `${vTok.toLocaleString()} tok · ${Math.round(vTok * 0.12)} MB KV`;
  $("d1AgentBar").style.width = Math.min(100, Math.round((aTok / 8000) * 100)) + "%";
  const vb = $("d1VanillaBar");
  vb.style.width = Math.min(100, Math.round((vTok / 8000) * 100)) + "%";
  vb.style.background = overflow ? "#c2554d" : "#94908a";
  $("d1VanillaTok").style.color = overflow ? "#c2554d" : "#94908a";
}
function d1Run() {
  if (D1.timer) clearTimeout(D1.timer);
  const sc = D1.data[D1.scenario];
  D1.running = true;
  $("d1Run").textContent = "Running…";
  $("d1Run").disabled = true;
  $("d1Agent").innerHTML = "";
  $("d1Vanilla").innerHTML = "";
  $("d1DotA").classList.add("live");
  $("d1DotB").classList.add("live");
  let aTok = 600, vTok = 600, adapters = 0, i = 0;
  d1Meters(aTok, vTok, false);
  const agentFeed = $("d1Agent"), vanFeed = $("d1Vanilla");
  const push = (feed, kind, text) => { feed.appendChild(d1Entry(kind, text)); feed.scrollTop = feed.scrollHeight; };
  const step = () => {
    const isLast = i === sc.turns.length - 1;
    const userTxt = `${i + 1}. ${sc.turns[i]}`;
    push(agentFeed, "user", userTxt);
    push(vanFeed, "user", userTxt);
    aTok += 1150; vTok += 1150;
    let overflow = false;
    if (isLast) {
      push(agentFeed, "d2l", "Doc-to-LoRA: compacted remaining turns into the memory adapter before answering.");
      adapters += 1; aTok = 950;
      push(agentFeed, "pass", sc.agentAnswer);
      vTok = 8000; overflow = true;
      push(vanFeed, "fail", sc.vanillaAnswer);
    } else {
      if ((i + 1) % K === 0) {
        push(agentFeed, "d2l", "Doc-to-LoRA: compacted turns into adapter r=8 · rank-concat with memory adapter · evicted from context");
        adapters += 1; aTok = 950;
      } else {
        push(agentFeed, "note", "internalized");
      }
      if (vTok > 8000) { push(vanFeed, "sys", "Context window full (8k) — evicting oldest turn (the original request)."); vTok = 8000; overflow = true; }
      else push(vanFeed, "note", "appended to context");
    }
    $("d1Adapters").textContent = adapters + " adapters";
    d1Meters(aTok, vTok, overflow);
    i += 1;
    if (i < sc.turns.length) D1.timer = setTimeout(step, 800);
    else {
      D1.timer = null; D1.running = false;
      $("d1Run").textContent = "Run trajectory ▸"; $("d1Run").disabled = false;
      $("d1DotA").classList.remove("live"); $("d1DotB").classList.remove("live");
    }
  };
  D1.timer = setTimeout(step, 350);
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

  // demo 1
  d1Set(0);
  $("d1Run").onclick = d1Run;
  $("d1Reset").onclick = d1Reset;

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
    const badge = $("liveBadge");
    if (h.mock) {
      badge.textContent = "live · mock";
      badge.classList.add("mock");
      $("modeBadge").textContent = "backend: mock (no GPU)";
    } else {
      badge.textContent = "live · D2L";
      $("modeBadge").textContent = "backend: Doc-to-LoRA" + (h.model_loaded ? " (loaded)" : " (lazy)");
    }
  } catch (e) {
    $("modeBadge").textContent = "backend: unreachable";
  }
}
init();
