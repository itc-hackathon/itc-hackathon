# itc-hackathon — AgentHN

**AgentHN** (Agent + Hypernetwork): agents that write to their own **weights** at
inference time, built on [Doc-to-LoRA](https://github.com/SakanaAI/doc-to-lora).

Tracks live in their own folders under `src/agenthn/` so the team can work in
parallel:

| Folder | Owner | What |
|---|---|---|
| `core/` | shared | D2L model wrapper + config (keep stable) |
| `personalization/` | Eric | running profile doc → per-user adapter |
| `memory/` | Bryan, Nikash | long-horizon memory via stacked adapters |
| `skills/` | David | self-improving skills (T2L) |
| `webapp/` | Eric | demo UI |

## Setup

This package reuses the working `doc-to-lora` venv (which already has the full
GPU stack + the `ctx_to_lora` editable install). It is installed editable into
that venv — no separate environment needed.

```bash
export PATH="$HOME/.local/bin:$PATH"
uv pip install -e /home/ubuntu/itc-hackathon --python /home/ubuntu/doc-to-lora/.venv
```

Requires HF login for the gated `google/gemma-2-2b-it` base model
(`uv run --no-sync huggingface-cli login` in the doc-to-lora repo).

## Smoke test

```bash
/home/ubuntu/doc-to-lora/.venv/bin/python scripts/smoke_test.py
```

## Web app (demo UI)

A FastAPI app serves the single-page demo and wires the **personalization** track
live to the Doc-to-LoRA hook (memory & skills tabs are scripted previews).

```bash
# real model, on the GPU box (loads the checkpoint on first request)
/home/ubuntu/doc-to-lora/.venv/bin/python -m uvicorn agenthn.webapp.app:app \
    --host 0.0.0.0 --port 8000
```

Then open `http://<host>:8000`. The personalization demo flow is: chat with the
agent (preferences are extracted into a live diff panel per turn) → **Repersonalize**
to internalize the profile into a per-user LoRA → a **new empty-context session**
with an **adapter ON/OFF** toggle showing the personalization lives in the weights,
not the prompt.

No GPU? Run the mock backend anywhere (same UI, canned responses):

```bash
AGENTHN_MOCK=1 python -m uvicorn agenthn.webapp.app:app --port 8000
```

The mock also auto-activates if torch / `ctx_to_lora` can't be imported. `/api/health`
reports which backend is live.

## Layout

```
src/agenthn/
  core/
    config.py               # paths (D2L repo, checkpoint), device
    model.py                # D2LModel: load / internalize / snapshot / restore / chat
  personalization/
    extractor.py            # turns -> {category, value, action} updates
    profile_store.py        # per-user profile docs + cached adapters (swap)
  webapp/
    app.py                  # FastAPI: serves the SPA + personalization API
    service.py              # live (D2L) + mock services behind the API
    static/                 # index.html, app.js, styles.css (the demo page)
  memory/ skills/           # teammates' tracks
scripts/smoke_test.py       # load checkpoint, internalize, generate
```
