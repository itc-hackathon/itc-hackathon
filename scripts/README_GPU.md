# GPU demo server — one-script lifecycle

The static site (Vercel) talks to a live FastAPI backend on a GPU box over a
tunnel, and **auto-falls back to recorded fixtures** when the backend is gone
(health check fails, or the 9pm PT hard cutoff passes). One script does the
whole lifecycle: capture → serve → switch → kill → shutdown.

## What you need on the server (Prime Intellect, 1×A100)

1. **The doc-to-lora venv with the GPU stack** (torch + `ctx_to_lora` + the
   hypernetwork checkpoint), exactly as the rest of the project expects. Default
   path `/home/ubuntu/doc-to-lora/.venv`; override with `PY=/path/to/python`.
   - Install this repo into that venv: `uv pip install -e . --python <venv>`
   - HF auth if the checkpoint needs it: `huggingface-cli login`
2. **git push access** from the box (SSH key or token) so it can auto-commit the
   fixtures and the config switch to `main`.
3. **Outbound HTTPS** (for the cloudflared tunnel; the script auto-downloads the
   `cloudflared` binary if it's missing).
4. `curl` + `bash` (standard).

## Run it

```bash
git clone <repo> && cd itc-hackathon
bash scripts/run_demo_server.sh
```

That's it. The script:

1. Starts the backend and waits for `/api/health`.
2. Runs `scripts/capture_fixtures.py` → writes `static/fixtures/*.json`, commits + pushes.
3. Opens a cloudflared tunnel, writes the URL into `static/config.js`, commits + pushes
   (Vercel auto-redeploys → site goes live on the GPU).
4. Serves until **8:55pm PT** — concurrent website requests queue on the shared
   model lock (`runtime.py`), so the single A100 is never double-driven.
5. At **8:55pm PT**: reverts `config.js` (removes the GPU URL) → commits + pushes →
   site flips to replaying the recorded runs.
6. At **9:00pm PT**: kills the tunnel + server.
7. Powers off the instance.

## Knobs (env vars)

| var | default | meaning |
|-----|---------|---------|
| `PY` | `/home/ubuntu/doc-to-lora/.venv/bin/python` | interpreter w/ GPU stack |
| `PORT` | `8000` | backend port |
| `SWITCH_AT` | `2026-06-20T20:55:00-07:00` | when site → fixtures |
| `KILL_AT` | `2026-06-20T21:00:00-07:00` | when GPU server dies |
| `CAPTURE_SIZES` | `small medium` | memory sizes to record (add `large` if time) |
| `DO_SHUTDOWN` | `1` | poweroff instance at the end (`0` to keep it) |

## Just capture fixtures (no serving / scheduling)

```bash
# server already running on :8000
CAPTURE_BASE=http://127.0.0.1:8000 python scripts/capture_fixtures.py
```

Fixtures are bundled into the Vercel build (`static/` → `public/`), so once
pushed the site replays them with no backend at all.
