#!/usr/bin/env python3
"""Capture REAL demo outputs from the live FastAPI backend into static fixtures.

Hits every endpoint the web UI uses — with every suggestion-chip input — and
records the responses (including full SSE frame streams, with per-frame timing)
into ``src/agenthn/webapp/static/fixtures/``. The static site replays these when
the GPU backend is unreachable (see app.js / config.js).

Run it against an already-running server (the orchestrator does this for you):

    CAPTURE_BASE=http://127.0.0.1:8000 \\
    /home/ubuntu/doc-to-lora/.venv/bin/python scripts/capture_fixtures.py

Env:
  CAPTURE_BASE     base URL of the running server (default http://127.0.0.1:8000)
  CAPTURE_SIZES    space-separated memory sizes to capture (default "small medium")
  CAPTURE_FORCE    "1" to re-capture combos whose fixture already exists
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("CAPTURE_BASE", "http://127.0.0.1:8000").rstrip("/")
SIZES = os.environ.get("CAPTURE_SIZES", "small medium").split()
FORCE = os.environ.get("CAPTURE_FORCE", "") == "1"

OUT = Path(__file__).resolve().parents[1] / "src" / "agenthn" / "webapp" / "static" / "fixtures"

# Must mirror the suggestion chips in static/app.js exactly — the fixture is
# keyed by the message text, so a mismatch means no recorded reply.
SUGGEST_TEACH = [
    "I just moved to Seattle and I work as an ICU nurse.",
    "I'm vegetarian, always hunting for good meatless recipes.",
    "I have a golden retriever named Biscuit who joins me on hikes.",
    "Please keep your answers short and to the point.",
]
SUGGEST_PROBE = [
    "Where do I live?",
    "What should I make for dinner tonight?",
    "What's my dog's name?",
    "Plan a fun weekend for me.",
]


def _req(path: str, body=None, timeout=120):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")


def get_json(path: str):
    with urllib.request.urlopen(_req(path), timeout=120) as r:
        return json.loads(r.read().decode())


def post_json(path: str, body: dict):
    with urllib.request.urlopen(_req(path, body), timeout=600) as r:
        return json.loads(r.read().decode())


def stream_sse(path: str, timeout=1800):
    """Consume an SSE stream into [{"t": ms_offset, "f": frame}, ...]."""
    frames = []
    with urllib.request.urlopen(_req(path), timeout=timeout) as r:
        t0 = time.time()
        for raw in r:  # iterates the response line by line
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("data:"):
                payload = line[5:].strip()
                if not payload:
                    continue
                frames.append({"t": int((time.time() - t0) * 1000), "f": json.loads(payload)})
    return frames


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj))
    print(f"  wrote fixtures/{name}  ({(OUT / name).stat().st_size:,} bytes)")


# ----------------------------------------------------------------------------


def capture_memory() -> None:
    print("[memory] meta")
    meta = get_json("/api/memory/meta")
    write("memory_meta.json", meta)
    scenarios = meta.get("scenarios", [])
    for scenario in scenarios:
        for size in SIZES:
            name = f"memory_{scenario}_{size}.json"
            if (OUT / name).exists() and not FORCE:
                print(f"[memory] skip {scenario}/{size} (already captured)")
                continue
            print(f"[memory] streaming {scenario}/{size} …")
            t = time.time()
            frames = stream_sse(f"/api/memory/run?scenario={scenario}&size={size}")
            print(f"  {len(frames)} frames in {time.time() - t:.0f}s")
            write(name, frames)


def capture_personalization() -> None:
    print("[personalization] reset + teach")
    post_json("/api/personalization/reset", {"uid": "demo"})
    observe = {}
    for msg in SUGGEST_TEACH:
        r = post_json("/api/personalization/observe", {"uid": "demo", "message": msg})
        observe[msg] = {"reply": r.get("reply"), "diff": r.get("diff"), "profile": r.get("profile")}
        print(f"  observed: {msg[:40]!r}")

    print("[personalization] repersonalize")
    repersonalize = post_json("/api/personalization/repersonalize", {"uid": "demo"})

    print("[personalization] probe (adapter on/off)")
    chat = {}
    for msg in SUGGEST_PROBE:
        chat[msg] = {}
        for adapter in (True, False):
            r = post_json("/api/personalization/chat", {"uid": "demo", "message": msg, "adapter": adapter})
            chat[msg]["true" if adapter else "false"] = r.get("reply")
        print(f"  probed: {msg[:40]!r}")

    write("personalization.json", {"observe": observe, "repersonalize": repersonalize, "chat": chat})


def capture_skills() -> None:
    print("[skills] streaming product self-refine run …")
    t = time.time()
    frames = stream_sse("/api/skills/product/run")
    print(f"  {len(frames)} frames in {time.time() - t:.0f}s")
    write("skills_product.json", frames)


def main() -> None:
    print(f"capturing fixtures from {BASE} → {OUT}")
    print(f"  sizes={SIZES} force={FORCE}")
    # Health gate so we fail fast with a clear message.
    try:
        get_json("/api/health")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"backend not reachable at {BASE}: {e}")

    errors = []
    for label, fn in (("memory", capture_memory),
                      ("personalization", capture_personalization),
                      ("skills", capture_skills)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one broken demo shouldn't sink the rest
            print(f"!! {label} capture failed: {e}")
            errors.append(label)

    marker = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base": BASE,
        "sizes": SIZES,
        "errors": errors,
    }
    write(".captured.json", marker)
    if errors:
        print(f"done with errors in: {', '.join(errors)}")
    else:
        print("done — all fixtures captured")


if __name__ == "__main__":
    main()
