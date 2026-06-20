"""FastAPI app for the AgentHN demo page.

Serves the static single-page UI and the personalization API wired to the
Doc-to-LoRA hook (or the mock fallback — see ``service.build_service``).

Run (on the GPU box, real model):
    /home/ubuntu/doc-to-lora/.venv/bin/python -m uvicorn agenthn.webapp.app:app \\
        --host 0.0.0.0 --port 8000

Run (anywhere, mock — no GPU needed):
    AGENTHN_MOCK=1 python -m uvicorn agenthn.webapp.app:app --port 8000
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service import build_service

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AgentHN")
service = build_service()


class ObserveBody(BaseModel):
    uid: str = "demo"
    message: str


class RepersonalizeBody(BaseModel):
    uid: str = "demo"


class ChatBody(BaseModel):
    uid: str = "demo"
    message: str
    adapter: bool = True


class ResetBody(BaseModel):
    uid: str = "demo"


@app.get("/api/health")
def health() -> dict:
    return service.health()


@app.post("/api/personalization/observe")
def observe(body: ObserveBody) -> dict:
    return asdict(service.observe(body.uid, body.message))


@app.post("/api/personalization/repersonalize")
def repersonalize(body: RepersonalizeBody) -> dict:
    return asdict(service.repersonalize(body.uid))


@app.post("/api/personalization/chat")
def chat(body: ChatBody) -> dict:
    return {"reply": service.chat(body.uid, body.message, body.adapter)}


@app.post("/api/personalization/reset")
def reset(body: ResetBody) -> dict:
    service.reset(body.uid)
    return {"ok": True}


@app.get("/api/personalization/profile")
def profile(uid: str = "demo") -> dict:
    return {"profile": service.profile(uid)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
