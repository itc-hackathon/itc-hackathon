"""FastAPI app for the AgentHN demo page.

Serves the static single-page UI and the personalization API wired to the
Doc-to-LoRA hook (see ``service.py``).

Run (on the GPU box):
    /home/ubuntu/doc-to-lora/.venv/bin/python -m uvicorn agenthn.webapp.app:app \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .memory_service import build_memory_service
from .service import build_service
from .skills_service_formatting import build_skills_formatting_service
from .skills_service_product import build_skills_product_service
from .skills_service_router import build_skills_router_service

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="AgentHN")

# The static UI is hosted on Vercel (a different origin) and talks to this
# backend over a tunnel. Allow any origin so the cross-origin fetches + SSE
# (EventSource) from the deployed page reach the live model. Hackathon scope —
# tighten allow_origins if this ever serves real users.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = build_service()
memory_service = build_memory_service()
SKILLS_SERVICES = {
    "product": build_skills_product_service(),
    "formatting": build_skills_formatting_service(),
}
skills_router_service = build_skills_router_service()

# Eager-load the shared D2L model at import time (main thread). Loading it lazily
# from a request/SSE worker thread triggers a "copy out of meta tensor" error.
from .runtime import get_model  # noqa: E402

get_model()


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


class SkillsConverseBody(BaseModel):
    message: str


class SkillsClassifyBody(BaseModel):
    message: str


class SkillsInternalizeBody(BaseModel):
    skill: str


class SkillsConverseAgainBody(BaseModel):
    message: str
    skill: str


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


@app.get("/api/memory/meta")
def memory_meta() -> dict:
    return memory_service.meta()


@app.get("/api/memory/run")
def memory_run(scenario: str = "apollo_migration", size: str = "medium") -> StreamingResponse:
    """Server-Sent Events stream of the live memory run (one frame per turn/query)."""

    def gen():
        for frame in memory_service.run(scenario, size):
            yield f"data: {json.dumps(frame)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/skills/converse")
def skills_converse(body: SkillsConverseBody) -> dict:
    return skills_router_service.converse(body.message)


@app.post("/api/skills/classify")
def skills_classify(body: SkillsClassifyBody) -> dict:
    return skills_router_service.classify(body.message)


@app.post("/api/skills/internalize")
def skills_internalize(body: SkillsInternalizeBody) -> dict:
    try:
        return skills_router_service.internalize(body.skill)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/skills/converse-again")
def skills_converse_again(body: SkillsConverseAgainBody) -> dict:
    try:
        return skills_router_service.converse_again(body.message, body.skill)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _skills_service(name: str):
    try:
        return SKILLS_SERVICES[name]
    except KeyError:
        raise HTTPException(404, f"unknown skill {name!r}; have {list(SKILLS_SERVICES)}")


@app.get("/api/skills/{name}/meta")
def skills_meta(name: str) -> dict:
    return _skills_service(name).meta()


@app.get("/api/skills/{name}/run")
def skills_run(name: str) -> StreamingResponse:
    """Server-Sent Events stream of the live skill-acquisition run for `name`."""
    svc = _skills_service(name)

    def gen():
        for frame in svc.run():
            yield f"data: {json.dumps(frame)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
