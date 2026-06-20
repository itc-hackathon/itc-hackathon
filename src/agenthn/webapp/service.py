"""Service layer behind the web app's personalization demo.

Wraps the personalization track (extractor + PersonalizationStore + D2LModel) in
a small, JSON-friendly API the FastAPI routes call. Runs on the GPU box where
doc-to-lora's venv + checkpoint are available.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class Turn:
    """One observed conversation turn and what it changed in the profile."""

    reply: str
    diff: list[dict]          # [{kind, category, old, new}]
    profile: dict[str, str]   # full {category: value} after applying the turn


@dataclass
class Adapter:
    """Result of internalizing a user's profile doc into a LoRA adapter."""

    ready: bool
    name: str
    num_facts: int
    profile_doc: str


def _diff_to_dicts(diff) -> list[dict]:
    return [
        {"kind": kind, "category": cat, "old": old, "new": new}
        for (kind, cat, old, new) in diff
    ]


def _adapter_name(uid: str) -> str:
    return f"{uid}.lora"


class LiveService:
    """Drives the actual PersonalizationStore on the loaded D2L model.

    The model is heavy and not thread-safe, so every call holds a lock — fine for
    a single-user demo. Model load is lazy so the server starts instantly and the
    (slow) checkpoint load happens on the first request.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store = None  # built on first use

    def _ensure_store(self):
        if self._store is None:
            from ..core.model import D2LModel
            from ..personalization.profile_store import PersonalizationStore

            self._store = PersonalizationStore(D2LModel.load())
        return self._store

    def observe(self, uid: str, message: str) -> Turn:
        with self._lock:
            store = self._ensure_store()
            diff = store.observe(uid, message)
            # Reply on the base model: the conversation phase is about *capturing*
            # preferences; the adapter is only built later, on repersonalize().
            reply = store.model.chat(message, max_new_tokens=200)
            return Turn(
                reply=reply,
                diff=_diff_to_dicts(diff),
                profile=dict(store._profiles.get(uid, {})),
            )

    def repersonalize(self, uid: str) -> Adapter:
        with self._lock:
            store = self._ensure_store()
            store.repersonalize(uid)
            prof = store._profiles.get(uid, {})
            return Adapter(
                ready=bool(store._adapters.get(uid) is not None),
                name=_adapter_name(uid),
                num_facts=len(prof),
                profile_doc=store.profile_doc(uid),
            )

    def chat(self, uid: str, message: str, adapter: bool) -> str:
        with self._lock:
            store = self._ensure_store()
            if adapter:
                return store.chat(uid, message, max_new_tokens=200)
            store.model.reset()
            return store.model.chat(message, max_new_tokens=200)

    def profile(self, uid: str) -> dict[str, str]:
        with self._lock:
            store = self._ensure_store()
            return dict(store._profiles.get(uid, {}))

    def reset(self, uid: str) -> None:
        with self._lock:
            store = self._ensure_store()
            store.forget(uid)

    def health(self) -> dict:
        return {"model_loaded": self._store is not None}


def build_service() -> LiveService:
    return LiveService()


__all__ = ["Turn", "Adapter", "LiveService", "build_service"]
