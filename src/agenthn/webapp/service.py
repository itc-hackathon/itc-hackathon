"""Service layer behind the web app's personalization demo.

Wraps the personalization track (extractor + PersonalizationStore + D2LModel) in
a small, JSON-friendly API the FastAPI routes call. Two implementations:

- ``LiveService``  — loads the real D2L model and drives the actual hook. Used on
  the GPU box where doc-to-lora's venv + checkpoint are available.
- ``MockService``  — no model, no GPU. A tiny rule-based stand-in so the page can
  be developed, demoed, and screenshotted anywhere. It mirrors the live API
  exactly so the frontend is identical in both modes.

``build_service()`` picks Live when the GPU stack imports and ``AGENTHN_MOCK`` is
not set, else Mock. The frontend reads ``/api/health`` to show which is active.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Protocol


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


class PersonalizationService(Protocol):
    mock: bool

    def observe(self, uid: str, message: str) -> Turn: ...
    def repersonalize(self, uid: str) -> Adapter: ...
    def chat(self, uid: str, message: str, adapter: bool) -> str: ...
    def profile(self, uid: str) -> dict[str, str]: ...
    def reset(self, uid: str) -> None: ...
    def health(self) -> dict: ...


# --------------------------------------------------------------------------- #
# Live service: the real Doc-to-LoRA hook.
# --------------------------------------------------------------------------- #
class LiveService:
    """Drives the actual PersonalizationStore on the loaded D2L model.

    The model is heavy and not thread-safe, so every call holds a lock — fine for
    a single-user demo. Model load is lazy so the server starts instantly and the
    (slow) checkpoint load happens on the first request.
    """

    mock = False

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
        return {"mock": False, "model_loaded": self._store is not None}


# --------------------------------------------------------------------------- #
# Mock service: rule-based stand-in, no model.
# --------------------------------------------------------------------------- #
# (keyword -> (category, value)) used by the mock extractor. Mirrors the kind of
# durable facts the real LLM extractor pulls, so the diff feed looks realistic.
_MOCK_RULES: list[tuple[tuple[str, ...], str, str]] = [
    (("vegetarian",), "dietary", "vegetarian"),
    (("vegan",), "dietary", "vegan"),
    (("pescatarian",), "dietary", "pescatarian"),
    (("gluten-free", "gluten free"), "dietary", "gluten-free"),
    (("seattle",), "location", "Seattle"),
    (("new york", "nyc"), "location", "New York"),
    (("san francisco", "sf"), "location", "San Francisco"),
    (("nurse",), "profession", "nurse"),
    (("engineer", "developer", "programmer"), "profession", "software engineer"),
    (("teacher",), "profession", "teacher"),
    (("golden retriever",), "pet", "golden retriever"),
    (("dog",), "pet", "dog"),
    (("cat",), "pet", "cat"),
    (("short", "concise", "brief", "to the point"), "communication_style", "concise replies"),
    (("detailed", "thorough"), "communication_style", "detailed answers"),
    (("morning", "6am", "early"), "schedule", "morning person"),
    (("night owl", "late", "post-10pm", "after 10pm"), "schedule", "night owl"),
    (("budget", "cheap", "affordable"), "budget", "budget-conscious"),
    (("kids", "children"), "family", "has kids"),
    (("run", "running", "hike", "hiking"), "hobby", "outdoor exercise"),
    (("coffee",), "drink", "coffee enthusiast"),
]

# Canned, profile-aware answers the mock returns when the adapter is ON.
_MOCK_TEMPLATES = {
    "dietary": {
        "vegetarian": "Try a roasted-veg grain bowl — fully vegetarian.",
        "vegan": "A vegan Buddha bowl would hit the spot — no animal products.",
        "pescatarian": "How about seared salmon tacos — pescatarian-friendly.",
        "gluten-free": "A gluten-free rice-noodle stir fry works well for you.",
    },
}


def _mock_extract(message: str) -> list[tuple[str, str]]:
    msg = message.lower()
    found: list[tuple[str, str]] = []
    for keywords, cat, val in _MOCK_RULES:
        if any(k in msg for k in keywords):
            found.append((cat, val))
    # keep one value per category (last wins), preserving first-seen order
    seen: dict[str, str] = {}
    for cat, val in found:
        seen[cat] = val
    return list(seen.items())


class MockService:
    """Deterministic, model-free stand-in mirroring the live API."""

    mock = True

    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, str]] = {}
        self._adapters: dict[str, bool] = {}

    def observe(self, uid: str, message: str) -> Turn:
        prof = self._profiles.setdefault(uid, {})
        diff: list[dict] = []
        for cat, val in _mock_extract(message):
            old = prof.get(cat)
            if old is None:
                diff.append({"kind": "added", "category": cat, "old": None, "new": val})
            elif old != val:
                diff.append({"kind": "changed", "category": cat, "old": old, "new": val})
            else:
                continue
            prof[cat] = val
        return Turn(reply=self._ack(diff), diff=diff, profile=dict(prof))

    def _ack(self, diff: list[dict]) -> str:
        if not diff:
            return "Got it."
        cats = ", ".join(d["category"].replace("_", " ") for d in diff)
        return f"Noted — I'll remember that ({cats})."

    def repersonalize(self, uid: str) -> Adapter:
        prof = self._profiles.get(uid, {})
        self._adapters[uid] = bool(prof)
        return Adapter(
            ready=bool(prof),
            name=_adapter_name(uid),
            num_facts=len(prof),
            profile_doc=self._profile_doc(uid),
        )

    def chat(self, uid: str, message: str, adapter: bool) -> str:
        prof = self._profiles.get(uid, {})
        if not adapter or not self._adapters.get(uid) or not prof:
            return self._base_answer(message)
        return self._personalized_answer(message, prof)

    def profile(self, uid: str) -> dict[str, str]:
        return dict(self._profiles.get(uid, {}))

    def reset(self, uid: str) -> None:
        self._profiles.pop(uid, None)
        self._adapters.pop(uid, None)

    def health(self) -> dict:
        return {"mock": True, "model_loaded": True}

    # --- canned generation -------------------------------------------------
    def _profile_doc(self, uid: str) -> str:
        prof = self._profiles.get(uid, {})
        if not prof:
            return ""
        lines = ["The following facts describe the user, for personalizing responses."]
        lines += [f"- {c.replace('_', ' ')}: {v}" for c, v in prof.items()]
        return "\n".join(lines)

    def _base_answer(self, message: str) -> str:
        m = message.lower()
        if "dinner" in m or "eat" in m:
            return "There are a lot of options — any cuisine you prefer?"
        if "live" in m or "where" in m:
            return "I don't have that information about you."
        if "dog" in m or "pet" in m or "name" in m:
            return "I'm not sure — you haven't told me."
        if "weekend" in m or "saturday" in m:
            return "Plenty of options! What are you in the mood for?"
        return "I don't have any personal context to go on here."

    def _personalized_answer(self, message: str, prof: dict[str, str]) -> str:
        m = message.lower()
        if "dinner" in m or "eat" in m:
            diet = prof.get("dietary")
            tmpl = _MOCK_TEMPLATES["dietary"].get(diet) if diet else None
            return tmpl or "A nearby spot that matches your tastes — quick and easy."
        if "live" in m or "where" in m:
            loc = prof.get("location")
            return f"You live in {loc}." if loc else "You haven't mentioned where you live."
        if "dog" in m or "pet" in m or "name" in m:
            pet = prof.get("pet")
            return f"Your {pet}, of course!" if pet else "You haven't mentioned a pet."
        if "weekend" in m or "saturday" in m:
            bits = []
            if prof.get("hobby"):
                bits.append("a morning workout")
            if prof.get("dietary"):
                bits.append(f"a {prof['dietary']} brunch")
            plan = ", then ".join(bits) if bits else "something you'd enjoy"
            return f"How about {plan}?"
        facts = ", ".join(f"{c.replace('_', ' ')}: {v}" for c, v in prof.items())
        return f"Based on what I know about you ({facts}), here's a tailored take."


def build_service() -> PersonalizationService:
    """Pick the live service when the GPU stack is importable, else the mock."""
    if os.environ.get("AGENTHN_MOCK"):
        return MockService()
    try:
        import torch  # noqa: F401
        import ctx_to_lora  # noqa: F401
    except Exception:
        return MockService()
    return LiveService()


__all__ = [
    "Turn",
    "Adapter",
    "PersonalizationService",
    "LiveService",
    "MockService",
    "build_service",
]
