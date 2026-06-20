"""Service behind the step-by-step "self-improving skills" demo.

Breaks the question into four explicit, user-triggered stages that mirror the
personalization demo's stepper:
  1. converse()       -- bare question -> base model. No doc, no adapter.
  2. classify()        -- classify the question (physics vs formatting) and
                           surface the matching reference doc.
  3. internalize()      -- internalize that doc into a LoRA adapter, cached
                           per skill and built once.
  4. converse_again()   -- the SAME question again, now with the cached
                           adapter restored and nothing pasted into the prompt.

Runs on the shared, lock-protected D2L model (see runtime.py) -- only one
demo (memory / personalization / skills) can generate at a time.
"""

from __future__ import annotations

import time

from ..skills import formatting_bench, physics_bench
from .runtime import MODEL_LOCK, get_model
from .skills_common import generate_with_prefill

SKILLS = {
    "physics": {
        "doc": physics_bench.DOC,
        "label": "Physics",
        # Matches the chain-of-thought style the doc's worked examples use --
        # without it the model free-styles a verbose, header/bullet-heavy
        # preamble and clips before the arithmetic under the token budget.
        "prefill": "Let's think step by step. ",
        # Multi-step problems (e.g. acceleration -> force) need more room
        # than the single-step ones even with the prefill.
        "max_new_tokens": 320,
    },
    "formatting": {
        "doc": formatting_bench.DOC,
        "label": "Formatting",
        # No chain-of-thought before raw JSON/YAML/proto/bullets.
        "prefill": "",
        "max_new_tokens": 200,
    },
}

CLASSIFY_PROMPT = (
    "Classify the question below into exactly one category: PHYSICS (a "
    "mechanics word problem involving force, mass, velocity, friction, "
    "gravity, momentum, or motion) or FORMATTING (a request to produce "
    "structured output like JSON, YAML, protobuf, or a bulleted list). "
    "Reply with one word: physics or formatting.\n\n"
    "Question: {question}\n"
    "Category:"
)
_FORMATTING_KEYWORDS = ("json", "yaml", "yml", "proto", "bullet", "list", "format")

# Cached adapter snapshots, keyed by skill -- built once, restored per chat.
_ADAPTERS: dict[str, object] = {}


def _classify(model, message: str) -> tuple[str, float]:
    raw, elapsed = generate_with_prefill(
        model,
        CLASSIFY_PROMPT.format(question=message.strip()),
        prefill="",
        max_new_tokens=6,
    )
    label = raw.strip().lower()
    if "format" in label:
        skill = "formatting"
    elif "phys" in label:
        skill = "physics"
    else:
        # Parse miss on the classifier's free-form output -- fall back to a
        # deterministic keyword check so the demo never dead-ends.
        skill = "formatting" if any(k in message.lower() for k in _FORMATTING_KEYWORDS) else "physics"
    return skill, elapsed


class SkillsRouterService:
    def __init__(self) -> None:
        self._lock = MODEL_LOCK
        # Step 1 classifies silently before generating (see converse()) so it
        # can apply the right skill's prefill/token budget; cached here so
        # step 2's explicit "Classify" click reuses that result instead of
        # re-running the tiny generation.
        self._last_classify: dict | None = None

    def converse(self, message: str) -> dict:
        """Step 1: bare question straight to the base model.

        Classifies first (silently -- the UI still presents this as a cold,
        unrouted answer) purely to pick the right prefill/token budget for
        the skill. Without it, physics questions lose the "Let's think step
        by step." cue the doc's worked examples use, ramble through a
        verbose unprefixed preamble, and clip before the arithmetic.
        """
        with self._lock:
            model = get_model()
            model.reset()
            skill, classify_elapsed = _classify(model, message)
            self._last_classify = {"message": message, "skill": skill, "elapsed": classify_elapsed}
            spec = SKILLS[skill]
            model.reset()
            answer, elapsed = generate_with_prefill(
                model, message, prefill=spec["prefill"], max_new_tokens=spec["max_new_tokens"]
            )
            return {
                "reply": answer,
                "elapsed": round(elapsed, 2),
                "prompt_tokens": model.count_tokens(message),
            }

    def classify(self, message: str) -> dict:
        """Step 2: classify the question and surface its reference doc."""
        with self._lock:
            model = get_model()
            cached = self._last_classify
            if cached and cached["message"] == message:
                skill, elapsed = cached["skill"], cached["elapsed"]
            else:
                model.reset()
                skill, elapsed = _classify(model, message)
            spec = SKILLS[skill]
            return {
                "skill": skill,
                "label": spec["label"],
                "classify_ms": round(elapsed * 1000),
                "doc": spec["doc"],
                "cached": skill in _ADAPTERS,
            }

    def internalize(self, skill: str) -> dict:
        """Step 3: internalize the skill's doc into a cached LoRA adapter."""
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r}; have {list(SKILLS)}")
        with self._lock:
            if skill in _ADAPTERS:
                return {"skill": skill, "elapsed": 0.0, "cached": True}
            model = get_model()
            model.reset()
            start = time.perf_counter()
            _ADAPTERS[skill] = model.internalize_segment(SKILLS[skill]["doc"])
            elapsed = time.perf_counter() - start
            model.reset()
            return {"skill": skill, "elapsed": round(elapsed, 2), "cached": False}

    def converse_again(self, message: str, skill: str) -> dict:
        """Step 4: the same question again, this time with the adapter restored."""
        if skill not in SKILLS:
            raise ValueError(f"unknown skill {skill!r}; have {list(SKILLS)}")
        spec = SKILLS[skill]
        with self._lock:
            model = get_model()
            if skill not in _ADAPTERS:
                model.reset()
                _ADAPTERS[skill] = model.internalize_segment(spec["doc"])
            model.restore(_ADAPTERS[skill])
            answer, elapsed = generate_with_prefill(
                model, message, prefill=spec["prefill"], max_new_tokens=spec["max_new_tokens"]
            )
            model.reset()
            return {
                "reply": answer,
                "elapsed": round(elapsed, 2),
                "prompt_tokens": model.count_tokens(message),
            }


def build_skills_router_service() -> SkillsRouterService:
    return SkillsRouterService()


__all__ = ["SkillsRouterService", "build_skills_router_service"]
