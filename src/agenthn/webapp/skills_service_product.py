"""Service behind the live self-improving skills demo (Product Q&A track).

This is the SELF-REFINE loop, streamed turn-by-turn over SSE so the UI can show
the agent improving its own weights over rounds:

  Round 0  -- ATTEMPT cold. No skill in the weights, so on a *fictional* product
              the base model scores ~0. This is the "no info initially" baseline.
  Round 1  -- STUDY the reference doc -> write study notes (the "skill") ->
              INTERNALIZE those notes into a LoRA adapter -> RETRY the held-out
              questions with the adapter active (nothing in the prompt).
  Round 2+ -- REFLECT on the questions it still got wrong, REWRITE the notes to
              cover them, RE-INTERNALIZE, RETRY.

There is deliberately NO keep-best ratchet: each round internalizes the latest
notes and we report the real accuracy, even if a round dips. The point is an
honest trajectory of an agent editing its own weights, not a faked hill-climb.

Runs on the shared, lock-protected D2L model (see runtime.py) -- only one demo
(memory / personalization / skills) can generate at a time.
"""

from __future__ import annotations

import time
from typing import Iterator

import torch

from ..skills.lumen_bench import QUESTIONS, SOURCE, is_correct
from .runtime import MODEL_LOCK, get_model

N_QUESTIONS = 16       # held-out battery (each ~6% -> a readable accuracy curve)
MAX_STUDY_ROUNDS = 3   # study rounds after the cold baseline (round 0)
ANSWER_TOKENS = 40


def _qprompt(q: str) -> str:
    return f"Answer the customer's question in one short sentence.\n\nQuestion: {q}"


def _study_prompt(source: str) -> str:
    return (
        "Read the following product reference material and write concise study "
        "notes (about 5 sentences) capturing the most important facts a support "
        f"agent would need.\n\nMaterial:\n{source}\n\nNotes:"
    )


def _reflect_prompt(notes: str, wrong: list[dict], source: str) -> str:
    # Honest feedback: the agent is told ONLY which questions it got wrong (no
    # answer key) and re-handed the doc. It has to locate the facts itself.
    missed = "\n".join(f"- {r['q']}" for r in wrong)
    return (
        "You are studying product reference material to answer customer "
        "questions.\n\n"
        f"Your current notes:\n{notes}\n\n"
        f"You could NOT correctly answer these questions:\n{missed}\n\n"
        f"The full reference material:\n{source}\n\n"
        "Find the facts these questions ask about in the reference material and "
        "rewrite your study notes so you can answer them next time. Keep the notes "
        "concise but include the specific facts needed. Output only the revised "
        "notes."
    )


class SkillsProductService:
    def __init__(self) -> None:
        self._lock = MODEL_LOCK

    def meta(self) -> dict:
        qs = QUESTIONS[:N_QUESTIONS]
        return {
            "questions": [{"q": q, "gold": accepts[0]} for q, accepts in qs],
            "max_rounds": MAX_STUDY_ROUNDS,
        }

    def run(self) -> Iterator[dict]:
        """Yield SSE frames for the whole self-refine run (held under the lock)."""
        with self._lock:
            torch.manual_seed(0)
            model = get_model()
            qs = QUESTIONS[:N_QUESTIONS]
            source_tokens = model.count_tokens(SOURCE)

            yield {
                "type": "meta",
                "n": len(qs),
                "source": SOURCE,
                "source_tokens": source_tokens,
                "max_rounds": MAX_STUDY_ROUNDS,
                "questions": [{"q": q, "gold": accepts[0]} for q, accepts in qs],
            }

            trajectory = []

            def test(restore_adapter) -> Iterator[dict]:
                """Answer the battery; yield a frame per question, return wrong list."""
                correct = 0
                wrong: list[dict] = []
                for i, (q, accepts) in enumerate(qs):
                    if restore_adapter is None:
                        model.reset()
                    else:
                        model.restore(restore_adapter)
                    ans = model.chat(_qprompt(q), max_new_tokens=ANSWER_TOKENS).strip()
                    ok = is_correct(ans, accepts)
                    correct += ok
                    rec = {"q": q, "gold": accepts[0], "answer": ans, "ok": ok}
                    if not ok:
                        wrong.append(rec)
                    yield {
                        "type": "answer", "round": _round[0], "index": i,
                        "q": q, "gold": accepts[0], "answer": ans, "correct": ok,
                    }
                _result["correct"] = correct
                _result["wrong"] = wrong

            # mutable round counter + scratch result shared with the test() generator
            _round = [0]
            _result: dict = {}

            # ---- Round 0: cold attempt, no skill -----------------------------
            yield {"type": "round_start", "round": 0, "phase": "attempt",
                   "label": "Attempt cold — no skill in weights"}
            model.reset()
            yield from test(None)
            acc = _result["correct"] / len(qs)
            trajectory.append({"round": 0, "correct": _result["correct"],
                               "total": len(qs), "accuracy": acc})
            yield {"type": "round_done", "round": 0, "correct": _result["correct"],
                   "total": len(qs), "accuracy": acc, "trajectory": list(trajectory)}

            notes = ""
            prev_lines: set[str] = set()

            # ---- Study rounds ------------------------------------------------
            for r in range(1, MAX_STUDY_ROUNDS + 1):
                _round[0] = r
                first = r == 1
                yield {"type": "round_start", "round": r,
                       "phase": "study" if first else "reflect",
                       "label": ("Study the doc — write the skill" if first
                                 else "Reflect on misses — rewrite the skill")}

                model.reset()
                if first:
                    notes = model.chat(_study_prompt(SOURCE), max_new_tokens=256).strip()
                else:
                    notes = model.chat(
                        _reflect_prompt(notes, _result["wrong"], SOURCE),
                        max_new_tokens=400,
                    ).strip()
                lines = [ln for ln in notes.split("\n") if ln.strip()]
                added = [ln for ln in lines if ln not in prev_lines]
                prev_lines = set(lines)
                yield {"type": "notes", "round": r, "notes": notes,
                       "added_lines": added, "reflect": not first}

                # Internalize the self-written notes into a LoRA adapter.
                yield {"type": "internalizing", "round": r}
                start = time.perf_counter()
                adapter = model.internalize_segment(notes)
                elapsed = time.perf_counter() - start
                yield {"type": "internalized", "round": r,
                       "elapsed": round(elapsed, 2),
                       "tokens": model.count_tokens(notes)}

                # Retry the held-out battery with the adapter active.
                yield from test(adapter)
                acc = _result["correct"] / len(qs)
                trajectory.append({"round": r, "correct": _result["correct"],
                                   "total": len(qs), "accuracy": acc})
                yield {"type": "round_done", "round": r,
                       "correct": _result["correct"], "total": len(qs),
                       "accuracy": acc, "trajectory": list(trajectory)}

            model.reset()
            best = max(trajectory, key=lambda t: t["accuracy"])
            yield {"type": "done", "trajectory": trajectory,
                   "base_accuracy": trajectory[0]["accuracy"],
                   "best_round": best["round"], "best_accuracy": best["accuracy"],
                   "final_accuracy": trajectory[-1]["accuracy"]}


def build_skills_product_service() -> SkillsProductService:
    return SkillsProductService()


__all__ = ["SkillsProductService", "build_skills_product_service"]
