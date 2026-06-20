"""SkillStore: acquire a skill by internalizing its doc into the weights (D2L).

Mirrors the personalization track, but for skills/tools: a skill is a document
(tool docs, a policy, worked examples). Internalizing it puts the knowledge in
the adapter; a tiny doc-free behavioral NUDGE in the prompt activates its use
(D2L gives knowledge, not the behavior to apply it — the nudge supplies that).
"""

from __future__ import annotations

from ..core.model import D2LModel

# doc-free: tells the model to use what it internalized, WITHOUT restating the doc.
NUDGE = (
    "You have already studied and memorized the relevant reference material. "
    "Use what you learned to respond. Do not ask for information you already know."
)


class SkillStore:
    def __init__(self, model: D2LModel):
        self.model = model
        self._docs: dict[str, str] = {}      # skill name -> doc text
        self._adapters: dict[str, object] = {}  # skill name -> cached adapter

    def acquire(self, name: str, doc: str) -> None:
        """Internalize a skill document into a cached LoRA adapter."""
        self._docs[name] = doc
        self.model.reset()
        self.model.internalize(doc)
        self._adapters[name] = self.model.snapshot()

    def use(self, name: str | None, task: str, nudge: bool = True, **kw) -> str:
        """Answer a task. name=None -> base model (skill not acquired)."""
        prompt = f"{NUDGE}\n\n{task}" if (nudge and name) else task
        if name is None:
            self.model.reset()
        else:
            self.model.restore(self._adapters[name])
        return self.model.chat(prompt, **kw)
