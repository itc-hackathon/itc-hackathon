"""PersonalizationStore: per-user profile docs + cached LoRA adapters.

Lifecycle per user:
  observe(uid, message)  -> extract structured updates from a turn, apply them,
                            return the diff (for the UI / demo)
  repersonalize(uid)     -> internalize the profile doc into ONE adapter, cache it
  chat(uid, message)     -> swap that user's adapter in, then generate

The profile doc is the human-readable "personalization memory" (one .md per user,
written on every update) and NEVER enters the prompt — it only feeds internalize().
We keep ONE value per category, so updates are simple set/remove (see
personalization-track-design: no stacking for this track).
"""

from __future__ import annotations

from ..core.config import PROFILES_DIR
from ..core.model import D2LModel
from .extractor import ProfileUpdate, extract_updates

# A diff entry: (kind, category, old_value, new_value); kind in {added, changed, removed}.
DiffEntry = tuple[str, str, str | None, str | None]


class PersonalizationStore:
    def __init__(self, model: D2LModel):
        self.model = model
        self._profiles: dict[str, dict[str, str]] = {}   # uid -> {category: value}
        self._adapters: dict[str, object] = {}           # uid -> cached generated adapter

    # --- profile doc maintenance -----------------------------------------
    def observe(self, uid: str, user_message: str) -> list[DiffEntry]:
        """Extract updates from one user turn and apply them; return the diff."""
        return self.update_profile(uid, extract_updates(self.model, user_message))

    def update_profile(self, uid: str, updates: list[ProfileUpdate]) -> list[DiffEntry]:
        """Apply structured updates to the user's profile; return the diff."""
        prof = self._profiles.setdefault(uid, {})
        diff: list[DiffEntry] = []
        for u in updates:
            if u.action == "remove":
                if u.category in prof:
                    diff.append(("removed", u.category, prof.pop(u.category), None))
                continue
            old = prof.get(u.category)
            if old is None:
                diff.append(("added", u.category, None, u.value))
            elif old != u.value:
                diff.append(("changed", u.category, old, u.value))
            else:
                continue  # no-op, skip
            prof[u.category] = u.value
        if diff:
            self._write_doc(uid)
        return diff

    def profile_doc(self, uid: str) -> str:
        """Render the user's profile as the document fed to internalize()."""
        prof = self._profiles.get(uid, {})
        if not prof:
            return ""
        lines = ["The following facts describe the user, for personalizing responses."]
        lines += [f"- {cat.replace('_', ' ')}: {val}" for cat, val in prof.items()]
        return "\n".join(lines)

    def _write_doc(self, uid: str) -> None:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        (PROFILES_DIR / f"{uid}.md").write_text(self.profile_doc(uid))

    def forget(self, uid: str) -> None:
        """Drop a user's profile and cached adapter (restart the demo)."""
        self._profiles.pop(uid, None)
        self._adapters.pop(uid, None)
        doc = PROFILES_DIR / f"{uid}.md"
        if doc.exists():
            doc.unlink()

    # --- adapter cache / swap --------------------------------------------
    def repersonalize(self, uid: str) -> None:
        """Internalize the user's current doc into an adapter and cache it."""
        doc = self.profile_doc(uid)
        if not doc:
            return
        self.model.reset()
        self.model.internalize(doc)
        self._adapters[uid] = self.model.snapshot()

    def chat(self, uid: str, message: str, **kw) -> str:
        """Swap in the user's cached adapter (if any), then generate."""
        adapter = self._adapters.get(uid)
        if adapter is None:
            self.model.reset()           # no profile yet -> base behavior
        else:
            self.model.restore(adapter)
        return self.model.chat(message, **kw)
