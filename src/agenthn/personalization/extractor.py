"""Preference extractor: a user message -> structured profile updates.

Runs on the LOCAL gemma-2-2b base model (no API key, fully self-contained).
A 2B model is weak at JSON, so we use a forgiving line format
`ACTION | CATEGORY | VALUE` and parse leniently.

Updates have the form {category, value, action: add|update|remove} so that
"I went vegan" can remove "loves steak" instead of contradicting it. These feed
both the profile doc and the UI diff view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..core.model import D2LModel


@dataclass
class ProfileUpdate:
    category: str
    value: str
    action: Literal["add", "update", "remove"]


_PROMPT = """You extract durable facts and preferences about the user from a message.
Output one line per fact, in the format: ACTION | CATEGORY | VALUE
- ACTION is "add" for a new or changed fact, or "remove" to retract a previous fact.
- CATEGORY is a short snake_case key, e.g. dietary, location, profession, pet, communication_style.
- VALUE is a short phrase.
Only extract durable facts or preferences (ignore small talk and questions).
If there is nothing to extract, output exactly: NONE

Examples:
Message: "I just moved to Seattle and I work as a nurse."
add | location | Seattle
add | profession | nurse

Message: "I'm vegetarian."
add | dietary | vegetarian

Message: "Actually I'm not vegetarian anymore, I eat meat now."
remove | dietary | vegetarian
add | dietary | eats meat

Message: "What's the capital of France?"
NONE

Now do this one. Output only the lines, nothing else.
Message: "{msg}\""""


def extract_updates(model: "D2LModel", user_message: str) -> list[ProfileUpdate]:
    """Extract profile updates from one user message using the base model."""
    model.reset()  # extraction runs on the base model, no adapter
    raw = model.chat(
        _PROMPT.format(msg=user_message.replace('"', "'")), max_new_tokens=128
    )
    return _parse(raw)


def _parse(raw: str) -> list[ProfileUpdate]:
    updates: list[ProfileUpdate] = []
    for line in raw.splitlines():
        line = line.strip().strip("`").strip("-").strip()
        if not line or line.upper() == "NONE":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        action, category, value = parts
        action = action.lower()
        category = category.lower().replace(" ", "_")
        if action not in ("add", "update", "remove") or not category or not value:
            continue
        updates.append(ProfileUpdate(category=category, value=value, action=action))
    return updates
