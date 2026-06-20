"""Personalization demo (CLI).

Flow:
  1. user has a conversation -> preferences extracted -> written via diffs to the
     personalization memory (data/profiles/<uid>.md)
  2. internalize the memory into ONE LoRA adapter
  3. open a NEW context (no conversation in the prompt) -> ask questions; the
     personalized answers still hold, while the base model (adapter off) does not.

Run with the shared venv:
  /home/ubuntu/doc-to-lora/.venv/bin/python scripts/personalization_demo.py
"""

from agenthn.core.model import D2LModel
from agenthn.personalization.profile_store import PersonalizationStore

UID = "alex"

CONVERSATION = [
    "Hey! I just moved to Seattle for a new job as an ICU nurse.",
    "I'm vegetarian, so I'm always hunting for good meatless recipes.",
    "I have a golden retriever named Biscuit who keeps me company on hikes.",
    "Also, please keep your answers short and to the point.",
]

PROBES = [
    "Where do I live?",
    "What should I make for dinner tonight?",
    "What's my dog's name?",
    "Suggest a fun weekend activity for me.",
]

_SYM = {"added": "+", "changed": "~", "removed": "-"}


def fmt_diff(diff):
    if not diff:
        return "    (no preferences extracted)"
    out = []
    for kind, cat, old, new in diff:
        if kind == "removed":
            out.append(f"    - {cat}: {old}")
        elif kind == "changed":
            out.append(f"    ~ {cat}: {old} -> {new}")
        else:
            out.append(f"    + {cat}: {new}")
    return "\n".join(out)


def main():
    print("Loading model...")
    store = PersonalizationStore(D2LModel.load())

    print("\n=== 1. CONVERSATION (extracting preferences -> personalization memory) ===")
    for turn in CONVERSATION:
        print(f"\n  user: {turn}")
        diff = store.observe(UID, turn)
        print(fmt_diff(diff))

    print(f"\n=== 2. PERSONALIZATION MEMORY (data/profiles/{UID}.md) ===")
    print("    " + store.profile_doc(UID).replace("\n", "\n    "))

    print("\n=== 3. INTERNALIZE memory -> LoRA adapter (written into the weights) ===")
    store.repersonalize(UID)
    print("    done.")

    print("\n=== 4. NEW CONTEXT (empty prompt) — base (adapter OFF) vs personalized (ON) ===")
    for q in PROBES:
        store.model.reset()
        base = store.model.chat(q, max_new_tokens=80)
        personalized = store.chat(UID, q, max_new_tokens=80)
        print(f"\n  Q: {q}")
        print(f"    [base]         {base}")
        print(f"    [personalized] {personalized}")


if __name__ == "__main__":
    main()
