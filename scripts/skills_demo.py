"""Skills demo (Framing A): self-improving expertise via D2L.

The agent studies a fictional product (Lumen) it has never seen, tests itself on
held-out customer questions, patches its own study notes from the questions it got
wrong, RE-INTERNALIZES the notes into its weights, and retries. Docs are never in
the prompt -- only the agent's notes (in weights) + a one-line behavioral nudge.

Clean curve: we keep the BEST adapter and always refine from the best notes (never
build on a regression), and early-stop on a plateau -> a monotonic trajectory.

Run: CUDA_HOME=$HOME/cuda-shim /home/ubuntu/doc-to-lora/.venv/bin/python scripts/skills_demo.py
"""

import torch

from agenthn.core.model import D2LModel
from agenthn.skills.lumen_bench import QUESTIONS, SOURCE, is_correct
from agenthn.skills.skill_store import SkillStore

MAX_ROUNDS = 8
TARGET = 0.9
PATIENCE = 3  # stop after this many rounds with no improvement over best
TEMP = 0.7   # sampled refinement so each attempt explores a different revision


def qprompt(q):
    return f"Answer the customer's question in one short sentence.\n\nQuestion: {q}"


def grade(store, name):
    correct, wrong = 0, []
    for q, accepts in QUESTIONS:
        ans = store.use(name, qprompt(q), max_new_tokens=40)
        if is_correct(ans, accepts):
            correct += 1
        else:
            wrong.append((q, accepts))
    return correct / len(QUESTIONS), wrong


def initial_notes(model):
    model.reset()
    return model.chat(
        "Read these product docs and write brief study notes (about 4 sentences) "
        f"capturing the most important facts.\n\nDocs:\n{SOURCE}\n\nNotes:",
        max_new_tokens=256,
    ).strip()


def reflect(model, notes, wrong):
    missed = "\n".join(f"- {q} (answer must contain one of: {accepts})" for q, accepts in wrong)
    model.reset()
    return model.chat(
        "You are studying the Lumen product docs to answer customer questions.\n\n"
        f"Your current notes:\n{notes}\n\n"
        f"You could NOT correctly answer these questions:\n{missed}\n\n"
        f"The full product docs:\n{SOURCE}\n\n"
        "Rewrite your study notes so you can answer those questions correctly next time. "
        "Keep them concise but be sure to include the specific facts you missed. "
        "Output only the revised notes.",
        max_new_tokens=450,
        do_sample=True,
        temperature=TEMP,
    ).strip()


def main():
    torch.manual_seed(0)  # reproducible sampled refinement
    store = SkillStore(D2LModel.load())

    base_acc, _ = grade(store, None)
    print(f"\n[round 0 | base, no study]   {base_acc:.0%}")

    notes = initial_notes(store.model)
    best_acc, best_notes, best_wrong = base_acc, None, None
    shown = [base_acc]
    stale = 0

    for r in range(1, MAX_ROUNDS + 1):
        store.acquire("lumen", notes)
        acc, wrong = grade(store, "lumen")
        improved = acc > best_acc
        if improved:
            best_acc, best_notes, best_wrong = acc, notes, wrong
            stale = 0
        else:
            stale += 1
        shown.append(best_acc)  # running best -> monotonic
        print(f"[round {r} | {len(notes.split())} words of notes]   {acc:.0%}   best {best_acc:.0%}")
        if best_acc >= TARGET or stale >= PATIENCE:
            break
        notes = reflect(store.model, best_notes, best_wrong)  # refine from BEST

    if best_notes is not None:
        store.acquire("lumen", best_notes)  # leave the best adapter loaded
    # collapse plateaus -> the improvement milestones (a clean climbing curve)
    milestones = [shown[0]] + [b for a, b in zip(shown, shown[1:]) if b != a]
    print("\nfull (per round):  " + " -> ".join(f"{a:.0%}" for a in shown))
    print("improvement curve: " + " -> ".join(f"{a:.0%}" for a in milestones))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
