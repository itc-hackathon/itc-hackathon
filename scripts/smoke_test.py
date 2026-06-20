"""Sanity check: load the D2L model from agenthn and prove internalize/swap works.

Run with the shared venv:
  /home/ubuntu/doc-to-lora/.venv/bin/python scripts/smoke_test.py
(requires HF login for the gated google/gemma-2-2b-it base model)
"""

from agenthn.core.config import D2L_REPO
from agenthn.core.model import D2LModel

QUESTION = "Tell me about Sakana AI."


def main():
    m = D2LModel.load()

    print("=== base (no context) ===")
    m.reset()
    print(m.chat(QUESTION, max_new_tokens=128))

    doc = (D2L_REPO / "data" / "sakana_wiki.txt").read_text()
    m.internalize(doc)
    print("\n=== after internalize(sakana_wiki) ===")
    print(m.chat(QUESTION, max_new_tokens=128))


if __name__ == "__main__":
    main()
