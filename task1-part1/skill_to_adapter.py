"""skill.md -> D2L LoRA adapter -> base-vs-adapter evaluation.

Pipeline:
  1. Solve a problem with the BASE model (no adapter).
  2. Turn a skill.md document into a LoRA adapter via D2L `internalize()`, and
     export the generated adapter weights to disk.
  3. Solve the SAME problem with the BASE + ADAPTER model.
  4. If generation is deterministic (greedy decode reproduces itself), score
     correctness of each answer against an expected string (normalized
     exact / contains match).
  5. Write both models' outputs + scoring into a JSON file.

Run inside the `doc2lora` conda env with PYTHONNOUSERSITE=1, e.g.:
    conda run -n doc2lora env PYTHONNOUSERSITE=1 python skill_to_adapter.py

Requires the D2L package (`ctx_to_lora`, installed editable from
/home/nikash/projects/doc-to-lora) and flash-attn (built for sm_90).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path

import torch

import ctx_to_lora
from ctx_to_lora.eval_utils import normalize_answer
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# get_tokenizer resolves custom chat templates relative to CWD
# (chat_templates/<model>.jinja), so we run from the D2L repo root.
D2L_ROOT = Path(ctx_to_lora.__file__).resolve().parents[2]

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    "/home/nikash/projects/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/"
    "pytorch_model.bin"
)
DEFAULT_PROBLEM = (
    "What is the single most important rule, the 'Iron Law', of this debugging "
    "process? Answer in one sentence."
)
# The skill's Iron Law is "NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST".
# Correct = the answer expresses that core principle (find the root cause
# *before* fixing). Both normalized key phrases must be present.
DEFAULT_EXPECTED = "root cause | before"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill", default=str(HERE / "skills/systematic-debugging.md"))
    p.add_argument(
        "--problem",
        default=DEFAULT_PROBLEM,
        help="Question string, or @path to read it from a file.",
    )
    p.add_argument(
        "--expected",
        default=DEFAULT_EXPECTED,
        help="Expected answer. Use ' | ' to require multiple key phrases all present.",
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--base-override", default="unsloth/gemma-2-2b-it")
    p.add_argument("--adapter-out", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--reload-adapter",
        default=None,
        help="Path to an exported .pt adapter; load it and regenerate instead of "
        "internalizing (sanity check that a saved adapter reproduces results).",
    )
    return p.parse_args()


def _resolve_text(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text()
    return value


def correctness(output: str, expected: str) -> bool:
    """Normalized exact-or-contains match. ' | ' splits required key phrases."""
    norm_out = normalize_answer(output)
    keys = [k for k in (expected.split("|") if "|" in expected else [expected])]
    keys = [normalize_answer(k) for k in keys if k.strip()]
    if len(keys) == 1:
        return norm_out == keys[0] or keys[0] in norm_out
    # multi-key: every key phrase must be present
    return all(k in norm_out for k in keys)


def main() -> None:
    args = parse_args()
    problem = _resolve_text(args.problem)
    skill_path = Path(args.skill).resolve()
    if args.reload_adapter:
        args.reload_adapter = str(Path(args.reload_adapter).resolve())
    # resolve outputs to absolute before chdir into the D2L repo root
    args.adapter_out = str(Path(args.adapter_out).resolve()) if args.adapter_out else None
    args.out = str(Path(args.out).resolve()) if args.out else None
    os.chdir(D2L_ROOT)
    adapter_out = Path(
        args.adapter_out
        or HERE / "adapters" / f"{skill_path.stem}.lora.pt"
    )
    out_path = Path(
        args.out or HERE / "outputs" / f"{skill_path.stem}.result.json"
    )
    adapter_out.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load model with the ungated base override -----------------------------
    print(f"Loading checkpoint {args.checkpoint} (base -> {args.base_override})")
    state_dict = torch.load(args.checkpoint, weights_only=False)
    if state_dict.get("base_model_name_or_path") == "google/gemma-2-2b-it":
        state_dict["base_model_name_or_path"] = args.base_override
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    chat = [{"role": "user", "content": problem}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    def greedy() -> str:
        out = model.generate(
            input_ids=chat_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        return tokenizer.decode(
            out[0][chat_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    # --- step 1: base model ----------------------------------------------------
    model.reset()
    base_output = greedy()
    base_output_2 = greedy()

    # --- step 2: skill.md -> adapter ------------------------------------------
    if args.reload_adapter:
        print(f"Reloading adapter from {args.reload_adapter}")
        blob = torch.load(args.reload_adapter, weights_only=False)
        model.generated_loras = _to_device(blob["generated_loras"], model.device)
        model.patch_lora_forward()
    else:
        skill_text = skill_path.read_text()
        print(f"Internalizing skill {skill_path} ({len(skill_text)} chars) -> LoRA")
        model.internalize(skill_text)
        # export adapter weights + metadata
        torch.save(
            {
                "generated_loras": _to_device(model.generated_loras, "cpu"),
                "skill_file": str(skill_path),
                "base_model": model.base_model.name_or_path,
                "checkpoint": args.checkpoint,
                "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            },
            adapter_out,
        )
        print(f"Exported adapter -> {adapter_out}")

    # --- step 3: base + adapter ------------------------------------------------
    adapter_output = greedy()
    adapter_output_2 = greedy()

    # --- step 4: determinism + correctness ------------------------------------
    deterministic = (base_output == base_output_2) and (
        adapter_output == adapter_output_2
    )
    if deterministic and args.expected:
        base_correct = correctness(base_output, args.expected)
        adapter_correct = correctness(adapter_output, args.expected)
    else:
        base_correct = adapter_correct = None

    result = {
        "skill_file": str(skill_path),
        "checkpoint": args.checkpoint,
        "base_model": model.base_model.name_or_path,
        "problem": problem,
        "expected": args.expected,
        "base_output": base_output,
        "adapter_output": adapter_output,
        "deterministic": deterministic,
        "base_correct": base_correct,
        "adapter_correct": adapter_correct,
        "adapter_path": str(adapter_out) if not args.reload_adapter else args.reload_adapter,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))

    print("\n" + "=" * 80)
    print(f"deterministic={deterministic}  base_correct={base_correct}  "
          f"adapter_correct={adapter_correct}")
    print(f"wrote {out_path}")
    print("=" * 80)
    print(f"\n[BASE]\n{base_output}\n\n[ADAPTER]\n{adapter_output}")


def _to_device(obj, device):
    """Recursively move tensors in a (possibly nested) structure to a device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


if __name__ == "__main__":
    main()
