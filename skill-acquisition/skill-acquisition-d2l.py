import argparse
import importlib.machinery
import json
import math
import re
import sys
import time
import types

# deepspeed's import-time CUDA op compatibility check crashes whenever torch
# sees a GPU but no nvcc/CUDA_HOME is installed (no toolkit, just the pip CUDA
# runtime libs) -- stub it out before transformers/peft pull it in, since we
# never actually need deepspeed for single-GPU inference. The stub needs a
# real __spec__ or importlib.util.find_spec() (used by transformers' own
# is_deepspeed_available() check) raises instead of just returning truthy.
_deepspeed_stub = types.ModuleType("deepspeed")
_deepspeed_stub.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)
sys.modules.setdefault("deepspeed", _deepspeed_stub)

import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

DOC = """\
# Newton's Laws and Classical Mechanics: Formula Reference

## Newton's First Law (Law of Inertia)
An object at rest stays at rest, and an object in motion stays in motion at
constant velocity, unless acted on by a net external force. If the net force
on an object is zero, its acceleration is zero.

## Newton's Second Law
The net force on an object equals its mass times its acceleration:
F = m * a
Equivalently: a = F / m, and m = F / a.
Units: F in newtons (N), m in kilograms (kg), a in meters per second squared
(m/s^2). 1 N = 1 kg*m/s^2.

## Newton's Third Law
For every action there is an equal and opposite reaction: the force object A
exerts on object B is equal in magnitude and opposite in direction to the
force object B exerts on object A.

## Weight
Weight is the force of gravity on an object near a planet's surface:
W = m * g
where g = 9.8 m/s^2 near Earth's surface.

## Friction
The force of friction between two surfaces:
F_friction = mu * N
where mu is the (dimensionless) coefficient of friction and N is the normal
force (on a flat horizontal surface, N = m * g).

## Momentum
Momentum is the product of mass and velocity:
p = m * v

## Impulse
Impulse equals the change in momentum, and also equals force times the time
interval over which it acts:
J = F * delta_t = delta_p

## Newton's Law of Universal Gravitation
The gravitational force between two masses:
F = G * m1 * m2 / r^2
where G = 6.674e-11 N*m^2/kg^2, m1 and m2 are the two masses, and r is the
distance between their centers.

## Centripetal Force
The net force required to keep an object of mass m moving in a circle of
radius r at speed v:
F_c = m * v^2 / r

## Kinematics (often combined with F = m * a)
v = v0 + a*t
x = x0 + v0*t + (1/2)*a*t^2
v^2 = v0^2 + 2*a*(x - x0)

# Worked Examples

Problem: A 4 kg object experiences a net force of 20 N. What is its
acceleration?
Solution: Let's think step by step. Newton's second law states F = m*a, so
a = F/m. Substituting, a = 20 N / 4 kg = 5 m/s^2.
Answer: 5 m/s^2

Problem: What is the weight of a 12 kg object on Earth's surface?
Solution: Let's think step by step. Weight is W = m*g. Using g = 9.8 m/s^2,
W = 12 kg * 9.8 m/s^2 = 117.6 N.
Answer: 117.6 N

Problem: A 20 kg crate rests on a horizontal floor with a coefficient of
friction of 0.25. What is the maximum friction force?
Solution: Let's think step by step. The normal force on a horizontal surface
equals the weight, N = m*g = 20 kg * 9.8 m/s^2 = 196 N. Friction is
F = mu*N = 0.25 * 196 N = 49 N.
Answer: 49 N

Problem: A 3 kg ball moves at 7 m/s. What is its momentum?
Solution: Let's think step by step. Momentum is p = m*v = 3 kg * 7 m/s =
21 kg*m/s.
Answer: 21 kg*m/s

Problem: A 0.4 kg ball is hit with a force of 15 N for 0.1 s. What is the
impulse (change in momentum)?
Solution: Let's think step by step. Impulse equals force times time,
J = F*delta_t = 15 N * 0.1 s = 1.5 kg*m/s. This equals the change in
momentum.
Answer: 1.5 kg*m/s

Problem: Two masses of 5 kg and 8 kg are 4 m apart. What is the
gravitational force between them?
Solution: Let's think step by step. Newton's law of gravitation gives
F = G*m1*m2/r^2. Substituting G = 6.674e-11 N*m^2/kg^2, m1 = 5 kg, m2 = 8 kg,
r = 4 m: F = 6.674e-11 * 5 * 8 / 16 = 6.674e-11 * 2.5 = 1.6685e-10 N.
Answer: 1.6685e-10 N

Problem: A 1.5 kg mass moves in a circle of radius 2 m at a speed of 6 m/s.
What centripetal force is required?
Solution: Let's think step by step. Centripetal force is
F_c = m*v^2/r = 1.5 kg * (6 m/s)^2 / 2 m = 1.5 * 36 / 2 = 27 N.
Answer: 27 N

Problem: A 500 kg car accelerates from rest to 15 m/s in 3 seconds. What net
force acted on the car?
Solution: Let's think step by step. First find acceleration using
kinematics: a = (v - v0)/t = (15 m/s - 0 m/s)/3 s = 5 m/s^2. Then apply
Newton's second law: F = m*a = 500 kg * 5 m/s^2 = 2500 N.
Answer: 2500 N

Problem: A 60 kg astronaut pushes on a 900 kg space module with a force of
40 N. What force does the module exert on the astronaut?
Solution: Let's think step by step. By Newton's third law, the force the
module exerts on the astronaut is equal in magnitude and opposite in
direction to the force the astronaut exerts on the module. Since the
astronaut pushes with 40 N, the module pushes back with 40 N.
Answer: 40 N

Problem: A 10 kg object has two horizontal forces acting on it: 25 N to the
right and 25 N to the left. What is its acceleration?
Solution: Let's think step by step. The net force is the vector sum:
25 N - 25 N = 0 N. By Newton's second law, a = F_net/m = 0 N / 10 kg =
0 m/s^2.
Answer: 0 m/s^2
"""

HOLDOUT_QUESTIONS = [
    dict(
        question="A 5 kg object experiences a net force of 18 N. What is its acceleration?",
        expected=18 / 5,
        formula="F = m*a",
    ),
    dict(
        question="What is the weight of an 8 kg object on Earth's surface (g = 9.8 m/s^2)?",
        expected=8 * 9.8,
        formula="W = m*g",
    ),
    dict(
        question=(
            "A 15 kg box sits on a horizontal floor with a coefficient of "
            "friction of 0.3. What is the maximum friction force?"
        ),
        expected=0.3 * (15 * 9.8),
        formula="F_friction = mu*N, N = m*g",
    ),
    dict(
        question="A 6 kg object moves at 4 m/s. What is its momentum?",
        expected=6 * 4,
        formula="p = m*v",
    ),
    dict(
        question=(
            "A 0.6 kg ball experiences a force of 12 N for 0.25 s. What is "
            "the impulse (change in momentum)?"
        ),
        expected=12 * 0.25,
        formula="J = F*delta_t",
    ),
    dict(
        question=(
            "Two masses of 10 kg and 3 kg are 2 m apart. What is the "
            "gravitational force between them (G = 6.674e-11 N*m^2/kg^2)?"
        ),
        expected=6.674e-11 * 10 * 3 / 2**2,
        formula="F = G*m1*m2/r^2",
    ),
    dict(
        question=(
            "A 2 kg mass moves in a circle of radius 0.5 m at 8 m/s. What "
            "centripetal force is required?"
        ),
        expected=2 * 8**2 / 0.5,
        formula="F_c = m*v^2/r",
    ),
    dict(
        question=(
            "A 1000 kg car accelerates from rest to 20 m/s in 4 seconds. "
            "What net force acted on the car?"
        ),
        expected=1000 * (20 / 4),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question=(
            "A 1200 kg car traveling at 24 m/s skids to a stop on a "
            "horizontal road with a coefficient of friction of 0.4. What is "
            "the magnitude of the deceleration caused by friction?"
        ),
        expected=0.4 * 9.8,
        formula="F_friction = mu*m*g, a = F/m (mass cancels)",
    ),
    dict(
        question=(
            "Two masses of 50 kg and 70 kg are 5 m apart. What is the "
            "gravitational force between them (G = 6.674e-11 N*m^2/kg^2)?"
        ),
        expected=6.674e-11 * 50 * 70 / 5**2,
        formula="F = G*m1*m2/r^2",
    ),
    dict(
        question=(
            "A 1500 kg car traveling at 25 m/s comes to a stop in 5 seconds "
            "due to braking. What is the magnitude of the net braking force?"
        ),
        expected=1500 * (25 / 5),
        formula="a = (v-v0)/t, F = m*a",
    ),
    dict(
        question=(
            "A 2 kg block sliding at 10 m/s on a horizontal surface "
            "(coefficient of friction 0.5) comes to a stop. What is the "
            "magnitude of the impulse needed to stop it?"
        ),
        expected=2 * 10,
        formula="J = delta_p = m*delta_v (mu is a distractor, not needed)",
    ),
    dict(
        question=(
            "A 0.8 kg ball moving at 5 m/s is brought to rest by a constant "
            "force acting over 0.4 s. What is the magnitude of the average "
            "force applied?"
        ),
        expected=0.8 * 5 / 0.4,
        formula="delta_p = m*delta_v, F = delta_p/delta_t",
    ),
]

NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
LIST_MARKER_RE = re.compile(r"(?m)^\s*\*{0,2}\d+\.\s")


def normalize_scientific_notation(text):
    # strip numbered-step markers ("1. ", "**4.** ") at line starts -- they
    # get misparsed as numeric answers, e.g. a truncated response ending in
    # "4. Cancel..." can spuriously match a small expected value like 3.92.
    text = LIST_MARKER_RE.sub("", text)
    # the model sometimes writes "5.0055 × 10⁻¹⁰" (unicode superscripts)
    # instead of "5.0055e-10" -- fold both into the same parseable form.
    text = text.translate(SUPERSCRIPT_MAP)
    return re.sub(r"(\d+\.?\d*)\s*[×x\*]\s*10\^?(-?\d+)", r"\1e\2", text)


def check_numeric(answer, expected, rel_tol=0.02):
    for match in NUMBER_RE.finditer(normalize_scientific_notation(answer)):
        try:
            val = float(match.group())
        except ValueError:
            continue
        if math.isclose(val, expected, rel_tol=rel_tol):
            return True
    return False


def generate_text(model, tokenizer, user_message, max_new_tokens, device, prefill=""):
    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    chat_str += prefill
    inputs = tokenizer(chat_str, return_tensors="pt", add_special_tokens=False).to(device)
    # Qwen3-4B-Instruct-2507 ships with do_sample=True (temp 0.7) in its
    # generation_config, which makes identical inputs produce different
    # outputs across runs -- pin greedy decoding for reproducible comparisons.
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return (prefill + continuation).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2l-dir", default="trained_d2l/qwen_4b_d2l")
    parser.add_argument("--checkpoint", default="checkpoint-20000/pytorch_model.bin")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--assistant-prefill",
        default="Let's think step by step. ",
        help=(
            "Prefix the model's response with this before generating the answer. "
            "Applied identically to the base, in-context, and adapter-boosted "
            "answers so the comparison isolates the adapter's effect."
        ),
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoint_path = f"{args.d2l_dir}/{args.checkpoint}"
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    print(f"\n=== Doc ({len(DOC)} chars) ===\n{DOC}")

    # Step 1: base model attempts each held-out question with no context at all
    model.reset()
    base_answers = [
        generate_text(model, tokenizer, q["question"], args.max_new_tokens, device, args.assistant_prefill)
        for q in HOLDOUT_QUESTIONS
    ]

    # Step 2: in-context ceiling -- paste the whole formula sheet + worked
    # examples directly into the prompt, still with no internalized LoRA.
    in_context_answers = [
        generate_text(
            model,
            tokenizer,
            f"{DOC}\n\n{q['question']}",
            args.max_new_tokens,
            device,
            args.assistant_prefill,
        )
        for q in HOLDOUT_QUESTIONS
    ]

    # Step 3: internalize the doc into a D2L-generated LoRA adapter, then
    # retry the bare held-out questions (no doc in the prompt this time).
    model.internalize(DOC)
    adapter_answers = [
        generate_text(model, tokenizer, q["question"], args.max_new_tokens, device, args.assistant_prefill)
        for q in HOLDOUT_QUESTIONS
    ]
    model.reset()

    rows = []
    for q, base, ic, ad in zip(HOLDOUT_QUESTIONS, base_answers, in_context_answers, adapter_answers):
        row = {
            "question": q["question"],
            "formula": q["formula"],
            "expected": q["expected"],
            "base_answer": base,
            "base_pass": check_numeric(base, q["expected"]),
            "in_context_answer": ic,
            "in_context_pass": check_numeric(ic, q["expected"]),
            "adapter_answer": ad,
            "adapter_pass": check_numeric(ad, q["expected"]),
        }
        rows.append(row)
        print(f"\n{'=' * 80}\nQ: {q['question']}  [{q['formula']}]\nExpected: {q['expected']:g}\n{'=' * 80}")
        print(f"\n--- base ({'PASS' if row['base_pass'] else 'FAIL'}) ---\n{base}")
        print(f"\n--- in_context ({'PASS' if row['in_context_pass'] else 'FAIL'}) ---\n{ic}")
        print(f"\n--- adapter ({'PASS' if row['adapter_pass'] else 'FAIL'}) ---\n{ad}")

    n = len(rows)
    base_n = sum(r["base_pass"] for r in rows)
    ic_n = sum(r["in_context_pass"] for r in rows)
    ad_n = sum(r["adapter_pass"] for r in rows)
    print(f"\n\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    print(f"base={base_n}/{n}  in_context={ic_n}/{n}  adapter={ad_n}/{n}")

    output_path = (
        args.output
        or f"{args.d2l_dir}/skill_acquisition_physics_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w") as f:
        json.dump({"doc": DOC, "rows": rows}, f, indent=2)
    print(f"\nSaved full transcript to {output_path}")


if __name__ == "__main__":
    main()
