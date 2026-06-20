import argparse
import json
import random
import string
import time

import torch
from peft import get_peft_config, PeftConfig

from hyper_llm_modulator.utils import get_layers, embed_texts
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint, save_lora

DEFAULT_TASK = (
    "A bakery sells cupcakes in boxes of 6 and cookies in bags of 8. Last week, "
    "the bakery sold 17 boxes of cupcakes and 23 bags of cookies, but had to discard "
    "15% of the cupcakes and 10% of the cookies due to a refrigeration issue before "
    "selling them. Each remaining cupcake sells for $2.50 and each remaining cookie "
    "sells for $1.75. If the bakery's total operating cost for the week was $310, "
    "what was the bakery's net profit, rounded to the nearest cent?"
)

GUIDANCE_EXAMPLES = [
    "This task challenges your problem-solving abilities through mathematical "
    "reasoning. You must carefully read each scenario and systematically work "
    "through the data to compute the final outcome.",
    "Use your programming skill to think algorithmically, logically and "
    "systematically. Carefully read each scenario and systematically work "
    "through the data to compute the final outcome. Use your math skill to "
    "calculate correctly.",
]


def build_guidance_prompt(task, base_answer):
    examples = "\n".join(f'- "{ex}"' for ex in GUIDANCE_EXAMPLES)
    return (
        "Below is a task and an attempt at it that contains mistakes.\n\n"
        f"TASK:\n{task}\n\n"
        f"ATTEMPT (for your reference only):\n{base_answer}\n\n"
        "Privately notice which skill(s) the attempt needed but applied poorly "
        "(e.g. mathematical reasoning, algorithmic thinking, careful reading). "
        "Then write a short briefing for an expert about to attempt this TYPE of "
        "task for the first time, in the exact same style as these examples:\n\n"
        f"{examples}\n\n"
        "Your briefing must: name the relevant skill(s), explicitly instruct the "
        "solver to carefully read the scenario and systematically work through it "
        "step by step, and stay generic to the task type. Never mention the specific "
        "numbers, never reference \"the attempt\" or what it got wrong, and never use "
        "words like \"attempt\", \"mistake\", or \"error\" — write it as a forward-looking "
        "briefing, not a critique. Write only the briefing, under 250 tokens."
    )


def generate(model, tokenizer, user_message, max_new_tokens, device, prefill=""):
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    chat_prompt += prefill
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return (prefill + continuation).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2l-dir", default="trained_t2l/llama_8b_t2l")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--guidance-max-tokens", type=int, default=300)
    parser.add_argument(
        "--assistant-prefill",
        default="Let's think step by step. ",
        help=(
            "Prefix the model's response with this before generating the task answer "
            "(matches gsm8k's own assistant_prefill convention in tasks/gsm8k/metadata.yaml, "
            "which is what this T2L checkpoint's math/reasoning adapters were SFT-trained to "
            "continue from). Applied to both the base and adapter-boosted answers so the "
            "comparison isolates the adapter's effect."
        ),
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    checkpoint_path = f"{args.t2l_dir}/hypermod.pt"
    (
        hypermod_args,
        hypermod,
        model,
        tokenizer,
        emb_model,
        emb_tokenizer,
        task_desc_format_fn,
        pooling_fn,
    ) = load_hypermod_checkpoint(checkpoint_path, device)
    layer_indices = torch.tensor(
        range(len(get_layers(model))), dtype=torch.long, device=device
    )
    peft_config = get_peft_config(
        PeftConfig.from_json_file(f"{args.t2l_dir}/adapter_config.json")
    )

    print(f"\n=== Task ===\n{args.task}")

    # Step 1: base model attempts the task ("default" adapter is zero-init, i.e. a no-op)
    model.set_adapter("default")
    base_answer = generate(
        model, tokenizer, args.task, args.max_new_tokens, device, args.assistant_prefill
    )
    print(f"\n=== Base model answer ===\n{base_answer}")

    # Step 2: base model self-assesses and produces concise guidance
    guidance_prompt = build_guidance_prompt(args.task, base_answer)
    guidance_text = generate(
        model, tokenizer, guidance_prompt, args.guidance_max_tokens, device
    )
    print(f"\n=== Self-assessment guidance ===\n{guidance_text}")

    # Step 3: generate a LoRA adapter from the guidance text via T2L
    task_emb = embed_texts(
        [guidance_text], emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn, device
    )
    encoded_task_emb = hypermod.task_encoder(task_emb)["encoded_task_emb"].detach()
    lora_sd = hypermod.gen_lora(layer_indices, encoded_task_emb)

    curtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    uuid = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    lora_dir = f"{args.t2l_dir}/extras/user_generated/{curtime}_{uuid}/"
    save_lora(lora_sd, peft_config, lora_dir)
    with open(f"{lora_dir}/task_desc.txt", "w") as f:
        f.write(guidance_text)
    print(f"\nSaved generated LoRA to {lora_dir}")

    # Step 4: retry the same task with the generated adapter active
    model.load_adapter(lora_dir, "skill_adapter")
    model.set_adapter("skill_adapter")
    adapter_answer = generate(
        model, tokenizer, args.task, args.max_new_tokens, device, args.assistant_prefill
    )
    print(f"\n=== Adapter-boosted answer ===\n{adapter_answer}")

    result_path = f"{lora_dir}/skill_acquisition_result.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "task": args.task,
                "base_answer": base_answer,
                "guidance_text": guidance_text,
                "lora_dir": lora_dir,
                "adapter_answer": adapter_answer,
            },
            f,
            indent=2,
        )
    print(f"\nSaved transcript to {result_path}")


if __name__ == "__main__":
    main()
