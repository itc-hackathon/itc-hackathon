"""Thin wrapper around the Doc-to-LoRA model for AgentHN.

Loads the hypernetwork checkpoint once and exposes the small surface we need:
internalize a document into a LoRA adapter, generate text, and snapshot/restore
the generated adapter (so the personalization layer can cache + swap per user).
"""

from __future__ import annotations

import sys as _sys

import torch

from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling import hypernet
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

from . import config

# from_state_dict unpickles dataclasses that were saved under this module alias.
_sys.modules.setdefault("ctx_to_lora.modeling_utils", hypernet)


def _load_custom_chat_template(tokenizer, model_name: str) -> None:
    """Load D2L's custom chat template by ABSOLUTE path (CWD-independent).

    ctx_to_lora.get_tokenizer resolves the template via a CWD-relative path
    ("chat_templates/<model>.jinja"), so it silently falls back to the model's
    built-in template when run from anywhere but the doc-to-lora repo — and the
    built-in gemma template rejects the `system` role that tokenize_ctx_text
    uses. We point at the repo copy explicitly and apply the same normalization
    get_tokenizer does.
    """
    template_path = config.D2L_REPO / "chat_templates" / f"{model_name}.jinja"
    if not template_path.exists():
        return
    text = template_path.read_text().replace("    ", "").replace("\n", "")
    tokenizer.chat_template = text


class D2LModel:
    """Loaded D2L model + tokenizer with a minimal internalize/generate API."""

    def __init__(self, model: ModulatedPretrainedModel, tokenizer, ctx_tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.ctx_tokenizer = ctx_tokenizer

    @classmethod
    def load(cls, checkpoint=config.CHECKPOINT) -> "D2LModel":
        state_dict = torch.load(str(checkpoint), weights_only=False)
        model = ModulatedPretrainedModel.from_state_dict(
            state_dict, train=False, use_sequence_packing=False
        )
        model.reset()

        tokenizer = get_tokenizer(model.base_model.name_or_path)
        _load_custom_chat_template(tokenizer, model.base_model.name_or_path)

        # Separate tokenizer for context (the doc being internalized). Its custom
        # template supports the `system` role required by tokenize_ctx_text.
        ctx_name = model.ctx_encoder.base_model.name_or_path
        ctx_tokenizer = get_tokenizer(ctx_name)
        _load_custom_chat_template(ctx_tokenizer, ctx_name)

        return cls(model, tokenizer, ctx_tokenizer)

    # --- adapter lifecycle ------------------------------------------------
    def internalize(self, doc: str) -> None:
        """Convert a document into a LoRA adapter and make it active.

        NOTE: replaces the active adapter (does not accumulate). For per-user
        personalization, snapshot() the result and restore() to swap.

        Mirrors ModulatedPretrainedModel.internalize but uses our own
        absolute-path-templated ctx tokenizer instead of building one from CWD.
        """
        ctx_ids = tokenize_ctx_text(dict(context=[doc]), self.ctx_tokenizer)["ctx_ids"]
        self.model._internalize_from_ids(
            torch.tensor(ctx_ids, device=self.model.device)
        )

    def reset(self) -> None:
        self.model.reset()

    def snapshot(self):
        """Return the currently active generated adapter (to cache per user)."""
        return self.model.generated_loras

    def restore(self, adapter) -> None:
        """Make a previously snapshotted adapter active (swap per user).

        generate() only *binds* per-call A/B onto already-patched lora forwards
        (it doesn't install them), so we must re-establish the clean patched
        state here: reset reverts any prior binding, patch_lora_forward installs
        fresh lora_forward partials, then we set the adapter. This makes each
        swap idempotent and avoids stacking partials across repeated calls.
        """
        self.model.reset()
        self.model.patch_lora_forward()
        self.model.generated_loras = adapter

    # --- generation -------------------------------------------------------
    @torch.inference_mode()
    def chat(self, message: str, max_new_tokens: int = 512) -> str:
        chat = [{"role": "user", "content": message}]
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        completion = out[0][input_ids.shape[1] :]  # drop the echoed prompt
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()
