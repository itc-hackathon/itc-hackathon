"""Thin wrapper around the Doc-to-LoRA model for AgentHN.

Loads the hypernetwork checkpoint once and exposes the small surface we need:
internalize a document into a LoRA adapter, generate text, and snapshot/restore
the generated adapter (so the personalization layer can cache + swap per user).
"""

from __future__ import annotations

import sys as _sys

import torch

from ctx_to_lora.data.definitions import CTX_AFFIXES
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

    def internalize_segment(self, doc: str):
        """Internalize a doc and return its adapter as a detached snapshot.

        Single-chunk adapter of shape {module: {A,B: [1, n_layers, r, dim]}}.
        These are the per-segment "memory" deltas the long-horizon memory stacks
        along the chunk dimension (rank-concatenation) — see compose().
        """
        self.reset()
        self.internalize(doc)
        snap = self.model.generated_loras
        # detach() drops the hypernetwork autograd graph (we never backprop through
        # stored memory) — without it, every nap retains its forward-pass
        # activations and GPU memory explodes after a few hundred naps. clone()
        # makes the copy independent of the next internalize() overwrite.
        return {
            mod: {k: v.detach().clone() for k, v in mats.items()}
            for mod, mats in snap.items()
        }

    @staticmethod
    def compose(segments: list):
        """Rank-concatenate per-segment adapters into one composed adapter.

        Each segment is {module: {A,B: [1, n_layers, r, dim]}}; we concat along
        the chunk dim (0) -> [n_seg, n_layers, r, dim]. Paired with
        n_ctx_chunks=[n_seg] at generate time, combine_lora() flattens this into a
        single rank-(n_seg*r) adapter whose delta is the SUM of the per-segment
        deltas — i.e. all memories active at once. This is exactly D2L's trained
        long-context chunking mechanism, reused for independently-encoded memories.
        """
        if not segments:
            return None
        modules = segments[0].keys()
        return {
            mod: {
                "A": torch.cat([s[mod]["A"] for s in segments], dim=0),
                "B": torch.cat([s[mod]["B"] for s in segments], dim=0),
            }
            for mod in modules
        }

    @torch.inference_mode()
    def respond(
        self,
        messages: list,
        composed=None,
        n_segments: int = 1,
        max_new_tokens: int = 256,
    ) -> str:
        """Deterministic (greedy) generation over a chat `messages` list.

        One generation path for every memory strategy:
          - NapLoRA: tiny `messages` (recent turns + query) + a composed memory
            adapter (rank-concatenated segments) -> set composed, n_segments.
          - text baselines (vanilla / markdown): full context inside `messages`,
            composed=None -> plain base-model behavior.
        """
        self.model.reset()
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        if composed is not None:
            self.model.patch_lora_forward()
            self.model.generated_loras = composed
            gen_kwargs["n_ctx_chunks"] = torch.tensor(
                [n_segments], device=self.model.device
            )
        out = self.model.generate(**gen_kwargs)
        completion = out[0][input_ids.shape[1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    def count_tokens(self, text: str) -> int:
        """Token count under the base tokenizer (for context-budget accounting)."""
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

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

    # --- multi-chunk memory (rank concatenation) --------------------------
    # The chunk-trained checkpoint (config.CHUNK_CHECKPOINT) can rank-concatenate
    # several independently-internalized adapters via combine_lora — this is the
    # primitive the memory track is built on. Each chunk is encoded on its own,
    # so adapters can be cached per memory and stacked incrementally.
    @torch.inference_mode()
    def encode_chunk(self, text: str):
        """Encode one memory string into a single-chunk adapter (A/B: [1, ...]).

        Builds the interior-chunk token form (prefix + raw tokens + suffix) that
        the chunk-trained hypernet expects, runs one generate_weights pass, and
        returns a detached/cloned adapter dict that can be cached and later
        rank-concatenated with others (see chat_memory).
        """
        affixes = CTX_AFFIXES[self.model.base_model.name_or_path]
        raw = self.ctx_tokenizer(text.strip(), add_special_tokens=False)["input_ids"]
        ids = affixes["prefix"] + raw + affixes["suffix"]
        ctx_ids = torch.tensor([ids], device=self.model.device)
        ctx_attn = torch.ones_like(ctx_ids)
        loras, _ = self.model.generate_weights(ctx_ids, ctx_attn)
        return {m: {"A": v["A"].clone(), "B": v["B"].clone()} for m, v in loras.items()}

    @staticmethod
    def stack_chunks(chunks: list) -> dict:
        """Rank-concatenate per-memory adapters along the chunk dim (dim 0)."""
        modules = chunks[0].keys()
        return {
            m: {
                "A": torch.cat([c[m]["A"] for c in chunks], dim=0),
                "B": torch.cat([c[m]["B"] for c in chunks], dim=0),
            }
            for m in modules
        }

    @torch.inference_mode()
    def chat_memory(self, message: str, chunks: list, scalers=None,
                    max_new_tokens: int = 64) -> str:
        """Generate with a rank-concatenated stack of memory-chunk adapters.

        chunks: list of single-chunk adapter dicts (from encode_chunk). combine_lora
        (invoked inside model.generate via n_ctx_chunks) stacks them along the rank
        dim. The prompt carries NO memory text — recall comes from the weights.
        """
        if not chunks:
            self.reset()
            return self.chat(message, max_new_tokens=max_new_tokens)

        stacked = self.stack_chunks(chunks)
        n = len(chunks)
        # Re-establish a clean patched state (mirrors restore()): reset un-patches,
        # patch_lora_forward installs fresh lora_forward partials, then bind the stack.
        self.model.reset()
        self.model.patch_lora_forward()
        self.model.generated_loras = stacked

        chat = [{"role": "user", "content": message}]
        input_ids = self.tokenizer.apply_chat_template(
            chat, add_special_tokens=False, return_attention_mask=False,
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        kw = {}
        if scalers is not None:
            kw["scalers"] = torch.as_tensor(scalers, device=self.model.device, dtype=torch.float32)
        out = self.model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens,
            n_ctx_chunks=torch.tensor([n], device=self.model.device), **kw,
        )
        completion = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()

    # --- generation -------------------------------------------------------
    @torch.inference_mode()
    def chat(
        self,
        message: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        chat = [{"role": "user", "content": message}]
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
        out = self.model.generate(input_ids=input_ids, **gen_kwargs)
        completion = out[0][input_ids.shape[1] :]  # drop the echoed prompt
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()
