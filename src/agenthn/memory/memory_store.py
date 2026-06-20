"""WeightMemory: long-horizon agent memory stored in LoRA blocks, not context.

The agent journals observations over a long session. Instead of keeping that
text in the prompt (linear token cost, eventually overflows the window), each
memory is internalized into a single-chunk LoRA adapter and rank-concatenated
onto a running stack (ctx_to_lora.combine_lora). At recall time the prompt holds
only the question — the facts live in the weights, so per-query context cost is
~constant regardless of how much the agent has remembered.

  remember(text)     -> encode one observation into a chunk adapter, append to the stack
  recall(question)   -> generate with the concatenated stack (empty-context recall)
  context_prompt(q)  -> the equivalent in-context prompt (for the cost/accuracy baseline)

Requires the chunk-trained checkpoint (config.CHUNK_CHECKPOINT); gemma_demo's
single-chunk-only training makes combine_lora produce garbage.
"""

from __future__ import annotations

import torch

from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text

from ..core.config import MAX_CHUNK_LEN, MAX_CHUNKS
from ..core.model import D2LModel


def encode_doc_chunks(model: D2LModel, texts: list[str],
                      max_chunk_len: int = MAX_CHUNK_LEN) -> list[dict]:
    """Pack the memory log into ONE document, chunk it the way the model was
    trained to (split_too_long_ctx adds continuation affixes), and return one
    single-chunk adapter per chunk — ready for chat_memory to rank-concatenate.

    This is the in-distribution way to get K composable chunk-adapters: the chunk
    count grows only as the log exceeds max_chunk_len (~512 tok), so a handful of
    short observations live in ONE chunk and concatenation kicks in as memory
    actually grows. Re-encodes the whole log (cheap: one hypernet forward).
    """
    doc = "\n".join(texts)
    ctx_ids_full = tokenize_ctx_text({"context": [doc]}, model.ctx_tokenizer)["ctx_ids"][0]
    split = split_too_long_ctx(
        {"ctx_ids": ctx_ids_full}, model.model.base_model.name_or_path,
        num_chunk_probs=None, max_chunk_len=max_chunk_len, min_chunk_len=1,
        max_num_split=None, is_train=False,
    )
    chunk_id_lists = split["ctx_ids"]
    ids = [torch.tensor(c) for c in chunk_id_lists]
    ctx_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0).to(model.model.device)
    ctx_attn = torch.nn.utils.rnn.pad_sequence(
        [torch.ones_like(x) for x in ids], batch_first=True, padding_value=0).to(model.model.device)
    with torch.inference_mode():
        loras, _ = model.model.generate_weights(ctx_ids, ctx_attn)
    # Slice the [K, ...] adapter into K single-chunk dicts.
    k = ctx_ids.shape[0]
    return [
        {m: {"A": v["A"][i:i + 1].clone(), "B": v["B"][i:i + 1].clone()} for m, v in loras.items()}
        for i in range(k)
    ]


class WeightMemory:
    """Long-horizon memory in LoRA blocks.

    mode="doc" (default): pack the log into a chunked document — best recall,
      re-encodes the whole log on each write (one forward pass).
    mode="incremental": cache one adapter per observation and stack them — fully
      incremental, but recall caps lower (stacking standalone facts is harder for
      the hypernet than chunks of one doc).
    """

    def __init__(self, model: D2LModel, max_chunks: int = MAX_CHUNKS, mode: str = "doc"):
        self.model = model
        self.max_chunks = max_chunks
        self.mode = mode
        self.texts: list[str] = []     # human-readable memory log (never enters the prompt)
        self._inc_chunks: list[dict] = []   # incremental mode: one adapter per memory
        self._doc_chunks: list[dict] | None = None  # doc mode: cache, invalidated on write

    @property
    def chunks(self) -> list[dict]:
        """The chunk-adapters to rank-concatenate at recall time."""
        if self.mode == "incremental":
            return self._inc_chunks
        if self._doc_chunks is None:
            self._doc_chunks = encode_doc_chunks(self.model, self.texts) if self.texts else []
        return self._doc_chunks

    # --- writing to weight memory ----------------------------------------
    def remember(self, text: str) -> None:
        """Add one observation to memory (and its weight encoding)."""
        self.texts.append(text)
        if self.mode == "incremental":
            self._inc_chunks.append(self.model.encode_chunk(text))
        else:
            self._doc_chunks = None  # invalidate; rebuilt lazily on next recall
        # In incremental mode chunk count == memory count, so we can warn cheaply
        # when we exceed the trained chunk count. In doc mode the count is only
        # known after the (lazy) re-encode, so we don't force a build here.
        if self.mode == "incremental" and len(self._inc_chunks) > self.max_chunks:
            import warnings
            warnings.warn(
                f"WeightMemory holds {len(self._inc_chunks)} chunks > trained max "
                f"{self.max_chunks}; recall is out-of-distribution. Use mode='doc' "
                f"so short observations pack into fewer chunks.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.texts)

    def clear(self) -> None:
        self.texts.clear()
        self._inc_chunks.clear()
        self._doc_chunks = None

    # --- reading from weight memory --------------------------------------
    def recall(self, question: str, scalers=None, max_new_tokens: int = 64) -> str:
        """Answer from weight memory with an empty context (just the question)."""
        return self.model.chat_memory(
            question, self.chunks, scalers=scalers, max_new_tokens=max_new_tokens
        )

    # --- baselines (for the cost/accuracy comparison) --------------------
    def context_prompt(self, question: str) -> str:
        """The in-context equivalent: dump every memory into the prompt."""
        log = "\n".join(f"- {t}" for t in self.texts)
        return (
            "You are an agent. Here is your memory log:\n"
            f"{log}\n\nUsing the log above, answer concisely.\n{question}"
        )
