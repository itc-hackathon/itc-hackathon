"""Memory track: long-horizon agent memory stored in LoRA blocks, not context.

Two complementary approaches live here:

NapLoRA (live demo, gemma_demo checkpoint): every K turns the oldest segment is
internalized into a LoRA adapter and evicted from context; at query time we
RETRIEVE the top-k relevant segment adapters and compose only those. Retrieval
sidesteps the destructive summation you get from rank-concatenating every segment
on the single-chunk gemma_demo checkpoint.
  NapLoRAMemory / MarkdownMemory / VanillaContextMemory / MemoryArena

WeightMemory (chunk-trained checkpoint): an agent journals observations; each is
internalized and rank-concatenated onto a running memory adapter (combine_lora),
then evicted. Requires config.CHUNK_CHECKPOINT — the gemma_demo checkpoint is
single-chunk only and its combine_lora produces garbage (see the track README).
  WeightMemory / generate_session
"""

from .baselines import MarkdownMemory, TextRAGMemory, VanillaContextMemory
from .live import MemoryArena
from .memory_store import WeightMemory, encode_doc_chunks
from .nap_memory import MemoryStats, NapLoRAMemory, Segment, Turn
from .retriever import TfidfRetriever
from .tasks import MemoryEntry, Probe, generate_session

__all__ = [
    # NapLoRA (retrieval-routed, gemma_demo)
    "NapLoRAMemory",
    "MemoryStats",
    "Segment",
    "Turn",
    "TfidfRetriever",
    "VanillaContextMemory",
    "MarkdownMemory",
    "TextRAGMemory",
    "MemoryArena",
    # WeightMemory (rank-concatenation, chunk checkpoint)
    "WeightMemory",
    "encode_doc_chunks",
    "MemoryEntry",
    "Probe",
    "generate_session",
]
