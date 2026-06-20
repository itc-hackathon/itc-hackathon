"""Memory track (Bryan, Nikash): long-horizon memory via stacked LoRA adapters.

Every K steps, convert oldest turns into an adapter and rank-concat with the
running memory adapter (see ctx_to_lora.modeling.lora_merger.combine_lora), then
evict those turns from context. At query time, retrieve the top-k relevant
segment adapters and compose only those (blind summation of all segments
interferes destructively as memory grows).
"""

from .baselines import MarkdownMemory, VanillaContextMemory
from .live import MemoryArena
from .nap_memory import MemoryStats, NapLoRAMemory, Segment, Turn
from .retriever import TfidfRetriever

__all__ = [
    "NapLoRAMemory",
    "MemoryStats",
    "Segment",
    "Turn",
    "TfidfRetriever",
    "VanillaContextMemory",
    "MarkdownMemory",
    "MemoryArena",
]
