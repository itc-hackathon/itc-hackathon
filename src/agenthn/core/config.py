"""Central paths and runtime config for AgentHN.

Override any of these via environment variables (same name) without editing code.
"""

import os
from pathlib import Path

# Where the doc-to-lora repo + its downloaded checkpoints live on this box.
# Auto-detect across the boxes the team uses (Prime Intellect = /home/ubuntu,
# this box = /root); override with D2L_REPO to force a path.
def _detect_d2l_repo() -> Path:
    env = os.environ.get("D2L_REPO")
    if env:
        return Path(env)
    for cand in ("/home/ubuntu/doc-to-lora", "/root/doc-to-lora"):
        if Path(cand).exists():
            return Path(cand)
    return Path("/home/ubuntu/doc-to-lora")


D2L_REPO = _detect_d2l_repo()

# The gemma_demo (gemma-2-2b-it, checkpoint-80000) D2L hypernetwork checkpoint.
CHECKPOINT = Path(
    os.environ.get(
        "AGENTHN_CHECKPOINT",
        str(D2L_REPO / "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"),
    )
)

# Base model the checkpoint modulates (gated on HF — must be logged in).
BASE_MODEL = os.environ.get("AGENTHN_BASE_MODEL", "google/gemma-2-2b-it")

DEVICE = os.environ.get("AGENTHN_DEVICE", "cuda")

# Repo root is three levels up: core/ -> agenthn/ -> src/ -> <repo root>.
REPO_ROOT = Path(__file__).parents[3]

# Where per-user profile docs and cached adapters get written at runtime.
DATA_DIR = Path(os.environ.get("AGENTHN_DATA_DIR", str(REPO_ROOT / "data")))
PROFILES_DIR = DATA_DIR / "profiles"
ADAPTERS_DIR = DATA_DIR / "adapters"
