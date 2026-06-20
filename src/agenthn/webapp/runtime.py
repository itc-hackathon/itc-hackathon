"""Shared model runtime for the web app.

The D2L model is heavy (a few GB of VRAM) and NOT thread-safe — its active LoRA
adapter is mutated on every call. Both demos (personalization + memory) therefore
share ONE loaded model behind ONE lock, so they can't stomp on each other's
adapter state. Load is lazy so the server starts instantly.
"""

from __future__ import annotations

import threading

_model = None
_load_lock = threading.Lock()
# Held for the duration of any model call (generation mutates the active adapter).
MODEL_LOCK = threading.Lock()


def get_model():
    """Return the shared, lazily-loaded D2LModel."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from ..core.model import D2LModel

                _model = D2LModel.load()
    return _model


def model_loaded() -> bool:
    return _model is not None
