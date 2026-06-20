"""Web app track (Eric): demo UI for the live runs.

A FastAPI app (``app.py``) serves the static single-page demo (``static/``) and
exposes the personalization hook over HTTP (``service.py``). The personalization
demo is live against Doc-to-LoRA; a mock service stands in when the GPU stack
isn't importable so the page runs anywhere.
"""
