# wd14-tagger-server

Standalone FastAPI service wrapping the [wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) image-tagging model (WD14). Given an uploaded image, returns predicted general/character tags above configurable confidence thresholds. Run standalone (PM2/uvicorn) or via Docker; managed with `pdm`. Third-party-derived project (see `LICENSE`, `NOTICE.md`) — avoid restructuring beyond what's needed for integration.

## Structure

- `main.py` — FastAPI app entrypoint (`/upload` endpoint).
- `app/settings.py` — service configuration/env loading.
- `app/values.py` — available model list/definitions.
- `app/infer/` — inference pipeline: `load.py` (model loading), `setup.py` (initialization), `error.py` (error types).
- `sdk.py` — Python client SDK for calling this service from other Python code (used by `python/` tooling or similar consumers).
- `tests/` — pytest suite (`test_app.py`) plus fixture image(s).
- `pm2.json` — PM2 process manager config for production runtime.
- `Dockerfile` — container build.

## Key conventions

- Config via `.env` (copy from `.env.exp`).
- Install/run with `pdm install && pdm run python main.py`, not raw pip/venv.
