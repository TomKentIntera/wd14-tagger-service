# -*- coding: utf-8 -*-
"""Unit tests for model/CSV download helpers (no ONNX load)."""

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import patch

import httpx

SETUP_PATH = Path(__file__).resolve().parents[1] / "app" / "infer" / "setup.py"


def _load_setup():
    spec = importlib.util.spec_from_file_location("wd14_setup", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_create_part_path_can_be_unlinked(tmp_path):
    setup = _load_setup()
    part_path = setup._create_part_path(str(tmp_path), "wd-swinv2-tagger-v3.csv")

    path = Path(part_path)
    assert path.exists()
    path.unlink()
    assert not path.exists()


def test_httpx_stream_download_writes_file(tmp_path):
    setup = _load_setup()
    body = b"tag_id,name\n1,test\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    with patch.object(setup.httpx, "AsyncClient", side_effect=factory):
        asyncio.run(
            setup._httpx_stream_download(
                "wd-swinv2-tagger-v3.csv",
                "https://example.com/selected_tags.csv",
                str(tmp_path),
            )
        )

    dest = tmp_path / "wd-swinv2-tagger-v3.csv"
    assert dest.read_bytes() == body
    assert list(tmp_path.glob("*.part")) == []
