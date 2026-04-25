"""Tests for config loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import load_config


class ConfigTestCase(unittest.TestCase):
    """Test config loader behavior."""

    def test_load_config(self) -> None:
        payload = {
            "mongodb": {
                "uri": "mongodb://localhost:27017",
                "database": "db",
                "collection": "col",
                "text_fields": ["title", "content"],
                "id_field": "_id",
            },
            "providers": {
                "chat": {
                    "provider": "openai",
                    "base_url": "http://example.com/v1",
                    "api_key": "test",
                    "model": "chat-model",
                },
                "embedding": {
                    "provider": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "api_key": "",
                    "model": "nomic-embed-text",
                },
            },
            "retrieval": {"top_k": 5},
            "storage": {
                "index_path": "data/index.faiss",
                "metadata_path": "data/metadata.jsonl",
            },
            "server": {"host": "127.0.0.1", "port": 18000, "debug": True},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(path))
        self.assertEqual(cfg.retrieval.top_k, 5)
        self.assertEqual(cfg.mongodb.text_fields, ["title", "content"])
        self.assertEqual(cfg.chat.provider, "openai")
        self.assertEqual(cfg.embedding.provider, "ollama")
        self.assertTrue(cfg.server.debug)


if __name__ == "__main__":
    unittest.main()
