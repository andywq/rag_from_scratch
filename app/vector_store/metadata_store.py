"""Metadata storage for vector records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetadataItem:
    """Metadata record mapped to one vector row."""

    doc_id: str
    text: str
    source: dict[str, object]


class MetadataStore:
    """Load and persist metadata records as JSONL."""

    @staticmethod
    def load(path: str) -> list[MetadataItem]:
        """Load metadata JSONL file."""
        items: list[MetadataItem] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            items.append(
                MetadataItem(
                    doc_id=str(payload["doc_id"]),
                    text=payload["text"],
                    source=payload.get("source", {}),
                )
            )
        return items

    @staticmethod
    def save(path: str, items: list[MetadataItem]) -> None:
        """Save metadata records to JSONL file."""
        output = []
        for item in items:
            output.append(
                json.dumps(
                    {
                        "doc_id": item.doc_id,
                        "text": item.text,
                        "source": item.source,
                    },
                    ensure_ascii=False,
                )
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(output), encoding="utf-8")
