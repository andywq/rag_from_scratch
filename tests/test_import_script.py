"""Tests for MongoDB import helpers."""

from __future__ import annotations

import unittest

from scripts.import_from_mongodb import build_text, get_values_by_path


class ImportScriptTestCase(unittest.TestCase):
    """Test helper behavior in import script."""

    def test_build_text_ignores_missing_and_empty_fields(self) -> None:
        document = {"title": "Title", "content": "", "extra": "ignored"}
        text = build_text(document, ["title", "content", "body"])
        self.assertEqual(text, "Title")

    def test_get_values_by_path_supports_nested_and_list_wildcard(self) -> None:
        document = {
            "fields": {
                "summary": "Issue summary",
                "comment": {
                    "comments": [
                        {"body": "First comment"},
                        {"body": "Second comment"},
                    ]
                },
            }
        }
        summary = get_values_by_path(document, "fields.summary")
        comments = get_values_by_path(document, "fields.comment.comments[].body")
        self.assertEqual(summary, ["Issue summary"])
        self.assertEqual(comments, ["First comment", "Second comment"])

    def test_build_text_supports_dotted_paths(self) -> None:
        document = {
            "key": "ISSUE-1000",
            "fields": {
                "summary": "Collector optimization",
                "description": "Improve timeout handling.",
                "comment": {"comments": [{"body": "Need more logs"}]},
            },
        }
        text = build_text(
            document,
            [
                "key",
                "fields.summary",
                "fields.description",
                "fields.comment.comments[].body",
            ],
        )
        self.assertEqual(
            text,
            "ISSUE-1000\nCollector optimization\nImprove timeout handling.\nNeed more logs",
        )


if __name__ == "__main__":
    unittest.main()
