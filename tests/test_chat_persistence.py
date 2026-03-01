"""Tests for chat persistence (_save_chat / _load_chat)."""

import json
from pathlib import Path

import pytest

from athena.frontend.app import _save_chat, _load_chat


class TestSaveChat:
    """Tests for _save_chat()."""

    def test_saves_messages_to_file(self, tmp_path):
        chat_path = tmp_path / "chat.json"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        _save_chat(messages, chat_path)

        assert chat_path.exists()
        saved = json.loads(chat_path.read_text())
        assert len(saved) == 2
        assert saved[0]["role"] == "user"
        assert saved[1]["content"] == "Hi there!"

    def test_saves_empty_list(self, tmp_path):
        chat_path = tmp_path / "chat.json"
        _save_chat([], chat_path)

        saved = json.loads(chat_path.read_text())
        assert saved == []

    def test_overwrites_existing_file(self, tmp_path):
        chat_path = tmp_path / "chat.json"
        _save_chat([{"role": "user", "content": "First"}], chat_path)
        _save_chat([{"role": "user", "content": "Second"}], chat_path)

        saved = json.loads(chat_path.read_text())
        assert len(saved) == 1
        assert saved[0]["content"] == "Second"


class TestLoadChat:
    """Tests for _load_chat()."""

    def test_loads_saved_messages(self, tmp_path):
        chat_path = tmp_path / "chat.json"
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        chat_path.write_text(json.dumps(messages))

        loaded = _load_chat(chat_path)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "Hello"

    def test_returns_empty_for_missing_file(self, tmp_path):
        chat_path = tmp_path / "nonexistent.json"
        loaded = _load_chat(chat_path)
        assert loaded == []

    def test_returns_empty_for_invalid_json(self, tmp_path):
        chat_path = tmp_path / "bad.json"
        chat_path.write_text("not valid json{{{")

        loaded = _load_chat(chat_path)
        assert loaded == []

    def test_returns_empty_for_non_list_json(self, tmp_path):
        chat_path = tmp_path / "obj.json"
        chat_path.write_text('{"key": "value"}')

        loaded = _load_chat(chat_path)
        assert loaded == []

    def test_roundtrip(self, tmp_path):
        chat_path = tmp_path / "chat.json"
        messages = [
            {"role": "user", "content": "What is AAPL?", "source": "web"},
            {"role": "assistant", "content": "Apple Inc.", "_thinking": "looked it up"},
        ]
        _save_chat(messages, chat_path)
        loaded = _load_chat(chat_path)
        assert loaded == messages
