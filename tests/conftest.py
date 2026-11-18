import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests never touch the network and need no keys."""

    def guard(*a, **k):
        raise RuntimeError("network access attempted in a test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setenv("URN_EXCHANGE_BACKOFF_SCALE", "0")
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
