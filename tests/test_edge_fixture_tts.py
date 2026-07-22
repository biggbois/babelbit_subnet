"""Unit tests for edge-tts fixture helpers."""

from __future__ import annotations

import pytest

from babelbit.benchmarks import edge_fixture_tts as module


@pytest.mark.asyncio
async def test_synthesize_edge_tts_mp3_collects_audio_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeCommunicate:
        def __init__(self, text: str, voice: str, rate: str) -> None:
            self.text = text
            self.voice = voice
            self.rate = rate

        async def stream(self):
            yield {"type": "audio", "data": b"abc"}
            yield {"type": "WordBoundary", "offset": 0}
            yield {"type": "audio", "data": b"def"}

    monkeypatch.setattr(module.edge_tts, "Communicate", _FakeCommunicate)
    out = await module.synthesize_edge_tts_mp3(
        "  Bonjour  ",
        voice="fr-FR-DeniseNeural",
        rate="-10%",
    )
    assert out == b"abcdef"


@pytest.mark.asyncio
async def test_synthesize_edge_tts_mp3_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        await module.synthesize_edge_tts_mp3("   ", voice="fr-FR-DeniseNeural")
