"""Microsoft Edge neural TTS helpers for fixture synthesis."""

from __future__ import annotations

import asyncio

import edge_tts

from babelbit.benchmarks.local_fixture_synth import (
    DEFAULT_EDGE_RATE,
    expand_digits_for_tts,
)


async def synthesize_edge_tts_mp3(
    text: str,
    *,
    voice: str,
    rate: str = DEFAULT_EDGE_RATE,
) -> bytes:
    """Call Microsoft Edge neural TTS; return MP3 bytes."""
    cleaned = expand_digits_for_tts(text)
    if not cleaned:
        raise ValueError("Cannot synthesize empty text with edge-tts")
    communicate = edge_tts.Communicate(cleaned, voice, rate=rate)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            data = chunk.get("data")
            if isinstance(data, (bytes, bytearray)):
                chunks.append(bytes(data))
    if not chunks:
        raise RuntimeError(f"edge-tts returned no audio for voice={voice!r}")
    return b"".join(chunks)


def synthesize_edge_tts_mp3_sync(
    text: str,
    *,
    voice: str,
    rate: str = DEFAULT_EDGE_RATE,
) -> bytes:
    """Sync wrapper around synthesize_edge_tts_mp3."""
    return asyncio.run(synthesize_edge_tts_mp3(text, voice=voice, rate=rate))
