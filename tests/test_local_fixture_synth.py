from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from babelbit.benchmarks import local_fixture_synth as module


def test_challenge_uid_for_locale_keeps_fr_base() -> None:
    base = "challenge-1784535696-1b911b82"
    assert module.challenge_uid_for_locale(base, source_language="fr") == base
    assert module.challenge_uid_for_locale(base, source_language="de") == f"{base}-de"


def test_parse_translation_response_strips_wrappers() -> None:
    raw = "Translation: Bonjour le monde\n"
    assert module.parse_translation_response(raw) == "Bonjour le monde"


def test_build_translate_system_prompt_mentions_languages() -> None:
    prompt = module.build_translate_system_prompt(source_language="de")
    assert "German" in prompt
    assert "English" in prompt


def test_split_clauses_for_tts_respects_max_words() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve"
    clauses = module.split_clauses_for_tts(text, max_words=5)
    assert clauses == [
        "one two three four five",
        "six seven eight nine ten",
        "eleven twelve",
    ]


def test_float_audio_to_wav_and_concat_roundtrip() -> None:
    audio = np.linspace(-0.2, 0.2, 2400, dtype=np.float32)
    part_a = module.float_audio_to_wav_bytes(audio, sample_rate_hz=24_000)
    part_b = module.float_audio_to_wav_bytes(audio * 0.5, sample_rate_hz=24_000)
    merged = module.concat_wav_bytes([part_a, part_b], pause_sec=0.1)
    duration = module.wav_duration_sec_from_bytes(merged)
    assert duration == pytest.approx(0.3, abs=1e-3)
    with wave.open(io.BytesIO(merged), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnchannels() == 1
