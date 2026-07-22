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


def test_edge_voice_for_locale_defaults() -> None:
    assert module.edge_voice_for_locale("fr") == "fr-FR-DeniseNeural"
    assert module.edge_voice_for_locale("de") == "de-DE-KatjaNeural"
    with pytest.raises(ValueError, match="No default edge-tts voice"):
        module.edge_voice_for_locale("xx")


def test_mp3_bytes_to_wav_bytes_invokes_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    pcm = np.linspace(-0.1, 0.1, 2400, dtype=np.float32)
    pcm_bytes = np.clip(np.round(pcm * 32767.0), -32768, 32767).astype(np.int16).tobytes()

    class _Completed:
        returncode = 0
        stdout = pcm_bytes
        stderr = b""

    def _fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(list(cmd))
        assert kwargs.get("input") == b"fake-mp3"
        return _Completed()

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    wav = module.mp3_bytes_to_wav_bytes(b"fake-mp3", target_rate_hz=24_000)
    assert calls and calls[0][0] == "ffmpeg"
    assert "-f" in calls[0] and "s16le" in calls[0]
    assert "24000" in calls[0]
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getframerate() == 24_000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == 2400
        assert handle.getnframes() / handle.getframerate() == pytest.approx(0.1)


def test_mp3_bytes_to_wav_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Empty MP3"):
        module.mp3_bytes_to_wav_bytes(b"", target_rate_hz=24_000)


def test_squash_internal_silence_shortens_long_gap() -> None:
    sr = 24_000
    speech = (np.linspace(-0.3, 0.3, sr, dtype=np.float32) * 10000).astype(np.int16)
    gap = np.zeros(sr, dtype=np.int16)  # 1.0s silence
    pcm = np.concatenate([speech, gap, speech])
    out = module.squash_internal_silence_pcm(
        pcm,
        sample_rate_hz=sr,
        max_internal_silence_sec=0.12,
        max_edge_silence_sec=0.05,
        silence_abs_thresh=500.0,
    )
    # Two 1s speech + <=0.12s internal (+ tiny edges)
    assert out.size < pcm.size
    assert out.size <= 2 * sr + int(0.12 * sr) + int(0.05 * sr) * 2
    assert out.size >= 2 * sr


def test_squash_silence_in_wav_bytes_roundtrip() -> None:
    sr = 24_000
    speech = (np.sin(np.linspace(0, 40, sr, dtype=np.float32)) * 8000).astype(np.int16)
    gap = np.zeros(int(0.8 * sr), dtype=np.int16)
    pcm = np.concatenate([speech, gap, speech])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())
    squashed = module.squash_silence_in_wav_bytes(buf.getvalue())
    dur = module.wav_duration_sec_from_bytes(squashed)
    assert dur < 2.8
    assert dur > 2.0
