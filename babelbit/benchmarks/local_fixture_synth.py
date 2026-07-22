"""Local (no DeepInfra) helpers for API-challenge fixture synthesis."""

from __future__ import annotations

import io
import re
import subprocess
import wave
from typing import Any

import numpy as np

SOURCE_LANGUAGE_LABELS = {
    "fr": "French",
    "de": "German",
    "en": "English",
}

DERIVED_FROM_LOCAL = "results.babelbit.ai/dialogue-scores+local-chatterbox-tts"
DERIVED_FROM_HAND_EDGE_TTS = "hand-translations+edge-tts"

# Microsoft neural voices via edge-tts (clean FR/DE, low artifact vs Chatterbox).
DEFAULT_EDGE_VOICES = {
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "en": "en-US-JennyNeural",
}
DEFAULT_EDGE_RATE = "-10%"
DEFAULT_MAX_INTERNAL_SILENCE_SEC = 0.12
DEFAULT_MAX_EDGE_SILENCE_SEC = 0.15
DEFAULT_SILENCE_ABS_THRESH = 500.0  # int16 RMS per analysis frame


def challenge_uid_for_locale(base_challenge_uid: str, *, source_language: str) -> str:
    """Keep real UID for FR; suffix other locales so benches can select by language."""
    lang = source_language.strip().lower()
    if lang in {"", "fr"}:
        return base_challenge_uid
    return f"{base_challenge_uid}-{lang}"


def edge_voice_for_locale(source_language: str) -> str:
    """Return default edge-tts neural voice for a locale."""
    lang = source_language.strip().lower()
    voice = DEFAULT_EDGE_VOICES.get(lang)
    if voice is None:
        raise ValueError(f"No default edge-tts voice for locale={source_language!r}")
    return voice


def mp3_bytes_to_wav_bytes(
    mp3_bytes: bytes,
    *,
    target_rate_hz: int,
    ffmpeg_bin: str = "ffmpeg",
) -> bytes:
    """Decode MP3 to mono 16-bit PCM WAV at target_rate_hz via ffmpeg.

    Uses raw s16le on the pipe (not WAV) because ffmpeg cannot rewind a pipe to
    patch RIFF sizes, which leaves wave.open() with nframes≈2**31-1.
    """
    if not mp3_bytes:
        raise ValueError("Empty MP3 payload")
    completed = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(int(target_rate_hz)),
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        err = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg MP3→PCM failed (code={completed.returncode}): {err}")
    pcm = completed.stdout
    if len(pcm) % 2 != 0:
        raise RuntimeError("ffmpeg returned odd PCM byte length")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(target_rate_hz))
        wav.writeframes(pcm)
    return out.getvalue()


def squash_internal_silence_pcm(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    max_internal_silence_sec: float = DEFAULT_MAX_INTERNAL_SILENCE_SEC,
    max_edge_silence_sec: float = DEFAULT_MAX_EDGE_SILENCE_SEC,
    silence_abs_thresh: float = DEFAULT_SILENCE_ABS_THRESH,
    frame_sec: float = 0.02,
) -> np.ndarray:
    """Compress long quiet runs so TTS sentence pauses do not fake utterance EOS.

    Keeps up to max_internal_silence_sec of silence between speech; trims long
    leading/trailing quiet to max_edge_silence_sec.
    """
    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
    if pcm.size == 0:
        return pcm
    hop = max(1, int(round(float(sample_rate_hz) * float(frame_sec))))
    max_internal = max(1, int(round(float(max_internal_silence_sec) * float(sample_rate_hz))))
    max_edge = max(0, int(round(float(max_edge_silence_sec) * float(sample_rate_hz))))
    thresh = float(silence_abs_thresh)

    # Frame-level silence mask
    n_frames = max(1, (pcm.size + hop - 1) // hop)
    silent_frames = np.zeros(n_frames, dtype=bool)
    for index in range(n_frames):
        start = index * hop
        chunk = pcm[start : start + hop]
        if chunk.size == 0:
            silent_frames[index] = True
            continue
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
        silent_frames[index] = rms < thresh

    kept: list[np.ndarray] = []
    index = 0
    while index < n_frames:
        if not silent_frames[index]:
            start = index * hop
            end = min(pcm.size, start + hop)
            kept.append(pcm[start:end])
            index += 1
            continue
        run_start = index
        while index < n_frames and silent_frames[index]:
            index += 1
        run_samples = min(pcm.size, index * hop) - run_start * hop
        is_leading = run_start == 0
        is_trailing = index >= n_frames
        if is_leading or is_trailing:
            keep_n = min(run_samples, max_edge)
        else:
            keep_n = min(run_samples, max_internal)
        if keep_n > 0:
            start = run_start * hop
            kept.append(pcm[start : start + keep_n])

    if not kept:
        return pcm[: min(pcm.size, max_edge)]
    return np.concatenate(kept).astype(np.int16, copy=False)


def squash_silence_in_wav_bytes(
    wav_bytes: bytes,
    *,
    max_internal_silence_sec: float = DEFAULT_MAX_INTERNAL_SILENCE_SEC,
    max_edge_silence_sec: float = DEFAULT_MAX_EDGE_SILENCE_SEC,
    silence_abs_thresh: float = DEFAULT_SILENCE_ABS_THRESH,
) -> bytes:
    """Apply squash_internal_silence_pcm to a mono 16-bit WAV payload."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit WAV")
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    squashed = squash_internal_silence_pcm(
        pcm,
        sample_rate_hz=rate,
        max_internal_silence_sec=max_internal_silence_sec,
        max_edge_silence_sec=max_edge_silence_sec,
        silence_abs_thresh=silence_abs_thresh,
    )
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(squashed.tobytes())
    return out.getvalue()


def build_translate_system_prompt(*, source_language: str, target_language: str = "en") -> str:
    source_label = SOURCE_LANGUAGE_LABELS.get(source_language, source_language)
    target_label = SOURCE_LANGUAGE_LABELS.get(target_language, target_language)
    return (
        f"Translate the {target_label} sentence into natural spoken {source_label}. "
        "Return only the translated sentence."
    )


def parse_translation_response(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    # Drop common chat wrappers / thinking crumbs.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = lines[-1]
    for prefix in ("translation:", "translated:", "output:"):
        if candidate.lower().startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
    return candidate.strip().strip('"').strip("'").strip()


def expand_digits_for_tts(text: str) -> str:
    """Leave digits as spoken-friendly tokens; keep simple and deterministic."""
    return re.sub(r"\s+", " ", str(text or "").strip())


def split_clauses_for_tts(text: str, *, max_words: int = 10) -> list[str]:
    cleaned = expand_digits_for_tts(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?;:])\s+", cleaned)
    clauses: list[str] = []
    for piece in pieces:
        words = piece.split()
        if not words:
            continue
        if len(words) <= max_words:
            clauses.append(" ".join(words))
            continue
        for start in range(0, len(words), max_words):
            clauses.append(" ".join(words[start : start + max_words]))
    return clauses or [cleaned]


def float_audio_to_wav_bytes(audio: np.ndarray, *, sample_rate_hz: int) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype(np.int16).tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate_hz))
        wav.writeframes(pcm)
    return out.getvalue()


def concat_wav_bytes(parts: list[bytes], *, pause_sec: float = 0.25) -> bytes:
    if not parts:
        raise ValueError("No WAV parts to concatenate")
    pcm_chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    for part in parts:
        with wave.open(io.BytesIO(part), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("Expected mono 16-bit WAV parts")
            rate = wav.getframerate()
            if sample_rate is None:
                sample_rate = rate
            elif rate != sample_rate:
                raise ValueError(f"Sample rate mismatch: {rate} != {sample_rate}")
            frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            pcm_chunks.append(frames)
    assert sample_rate is not None
    if pause_sec > 0 and len(pcm_chunks) > 1:
        silence = np.zeros(max(1, int(round(pause_sec * sample_rate))), dtype=np.int16)
        interleaved: list[np.ndarray] = []
        for index, chunk in enumerate(pcm_chunks):
            if index:
                interleaved.append(silence)
            interleaved.append(chunk)
        pcm = np.concatenate(interleaved)
    else:
        pcm = np.concatenate(pcm_chunks)
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


def wav_duration_sec_from_bytes(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def strip_private_entry_fields(entry: dict[str, Any]) -> dict[str, Any]:
    copied = dict(entry)
    copied.pop("_source_wav_bytes", None)
    return copied
