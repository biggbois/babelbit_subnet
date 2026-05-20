from __future__ import annotations

import io
import json
import os
import tempfile
from time import perf_counter
import wave
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from babelbit.scoring.scoring_common import WordTS, normalize_words
from babelbit.utils.hf_runtime import ensure_hf_transfer_available

STT_MODEL_TABLE: Dict[str, Tuple[str, str, str]] = {
    "faster-whisper-tiny": ("faster-whisper", "tiny", "float16"),
    "faster-whisper-base": ("faster-whisper", "base", "float16"),
    "faster-whisper-small": ("faster-whisper", "small", "float16"),
    "faster-whisper-medium": ("faster-whisper", "medium", "float16"),
    "faster-whisper-large-v3": ("faster-whisper", "large-v3", "float16"),
    "faster-whisper-large-v3-turbo": ("faster-whisper", "large-v3-turbo", "float16"),
    "faster-whisper-large-v3-turbo-int8": ("faster-whisper", "large-v3-turbo", "int8"),
    "faster-whisper-distil-large-v3": (
        "faster-whisper",
        "distil-large-v3",
        "float16",
    ),
}

_CPU_COMPUTE_TYPE = {"float16": "int8"}
_STT_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}
_FW_MODEL_CACHE: Dict[str, Any] = {}
_WHISPER_SAMPLE_RATE_HZ = 16000
logger = getLogger(__name__)


def validate_stt_model(model_name: str) -> None:
    if model_name not in STT_MODEL_TABLE:
        supported = ", ".join(sorted(STT_MODEL_TABLE.keys()))
        raise ValueError(
            f"Unsupported STT model: {model_name!r}. Supported models: {supported}"
        )


def _load_stt_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    cache_key = str(cache_path.resolve())
    if cache_key in _STT_CACHE:
        return _STT_CACHE[cache_key]

    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            wav_hash = row.get("wav_sha256")
            if not isinstance(wav_hash, str):
                continue
            cache[wav_hash] = {
                "text": str(row.get("text", "")),
                "words": normalize_words(row.get("words")),
            }

    _STT_CACHE[cache_key] = cache
    return cache


def _append_stt_cache(cache_path: Path, row: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resample_audio(
    audio: np.ndarray, source_rate_hz: int, target_rate_hz: int
) -> np.ndarray:
    if source_rate_hz <= 0:
        raise ValueError(f"Invalid source sample rate: {source_rate_hz}")
    if source_rate_hz == target_rate_hz:
        return np.ascontiguousarray(audio.astype(np.float32, copy=False))
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    duration_sec = float(audio.shape[0]) / float(source_rate_hz)
    target_length = max(1, int(round(duration_sec * float(target_rate_hz))))
    source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio)
    return np.ascontiguousarray(resampled.astype(np.float32, copy=False))


def _wav_bytes_to_whisper_input(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate_hz = wf.getframerate()
        frames = wf.readframes(n_frames)
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

    if n_frames <= 0:
        raise ValueError("WAV payload had no frames")

    dtype_map: Dict[int, type] = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)

    if sample_width == 2:
        audio /= 32768.0
    elif sample_width == 4:
        audio /= 2147483648.0
    elif sample_width == 1:
        audio = (audio - 128.0) / 128.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    audio = np.clip(audio, -1.0, 1.0)
    return _resample_audio(audio, sample_rate_hz, _WHISPER_SAMPLE_RATE_HZ)


def _get_fw_model(model_size: str, compute_type: str, device: str):
    if device == "cpu" and compute_type in _CPU_COMPUTE_TYPE:
        compute_type = _CPU_COMPUTE_TYPE[compute_type]
    cache_key = f"{model_size}:{device}:{compute_type}"
    if cache_key not in _FW_MODEL_CACHE:
        ensure_hf_transfer_available()
        from faster_whisper import WhisperModel

        _FW_MODEL_CACHE[cache_key] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _FW_MODEL_CACHE[cache_key]


def _stt_faster_whisper(
    audio_input: Union[Path, np.ndarray],
    *,
    model_id: str,
    compute_type: str,
    device: str,
    language: str,
) -> Tuple[str, List[WordTS]]:
    model = _get_fw_model(model_id, compute_type, device)

    transcribe_input: Union[str, np.ndarray] = (
        audio_input if isinstance(audio_input, np.ndarray) else str(audio_input)
    )

    segments, _info = model.transcribe(
        transcribe_input, language=language, word_timestamps=True
    )

    words: List[WordTS] = []
    text_parts: List[str] = []
    for segment in segments:
        text_parts.append(str(segment.text).strip())
        if not getattr(segment, "words", None):
            continue
        for word in segment.words:
            words.append(
                {
                    "word": str(word.word).strip(),
                    "start": float(word.start),
                    "end": float(word.end),
                }
            )
    return " ".join(part for part in text_parts if part).strip(), words


def transcribe_wav(
    wav_path: Path,
    *,
    stt_model: str,
    language: str,
    device: str,
) -> Tuple[str, List[WordTS]]:
    validate_stt_model(stt_model)
    backend, model_id, compute_type = STT_MODEL_TABLE[stt_model]
    if backend == "faster-whisper":
        return _stt_faster_whisper(
            wav_path,
            model_id=model_id,
            compute_type=compute_type,
            device=device,
            language=language,
        )
    raise ValueError(f"Unknown STT backend: {backend}")


def transcribe_wav_bytes(
    wav_bytes: bytes,
    *,
    wav_hash: str,
    stt_model: str,
    language: str,
    device: str,
    stt_cache_path: Path,
) -> Tuple[str, List[WordTS]]:
    cache = _load_stt_cache(stt_cache_path)
    if wav_hash in cache:
        cached = cache[wav_hash]
        logger.info(
            "STT cache hit: wav_hash=%s model=%s device=%s",
            wav_hash,
            stt_model,
            device,
        )
        return str(cached.get("text", "")), normalize_words(cached.get("words"))

    validate_stt_model(stt_model)
    backend, model_id, compute_type = STT_MODEL_TABLE[stt_model]

    transcribe_started_at = perf_counter()
    audio_duration_sec = None
    if backend == "faster-whisper":
        decode_started_at = perf_counter()
        audio_array = _wav_bytes_to_whisper_input(wav_bytes)
        decode_sec = perf_counter() - decode_started_at
        audio_duration_sec = float(audio_array.shape[0]) / float(
            _WHISPER_SAMPLE_RATE_HZ
        )
        stt_started_at = perf_counter()
        transcript_text, transcript_words = _stt_faster_whisper(
            audio_array,
            model_id=model_id,
            compute_type=compute_type,
            device=device,
            language=language,
        )
        stt_sec = perf_counter() - stt_started_at
        logger.info(
            "STT item profile: wav_hash=%s model=%s backend=%s device=%s "
            "audio_sec=%.3f decode_sec=%.3f stt_sec=%.3f text_chars=%d words=%d",
            wav_hash,
            stt_model,
            backend,
            device,
            audio_duration_sec,
            decode_sec,
            stt_sec,
            len(transcript_text),
            len(transcript_words),
        )
    elif backend == "openai":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(wav_bytes)
            temp_path = Path(temp_file.name)

        try:
            transcript_text, transcript_words = _stt_openai(
                temp_path, model_id=model_id, language=language
            )
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        raise ValueError(f"Unknown STT backend: {backend}")

    cache[wav_hash] = {"text": transcript_text, "words": transcript_words}
    _append_stt_cache(
        stt_cache_path,
        {"wav_sha256": wav_hash, "text": transcript_text, "words": transcript_words},
    )
    logger.info(
        "STT item complete: wav_hash=%s model=%s device=%s audio_sec=%s total_sec=%.3f",
        wav_hash,
        stt_model,
        device,
        f"{audio_duration_sec:.3f}" if audio_duration_sec is not None else "unknown",
        perf_counter() - transcribe_started_at,
    )
    return transcript_text, transcript_words


def transcribe_wav_bytes_batch(
    items: List[Dict[str, Any]],
    *,
    stt_model: str,
    language: str,
    device: str,
    stt_cache_path: Path,
) -> List[Dict[str, Any]]:
    batch_started_at = perf_counter()
    results: List[Dict[str, Any]] = []
    cache_errors = 0
    for item in items:
        try:
            text, words = transcribe_wav_bytes(
                item["wav_bytes"],
                wav_hash=item["wav_hash"],
                stt_model=stt_model,
                language=language,
                device=device,
                stt_cache_path=stt_cache_path,
            )
            results.append({"text": text, "words": words, "error": None})
        except Exception as exc:
            cache_errors += 1
            error_text = f"{type(exc).__name__}:{exc}"
            logger.warning(
                "Batch STT failed for wav_hash=%s model=%s device=%s: %s",
                item.get("wav_hash", "unknown"),
                stt_model,
                device,
                error_text,
            )
            results.append({"text": "", "words": [], "error": error_text})
    logger.info(
        "STT batch profile: items=%d errors=%d model=%s device=%s total_sec=%.3f",
        len(items),
        cache_errors,
        stt_model,
        device,
        perf_counter() - batch_started_at,
    )
    return results


__all__ = [
    "STT_MODEL_TABLE",
    "transcribe_wav",
    "transcribe_wav_bytes",
    "transcribe_wav_bytes_batch",
    "validate_stt_model",
]
