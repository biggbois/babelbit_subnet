"""Fetch production challenge metadata and utterances for local benchmarks."""

from __future__ import annotations

import base64
import fnmatch
import io
import json
import os
import shutil
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

from babelbit.benchmarks.miner_test_data import (
    api_challenge_fixtures_dir,
    estimate_min_source_duration_sec,
    legacy_api_challenge_fixtures_dir,
    source_audio_sidecar_path,
    transcript_word_recall,
)
from babelbit.scoring.reference_metadata import resolve_audio_reference_metadata

RESULTS_API_BASE = "https://results.babelbit.ai/api"
DEFAULT_UTTERANCE_ENGINE_URL = "https://api.babelbit.ai"
BENCHMARK_SAMPLE_RATE_HZ = 24_000


class ApiChallengeError(RuntimeError):
    pass


def fetch_json(url: str, *, params: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_latest_challenge(*, status: str = "completed") -> dict[str, Any]:
    payload = fetch_json(f"{RESULTS_API_BASE}/challenges", params={"status": status, "limit": 5})
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ApiChallengeError(f"No challenges returned for status={status!r}")
    for item in items:
        if isinstance(item, dict) and item.get("latest_challenge"):
            return item
    first = items[0]
    if not isinstance(first, dict):
        raise ApiChallengeError("Unexpected challenge list payload")
    return first


def get_challenge_uid(challenge: dict[str, Any]) -> str:
    uid = str(challenge.get("main_challenge_uid") or challenge.get("challenge_uid") or "").strip()
    if not uid:
        raise ApiChallengeError("Challenge payload is missing main_challenge_uid")
    return uid


def fetch_challenge_detail(challenge_uid: str) -> dict[str, Any]:
    payload = fetch_json(f"{RESULTS_API_BASE}/challenges/{challenge_uid}")
    if not isinstance(payload, dict):
        raise ApiChallengeError(f"Invalid challenge detail payload for {challenge_uid}")
    return payload


def fetch_dialogue_summary(
    challenge_uid: str,
    *,
    stage: str = "main",
    limit: int = 5,
) -> list[dict[str, Any]]:
    payload = fetch_json(
        f"{RESULTS_API_BASE}/challenges/{challenge_uid}/dialogue-summary",
        params={"stage": stage, "limit": limit},
    )
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ApiChallengeError(f"Invalid dialogue summary for {challenge_uid}")
    return [item for item in items if isinstance(item, dict)]


def fetch_dialogue_scores(
    challenge_uid: str,
    *,
    stage: str = "main",
    limit: int = 50,
    miner_hotkey: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"stage": stage, "limit": limit}
    if miner_hotkey:
        params["miner_hotkey"] = miner_hotkey
    payload = fetch_json(
        f"{RESULTS_API_BASE}/challenges/{challenge_uid}/dialogue-scores",
        params=params,
    )
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ApiChallengeError(f"Invalid dialogue-scores payload for {challenge_uid}")
    return [item for item in items if isinstance(item, dict)]


def dedupe_dialogue_score_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("dialogue_uid", "")), str(item.get("utterance_number", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return sorted(
        unique,
        key=lambda item: (
            int(str(item.get("dialogue_uid") or 0)),
            int(str(item.get("utterance_number") or 0)),
        ),
    )


def parse_dialogue_score_steps(item: dict[str, Any]) -> dict[str, Any]:
    raw_steps = item.get("steps")
    if not isinstance(raw_steps, str) or not raw_steps.strip():
        return {}
    try:
        parsed = json.loads(raw_steps)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_dialogue_score_transcript(item: dict[str, Any]) -> str:
    transcript = str(item.get("transcript") or "").strip()
    if transcript:
        return transcript
    steps = parse_dialogue_score_steps(item)
    return str(steps.get("transcript") or "").strip()


def fetch_submission_leader(challenge_uid: str, *, stage: str = "qualifying") -> dict[str, Any] | None:
    payload = fetch_json(
        f"{RESULTS_API_BASE}/challenges/{challenge_uid}/submissions",
        params={"stage": stage, "limit": 1},
    )
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    return first if isinstance(first, dict) else None


def fetch_top_miner_hotkey(challenge_uid: str, *, stage: str = "qualifying") -> str | None:
    leader = fetch_submission_leader(challenge_uid, stage=stage)
    if leader is None:
        return None
    hotkey = str(leader.get("miner_hotkey") or "").strip()
    return hotkey or None


def _deepinfra_headers() -> dict[str, str]:
    token = os.environ.get("DEEPINFRA_TOKEN", "").strip()
    if not token:
        raise ApiChallengeError("DEEPINFRA_TOKEN is not set")
    return {"Authorization": f"Bearer {token}"}


_SOURCE_LANGUAGE_LABELS = {
    "fr": "French",
    "de": "German",
    "en": "English",
}

_TTS_MODELS = {
    "en": "ResembleAI/chatterbox",
    "fr": "ResembleAI/chatterbox-multilingual",
    "de": "ResembleAI/chatterbox-multilingual",
}


def translate_reference_to_source(
    reference_text: str,
    *,
    source_language: str,
    target_language: str = "en",
) -> str:
    if source_language == target_language:
        return reference_text.strip()
    source_label = _SOURCE_LANGUAGE_LABELS.get(source_language, source_language)
    target_label = _SOURCE_LANGUAGE_LABELS.get(target_language, target_language)
    response = requests.post(
        "https://api.deepinfra.com/v1/openai/chat/completions",
        headers=_deepinfra_headers(),
        json={
            "model": os.environ.get("DEEPINFRA_LLM_MODEL", "Qwen/Qwen3-14B"),
            "temperature": 0,
            "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "/no_think\n"
                        f"Translate the {target_label} sentence into natural spoken {source_label}. "
                        "Return only the translated sentence."
                    ),
                },
                {"role": "user", "content": reference_text},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    text = str(response.json()["choices"][0]["message"]["content"]).strip()
    return text.strip().strip('"').strip()


def _render_tts_clauses(
    clauses: list[str],
    *,
    source_language: str,
    model: str,
    pause_sec: float = 0.25,
) -> bytes:
    import sys
    from pathlib import Path as _Path

    workspace_root = _Path(__file__).resolve().parents[3]
    miner_root = workspace_root / "babelbit_miner"
    if str(miner_root) not in sys.path:
        sys.path.insert(0, str(miner_root))
    from server import speech_text_utils

    wav_parts: list[bytes] = []
    for clause in clauses:
        text_field = "input" if model in {"Qwen/Qwen3-TTS", "bosonai/HiggsAudioV2.5"} else "text"
        payload: dict[str, Any] = {text_field: clause}
        if source_language:
            payload["language_id"] = source_language
        response = requests.post(
            f"https://api.deepinfra.com/v1/inference/{model}",
            headers=_deepinfra_headers(),
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        if "application/json" not in response.headers.get("content-type", ""):
            wav_parts.append(response.content)
            continue
        audio = str(response.json().get("audio") or "")
        if "," in audio:
            audio = audio.split(",", 1)[1]
        wav_parts.append(base64.b64decode(audio))
    return speech_text_utils.concat_wav_bytes(wav_parts, pause_sec=pause_sec)


def _asr_roundtrip_recall(wav_bytes: bytes, *, text: str, language: str) -> float:
    """Transcribe rendered TTS audio and measure word recall against the source text."""
    response = requests.post(
        "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3",
        headers=_deepinfra_headers(),
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        data={"language": language},
        timeout=180,
    )
    response.raise_for_status()
    asr_text = str(response.json().get("text", "")).strip()
    return transcript_word_recall(asr_text, text)


def render_source_tts_wav(text: str, *, source_language: str) -> bytes:
    import sys
    from pathlib import Path as _Path

    workspace_root = _Path(__file__).resolve().parents[3]
    miner_root = workspace_root / "babelbit_miner"
    if str(miner_root) not in sys.path:
        sys.path.insert(0, str(miner_root))
    from server import speech_text_utils

    model = os.environ.get("DEEPINFRA_TTS_MODEL", "").strip() or _TTS_MODELS.get(
        source_language, "ResembleAI/chatterbox-multilingual"
    )
    if source_language in {"fr", "de"} and "chatterbox" not in model.lower():
        model = _TTS_MODELS[source_language]

    min_duration_sec = estimate_min_source_duration_sec(text)
    attempts = [(10, 0.25), (10, 0.45), (8, 0.45), (8, 0.65), (6, 0.65), (6, 0.85)]
    best_key = (-1.0, -1.0)
    best_wav = b""
    for max_words, pause_sec in attempts:
        expanded = speech_text_utils.expand_digits_for_tts(text)
        clauses = speech_text_utils.split_clauses_for_tts(expanded, max_words=max_words)
        merged = resample_wav_to_rate(
            _render_tts_clauses(
                clauses,
                source_language=source_language,
                model=model,
                pause_sec=pause_sec,
            )
        )
        duration_sec = speech_text_utils.wav_duration_sec_from_bytes(merged)
        duration_ok = 1.0 if duration_sec >= min_duration_sec * 0.85 else 0.0
        recall = _asr_roundtrip_recall(merged, text=text, language=source_language)
        if duration_ok and recall >= 0.8:
            return merged
        if (duration_ok, recall) > best_key:
            best_key = (duration_ok, recall)
            best_wav = merged
    if best_wav:
        return best_wav
    raise ApiChallengeError(
        f"Failed to render usable source TTS audio for text: {text[:80]!r}"
    )


def build_utterance_entry_from_dialogue_score(
    *,
    item: dict[str, Any],
    challenge_uid: str,
    flat_index: int,
    source_language: str,
    target_language: str,
) -> dict[str, Any]:
    ground_truth = str(item.get("ground_truth") or "").strip()
    if not ground_truth:
        raise ApiChallengeError(
            f"Missing ground_truth for dialogue_uid={item.get('dialogue_uid')} "
            f"utterance_number={item.get('utterance_number')}"
        )
    source_text = translate_reference_to_source(
        ground_truth,
        source_language=source_language,
        target_language=target_language,
    )
    wav_bytes = render_source_tts_wav(source_text, source_language=source_language)
    metadata = resolve_audio_reference_metadata(
        challenge_uid=challenge_uid,
        utterance_id=str(flat_index),
        target_lang=target_language,
        challenge_doc={
            "challenge_uid": challenge_uid,
            "utterances": [
                {
                    "utterance_id": flat_index,
                    "utterance_translations": [
                        {
                            "language": target_language,
                            "text": ground_truth,
                        }
                    ],
                }
            ],
        },
        metadata_source="results.babelbit.ai/dialogue-scores",
    )
    return {
        "utterance_id": str(flat_index),
        "utterance_index": flat_index,
        "dialogue_index": int(str(item.get("dialogue_uid") or flat_index)),
        "dialogue_utterance_index": int(str(item.get("utterance_number") or flat_index)),
        "source_text": source_text,
        "production_ground_truth": ground_truth,
        "production_transcript": extract_dialogue_score_transcript(item),
        "production_accuracy": item.get("accuracy"),
        "utterance_translations": [
            {
                "language": target_language,
                "text": metadata.reference_text,
                "reference_wps": metadata.reference_wps,
                "words": metadata.reference_words,
            }
        ],
        "_source_wav_bytes": wav_bytes,
    }


def fixture_tts_parallel_workers(utterance_count: int) -> int:
    raw = os.environ.get("BB_FIXTURE_TTS_WORKERS", "").strip()
    if raw:
        try:
            configured = max(1, int(raw))
        except ValueError:
            configured = 6
    else:
        configured = 6
    return max(1, min(configured, utterance_count, 10))


def write_challenge_json_atomic(sample_path: Path, challenge_doc: dict[str, Any]) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(challenge_doc, indent=2) + "\n"
    temp_path = sample_path.with_suffix(".json.tmp")
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(sample_path)


def persist_utterance_fixture_sidecar(
    sample_path: Path,
    entry: dict[str, Any],
) -> Path:
    wav_bytes = entry.get("_source_wav_bytes")
    if not isinstance(wav_bytes, (bytes, bytearray)):
        raise ApiChallengeError(
            f"Missing cached source audio bytes for utterance {entry.get('utterance_id')}"
        )
    sidecar = source_audio_sidecar_path(sample_path, utterance_id=str(entry["utterance_id"]))
    sidecar.write_bytes(wav_bytes)
    return sidecar


def _persist_fixture_progress(
    *,
    sample_path: Path,
    challenge_uid: str,
    source_language: str,
    target_language: str,
    utterance_entries: list[dict[str, Any] | None],
    index: int,
    entry: dict[str, Any],
    on_progress: Callable[[str], None] | None,
) -> None:
    utterance_entries[index] = entry
    persist_utterance_fixture_sidecar(sample_path, entry)
    completed = [item for item in utterance_entries if item is not None]
    completed.sort(key=lambda item: int(item["utterance_id"]))
    write_challenge_json_atomic(
        sample_path,
        build_challenge_doc(
            challenge_uid=challenge_uid,
            source_language=source_language,
            target_language=target_language,
            utterance_entries=completed,
            derived_from="results.babelbit.ai/dialogue-scores+deepinfra-tts",
        ),
    )
    if on_progress is not None:
        on_progress(
            f"fixture saved {index + 1}/{len(utterance_entries)} "
            f"dialogue={entry.get('dialogue_index')} utt={entry.get('dialogue_utterance_index')}"
        )


def prepare_fixtures_from_dialogue_scores(
    *,
    challenge_uid: str,
    items: list[dict[str, Any]],
    out_dir: Path,
    source_language: str,
    target_language: str,
    max_utterances: int | None = None,
    on_progress: Callable[[str], None] | None = None,
    parallel_workers: int | None = None,
) -> Path:
    unique_items = dedupe_dialogue_score_items(items)
    if max_utterances is not None:
        unique_items = unique_items[: max(0, max_utterances)]
    if not unique_items:
        raise ApiChallengeError(f"No dialogue-scores utterances found for {challenge_uid}")

    total = len(unique_items)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "challenge.json"
    utterance_entries: list[dict[str, Any] | None] = [None] * total
    progress_lock = threading.Lock()
    workers = parallel_workers if parallel_workers is not None else fixture_tts_parallel_workers(total)

    def _build_one(index: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return index, build_utterance_entry_from_dialogue_score(
            item=item,
            challenge_uid=challenge_uid,
            flat_index=index,
            source_language=source_language,
            target_language=target_language,
        )

    def _handle_built(index: int, entry: dict[str, Any]) -> None:
        with progress_lock:
            _persist_fixture_progress(
                sample_path=sample_path,
                challenge_uid=challenge_uid,
                source_language=source_language,
                target_language=target_language,
                utterance_entries=utterance_entries,
                index=index,
                entry=entry,
                on_progress=on_progress,
            )

    if workers <= 1 or total <= 1:
        for index, item in enumerate(unique_items):
            built_index, entry = _build_one(index, item)
            _handle_built(built_index, entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_build_one, index, item)
                for index, item in enumerate(unique_items)
            ]
            for future in as_completed(futures):
                built_index, entry = future.result()
                _handle_built(built_index, entry)

    final_entries = [utterance_entries[index] for index in range(total)]
    if any(entry is None for entry in final_entries):
        raise ApiChallengeError(f"Incomplete fixture build for {challenge_uid}")
    write_challenge_json_atomic(
        sample_path,
        build_challenge_doc(
            challenge_uid=challenge_uid,
            source_language=source_language,
            target_language=target_language,
            utterance_entries=final_entries,
            derived_from="results.babelbit.ai/dialogue-scores+deepinfra-tts",
        ),
    )
    return sample_path


def flat_utterance_id(raw_utterance_id: Any, *, fallback_index: int) -> str:
    text = str(raw_utterance_id or "").strip()
    if not text:
        return str(fallback_index)
    if ":" in text:
        return text.rsplit(":", 1)[-1]
    return text


def decode_ue_audio_bytes(payload: dict[str, Any]) -> bytes:
    audio_b64 = str(payload.get("audio_b64") or "")
    if not audio_b64:
        raise ApiChallengeError("Utterance engine payload is missing audio_b64")
    raw = base64.b64decode(audio_b64)
    if raw.startswith(b"RIFF"):
        return raw
    sample_rate_hz = int(payload.get("sample_rate_hz") or 0)
    channels = int(payload.get("channels") or 1)
    sample_width_bytes = int(payload.get("sample_width_bytes") or 2)
    if sample_rate_hz <= 0 or channels <= 0 or sample_width_bytes <= 0:
        raise ApiChallengeError("Utterance engine PCM payload is missing audio format fields")
    if sample_width_bytes != 2:
        raise ApiChallengeError(f"Unsupported PCM sample width: {sample_width_bytes}")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width_bytes)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(raw)
    return out.getvalue()


def resample_wav_to_rate(wav_bytes: bytes, *, target_rate_hz: int = BENCHMARK_SAMPLE_RATE_HZ) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise ApiChallengeError(f"Unsupported WAV sample width: {sample_width}")
    if channels != 1:
        raise ApiChallengeError(f"Expected mono source audio, got channels={channels}")
    if sample_rate == target_rate_hz:
        return wav_bytes

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    source_count = samples.shape[0]
    if source_count == 0:
        raise ApiChallengeError("Cannot resample empty WAV")
    target_count = max(1, int(round(source_count * target_rate_hz / float(sample_rate))))
    source_positions = np.linspace(0.0, source_count - 1.0, num=source_count, dtype=np.float32)
    target_positions = np.linspace(0.0, source_count - 1.0, num=target_count, dtype=np.float32)
    resampled = np.interp(target_positions, source_positions, samples)
    pcm = np.clip(np.round(resampled), -32768, 32767).astype(np.int16).tobytes()

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(target_rate_hz)
        wav.writeframes(pcm)
    return out.getvalue()


def _lookup_transcription_entry(
    transcription_metadata: dict[str, Any] | None,
    *,
    utterance_id: str,
    utterance_index: int,
) -> dict[str, Any] | None:
    if not isinstance(transcription_metadata, dict):
        return None
    utterances = transcription_metadata.get("utterances")
    if not isinstance(utterances, list):
        return None
    for entry in utterances:
        if not isinstance(entry, dict):
            continue
        entry_id = flat_utterance_id(entry.get("utterance_id"), fallback_index=utterance_index)
        if entry_id == utterance_id:
            return entry
        if int(entry.get("utterance_index", -1)) == utterance_index:
            return entry
    return None


def build_utterance_entry_from_ue(
    *,
    payload: dict[str, Any],
    transcription_metadata: dict[str, Any] | None,
    fallback_index: int,
    target_language: str = "en",
    source_text: str = "",
) -> dict[str, Any]:
    utterance_id = flat_utterance_id(
        payload.get("utterance_id"),
        fallback_index=fallback_index,
    )
    utterance_index = int(payload.get("utterance_index", fallback_index) or fallback_index)
    transcription_entry = _lookup_transcription_entry(
        transcription_metadata,
        utterance_id=utterance_id,
        utterance_index=utterance_index,
    )
    if transcription_entry is None:
        raise ApiChallengeError(
            f"Missing transcription metadata for utterance_id={utterance_id} index={utterance_index}"
        )

    metadata = resolve_audio_reference_metadata(
        challenge_uid=str(payload.get("challenge_uid") or transcription_metadata.get("challenge_uid") or ""),
        utterance_id=utterance_id,
        target_lang=target_language,
        challenge_doc={
            "challenge_uid": payload.get("challenge_uid"),
            "utterances": [transcription_entry],
        },
        metadata_source="utterance_engine/transcription",
    )
    wav_bytes = resample_wav_to_rate(decode_ue_audio_bytes(payload))
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        source_duration_sec = wav.getnframes() / float(wav.getframerate())

    return {
        "utterance_id": utterance_id,
        "utterance_index": utterance_index,
        "dialogue_index": int(payload.get("dialogue_index", 0) or 0),
        "dialogue_utterance_index": int(
            payload.get("dialogue_utterance_index", utterance_index) or utterance_index
        ),
        "source_text": source_text,
        "utterance_translations": [
            {
                "language": target_language,
                "text": metadata.reference_text,
                "reference_wps": metadata.reference_wps,
                "words": metadata.reference_words,
            }
        ],
        "_source_wav_bytes": wav_bytes,
    }


def build_challenge_doc(
    *,
    challenge_uid: str,
    source_language: str,
    target_language: str,
    utterance_entries: list[dict[str, Any]],
    derived_from: str,
) -> dict[str, Any]:
    clean_entries: list[dict[str, Any]] = []
    for entry in utterance_entries:
        copied = dict(entry)
        copied.pop("_source_wav_bytes", None)
        clean_entries.append(copied)
    return {
        "challenge_uid": challenge_uid,
        "language": source_language,
        "source_language": source_language,
        "target_language": target_language,
        "derived_from": derived_from,
        "dialogues": [],
        "utterances": clean_entries,
    }


def materialize_challenge_fixtures(
    *,
    challenge_doc: dict[str, Any],
    utterance_entries: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "challenge.json"
    sample_path.write_text(json.dumps(challenge_doc, indent=2) + "\n", encoding="utf-8")
    for entry in utterance_entries:
        wav_bytes = entry.get("_source_wav_bytes")
        if not isinstance(wav_bytes, (bytes, bytearray)):
            raise ApiChallengeError(
                f"Missing cached source audio bytes for utterance {entry.get('utterance_id')}"
            )
        sidecar = source_audio_sidecar_path(sample_path, utterance_id=str(entry["utterance_id"]))
        sidecar.write_bytes(wav_bytes)
    return sample_path


def dialogue_scores_fetch_limit(max_utterances: int | None) -> int:
    if max_utterances is not None:
        return max(50, max_utterances * 5)
    return 200


def cache_dir_for_challenge(workspace_root: Path, *, challenge_uid: str) -> Path:
    return api_challenge_fixtures_dir(workspace_root, challenge_uid=challenge_uid)


def legacy_cache_dir_for_challenge(workspace_root: Path, *, challenge_uid: str) -> Path:
    return legacy_api_challenge_fixtures_dir(workspace_root, challenge_uid=challenge_uid)


def load_cached_challenge(cache_dir: Path) -> Path | None:
    sample_path = cache_dir / "challenge.json"
    if sample_path.is_file():
        return sample_path
    return None


def resolve_cached_challenge_path(
    workspace_root: Path,
    *,
    challenge_uid: str,
    stage_name: str | None = None,
) -> Path | None:
    primary = cache_dir_for_challenge(workspace_root, challenge_uid=challenge_uid)
    legacy = legacy_cache_dir_for_challenge(workspace_root, challenge_uid=challenge_uid)
    if stage_name:
        for root in (primary, legacy):
            cached = load_cached_challenge(root / "stages" / stage_name)
            if cached is not None:
                return cached
    for root in (primary, legacy):
        cached = load_cached_challenge(root)
        if cached is not None:
            return cached
    return None


def _is_challenge_fixture_name(name: str) -> bool:
    if name == "challenge.json":
        return True
    return fnmatch.fnmatch(name, "challenge.u*.source.wav")


def _legacy_fixture_moves(legacy_dir: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    if not legacy_dir.is_dir():
        return moves

    stages_root = legacy_dir / "stages"
    if stages_root.is_dir():
        for stage_dir in sorted(stages_root.iterdir()):
            if not stage_dir.is_dir():
                continue
            for path in sorted(stage_dir.iterdir()):
                if path.is_file() and _is_challenge_fixture_name(path.name):
                    moves.append((path, Path("stages") / stage_dir.name / path.name))

    for path in sorted(legacy_dir.iterdir()):
        if not path.is_file():
            continue
        if _is_challenge_fixture_name(path.name):
            moves.append((path, Path("stages") / "qualifying" / path.name))

    return moves


def migrate_legacy_api_challenge_fixtures(
    workspace_root: Path,
    *,
    challenge_uid: str | None = None,
    remove_legacy: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    legacy_root = workspace_root / "babelbit_subnet" / "benchmark_outputs" / "api_challenges"
    if not legacy_root.is_dir():
        return []

    challenge_dirs: list[Path]
    if challenge_uid:
        challenge_dirs = [legacy_root / challenge_uid]
    else:
        challenge_dirs = sorted(path for path in legacy_root.iterdir() if path.is_dir())

    reports: list[dict[str, Any]] = []
    for legacy_dir in challenge_dirs:
        if not legacy_dir.is_dir():
            continue
        uid = legacy_dir.name
        target_root = api_challenge_fixtures_dir(workspace_root, challenge_uid=uid)
        moved_files: list[str] = []
        skipped_files: list[str] = []

        for src, rel_dest in _legacy_fixture_moves(legacy_dir):
            dest = target_root / rel_dest
            if dest.is_file():
                skipped_files.append(str(rel_dest))
                continue
            if dry_run:
                moved_files.append(str(rel_dest))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved_files.append(str(rel_dest))

        if remove_legacy and not dry_run:
            stages_root = legacy_dir / "stages"
            if stages_root.is_dir():
                for stage_dir in sorted(stages_root.iterdir(), reverse=True):
                    if stage_dir.is_dir() and not any(stage_dir.iterdir()):
                        stage_dir.rmdir()
                if stages_root.is_dir() and not any(stages_root.iterdir()):
                    stages_root.rmdir()

        if moved_files or skipped_files:
            reports.append(
                {
                    "challenge_uid": uid,
                    "target_root": str(target_root),
                    "moved_files": moved_files,
                    "skipped_files": skipped_files,
                    "dry_run": dry_run,
                }
            )

    return reports


async def download_challenge_from_engine(
    *,
    utterance_engine_url: str,
    wallet_cold: str,
    wallet_hot: str,
    max_utterances: int | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    from babelbit.utils.predict_audio import (
        _inline_metadata_from_ue_payloads,
        _response_to_ue_utterance,
        fetch_transcription_ground_truth,
        next_source_audio_utterance,
        start_source_audio_session,
    )
    from babelbit.utils.utterance_auth import authenticate_utterance_engine, init_utterance_auth

    init_utterance_auth(utterance_engine_url, wallet_cold, wallet_hot)
    await authenticate_utterance_engine()

    start_data = await start_source_audio_session(utterance_engine_url)
    challenge_uid = str(start_data.get("challenge_uid") or "").strip()
    if not challenge_uid:
        raise ApiChallengeError("Utterance engine start payload is missing challenge_uid")

    payloads: list[dict[str, Any]] = [start_data]
    current = _response_to_ue_utterance(start_data)
    if current is None:
        raise ApiChallengeError("Utterance engine returned no active utterances")

    transcription_metadata: dict[str, Any] | None = None
    try:
        transcription_payload = await fetch_transcription_ground_truth(utterance_engine_url)
        payload_uid = str(transcription_payload.get("challenge_uid") or "")
        payload_metadata = transcription_payload.get("metadata")
        if payload_uid == challenge_uid and isinstance(payload_metadata, dict):
            transcription_metadata = payload_metadata
    except Exception:
        transcription_metadata = None

    if transcription_metadata is None:
        transcription_metadata = _inline_metadata_from_ue_payloads(
            challenge_uid=challenge_uid,
            payloads=payloads,
        )

    session_id = current.session_id
    while current is not None and not current.done:
        if max_utterances is not None and len(payloads) >= max_utterances:
            break
        next_data = await next_source_audio_utterance(utterance_engine_url, session_id)
        payloads.append(next_data)
        if transcription_metadata is None:
            transcription_metadata = _inline_metadata_from_ue_payloads(
                challenge_uid=challenge_uid,
                payloads=payloads,
            )
        current = _response_to_ue_utterance(next_data)

    if max_utterances is not None:
        payloads = payloads[: max(0, max_utterances)]

    return challenge_uid, start_data, payloads, transcription_metadata


def prepare_challenge_fixtures(
    *,
    challenge_uid: str,
    start_data: dict[str, Any],
    payloads: list[dict[str, Any]],
    transcription_metadata: dict[str, Any] | None,
    out_dir: Path,
    target_language: str = "en",
) -> Path:
    source_language = str(start_data.get("language") or "en")
    utterance_entries: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        if not str(payload.get("audio_b64") or "").strip():
            continue
        utterance_entries.append(
            build_utterance_entry_from_ue(
                payload={**payload, "challenge_uid": challenge_uid},
                transcription_metadata=transcription_metadata,
                fallback_index=index,
                target_language=target_language,
                source_text="",
            )
        )
    if not utterance_entries:
        raise ApiChallengeError(f"No utterances with audio found for {challenge_uid}")

    challenge_doc = build_challenge_doc(
        challenge_uid=challenge_uid,
        source_language=source_language,
        target_language=target_language,
        utterance_entries=utterance_entries,
        derived_from="api.babelbit.ai/source-audio",
    )
    return materialize_challenge_fixtures(
        challenge_doc=challenge_doc,
        utterance_entries=utterance_entries,
        out_dir=out_dir,
    )
