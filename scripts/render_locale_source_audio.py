#!/usr/bin/env python3
"""Render cached source WAV sidecars for locale miner-test-data files."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any

import requests

from babelbit.benchmarks.miner_test_data import (
    list_challenge_utterance_ids,
    load_miner_test_utterance,
    locale_sample_path,
    miner_test_data_root,
    normalize_locale_list,
    validate_source_audio_asr_roundtrip,
    validate_source_audio_duration,
    workspace_root_from,
)

TTS_MODELS = {
    "en": "ResembleAI/chatterbox",
    "fr": "ResembleAI/chatterbox-multilingual",
    "de": "ResembleAI/chatterbox-multilingual",
}


def _speech_text_utils():
    workspace_root = Path(__file__).resolve().parents[2]
    miner_root = workspace_root / "babelbit_miner"
    if str(miner_root) not in sys.path:
        sys.path.insert(0, str(miner_root))
    from server import speech_text_utils

    return speech_text_utils


def _render_tts_wav(text: str, *, model: str, language_id: str) -> bytes:
    speech_utils = _speech_text_utils()
    clauses = speech_utils.prepare_source_text_for_tts(text, language=language_id)
    wav_parts = [_tts_wav(clause, model=model, language_id=language_id) for clause in clauses]
    return speech_utils.concat_wav_bytes(wav_parts)


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _deepinfra_headers() -> dict[str, str]:
    token = os.environ.get("DEEPINFRA_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DEEPINFRA_TOKEN is not set")
    return {"Authorization": f"Bearer {token}"}


def _tts_wav(text: str, *, model: str, language_id: str) -> bytes:
    text_field = "input" if model in {"Qwen/Qwen3-TTS", "bosonai/HiggsAudioV2.5"} else "text"
    payload: dict[str, Any] = {text_field: text}
    if language_id:
        payload["language_id"] = language_id
    response = requests.post(
        f"https://api.deepinfra.com/v1/inference/{model}",
        headers=_deepinfra_headers(),
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    if "application/json" not in response.headers.get("content-type", ""):
        return response.content
    audio = str(response.json().get("audio") or "")
    if "," in audio:
        audio = audio.split(",", 1)[1]
    return base64.b64decode(audio)


def _asr_text(wav_bytes: bytes, *, language: str) -> str:
    response = requests.post(
        "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3",
        headers=_deepinfra_headers(),
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        data={"language": language},
        timeout=180,
    )
    response.raise_for_status()
    return str(response.json().get("text") or "").strip()


def render_sample(
    sample_path: Path,
    *,
    utterance_id: str,
    overwrite: bool,
) -> Path:
    utterance = load_miner_test_utterance(
        sample_path,
        utterance_id=utterance_id,
        require_source_audio=False,
    )
    out_path = utterance.source_audio_path
    if out_path.exists() and not overwrite:
        return out_path

    last_message = ""
    speech_utils = _speech_text_utils()
    strategies = (
        ["enhanced", "plain"]
        if speech_utils.should_enhance_source_tts(utterance.source_text)
        else ["plain"]
    )
    model = TTS_MODELS.get(utterance.source_language, TTS_MODELS["en"])
    for strategy in strategies:
        for _attempt in range(5):
            try:
                if strategy == "enhanced":
                    wav = _render_tts_wav(
                        utterance.source_text,
                        model=model,
                        language_id=utterance.source_language,
                    )
                else:
                    wav = _tts_wav(
                        utterance.source_text,
                        model=model,
                        language_id=utterance.source_language,
                    )
            except requests.RequestException as exc:
                last_message = str(exc)
                continue
            out_path.write_bytes(wav)
            ok, message = validate_source_audio_duration(utterance.source_text, out_path)
            if not ok:
                last_message = message
                continue
            asr_text = _asr_text(wav, language=utterance.source_language)
            ok, message = validate_source_audio_asr_roundtrip(
                utterance.source_text,
                asr_text=asr_text,
            )
            if ok:
                return out_path
            last_message = message
    raise RuntimeError(last_message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locale", action="append", choices=["en", "fr", "de"])
    parser.add_argument("--en-sample", action="append", dest="en_samples", default=[])
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--utterance-id", action="append", dest="utterance_ids", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--miner-env", type=Path, default=None)
    args = parser.parse_args()

    subnet_root = Path(__file__).resolve().parents[1]
    workspace_root = workspace_root_from(subnet_root)
    data_root = miner_test_data_root(workspace_root)
    miner_env = args.miner_env or (workspace_root / "babelbit_miner/.env")
    _load_env(miner_env)

    locales = normalize_locale_list(args.locale)
    en_samples = args.en_samples or [
        "npr/01/en-npr-001481.json",
        "npr/01/en-npr-001002.json",
        "npr/01/en-npr-001099.json",
    ]

    written: list[str] = []
    for rel in en_samples:
        for locale in locales:
            if locale == "en":
                sample_path = data_root / "en" / rel
            else:
                sample_path = locale_sample_path(
                    workspace_root,
                    locale=locale,
                    relative_en_path=rel,
                )
                if not sample_path.is_file():
                    raise FileNotFoundError(
                        f"Missing locale sample {sample_path}. "
                        "Run scripts/build_locale_test_data.py first."
                    )

            utterance_ids = list_challenge_utterance_ids(
                sample_path,
                max_utterances=args.max_utterances,
                min_words=args.min_words,
            )
            if args.utterance_ids:
                allowed = {str(uid) for uid in args.utterance_ids}
                utterance_ids = [uid for uid in utterance_ids if uid in allowed]
            for utterance_id in utterance_ids:
                out_path = render_sample(
                    sample_path,
                    utterance_id=utterance_id,
                    overwrite=args.overwrite,
                )
                written.append(str(out_path))

    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
