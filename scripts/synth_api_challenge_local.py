#!/usr/bin/env python3
"""Synthesize api_challenge fixtures from results.babelbit.ai using local Qwen + Chatterbox.

No DeepInfra. Intended to run on a GPU VM (e.g. Oblivus L40).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from babelbit.benchmarks.api_challenge import (
    BENCHMARK_SAMPLE_RATE_HZ,
    ApiChallengeError,
    build_challenge_doc,
    dedupe_dialogue_score_items,
    extract_dialogue_score_transcript,
    fetch_dialogue_scores,
    get_challenge_uid,
    get_latest_challenge,
    materialize_challenge_fixtures,
    resample_wav_to_rate,
)
from babelbit.benchmarks.local_fixture_synth import (
    DERIVED_FROM_LOCAL,
    SOURCE_LANGUAGE_LABELS,
    build_translate_system_prompt,
    challenge_uid_for_locale,
    concat_wav_bytes,
    expand_digits_for_tts,
    float_audio_to_wav_bytes,
    parse_translation_response,
    split_clauses_for_tts,
    wav_duration_sec_from_bytes,
)
from babelbit.benchmarks.miner_test_data import (
    api_challenge_fixtures_dir,
    estimate_min_source_duration_sec,
    miner_test_data_root,
    transcript_word_recall,
    workspace_root_from,
)
from babelbit.scoring.reference_metadata import resolve_audio_reference_metadata


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class LocalTranslator:
    def __init__(self, model_id: str, *, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(device)
        self.device = device
        self.model.eval()

    def translate(self, english_text: str, *, source_language: str) -> str:
        if source_language == "en":
            return english_text.strip()
        system = build_translate_system_prompt(source_language=source_language)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": english_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False,
            )
        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = parse_translation_response(raw)
        if not text:
            raise ApiChallengeError(f"Empty translation for {english_text!r}")
        return text


class LocalChatterboxTTS:
    def __init__(self, *, device: str) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        kwargs: dict[str, Any] = {"device": device}
        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(t3_model="v3", **kwargs)
        except TypeError:
            self.model = ChatterboxMultilingualTTS.from_pretrained(**kwargs)

    def synthesize_clause(self, text: str, *, language_id: str) -> bytes:
        candidates = [text]
        words = text.split()
        if len(words) > 4:
            candidates.append(" ".join(words[: max(4, len(words) // 2)]))
        if len(words) > 2:
            candidates.append(" ".join(words[:2]))
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                with torch.inference_mode():
                    wav = self.model.generate(candidate, language_id=language_id)
                if hasattr(wav, "detach"):
                    audio = wav.detach().float().cpu().numpy()
                else:
                    audio = np.asarray(wav, dtype=np.float32)
                return float_audio_to_wav_bytes(audio, sample_rate_hz=int(self.model.sr))
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Chatterbox failed for {text!r}: {last_error}")


class LocalWhisperASR:
    def __init__(self, model_size: str = "medium", *, device: str) -> None:
        from faster_whisper import WhisperModel

        compute_type = "float16" if device.startswith("cuda") else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_bytes: bytes, *, language: str) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(wav_bytes)
            handle.flush()
            segments, _info = self.model.transcribe(handle.name, language=language)
            return " ".join(segment.text.strip() for segment in segments).strip()


def render_source_tts_wav_local(
    text: str,
    *,
    source_language: str,
    tts: LocalChatterboxTTS,
    asr: LocalWhisperASR | None,
) -> bytes:
    min_duration_sec = estimate_min_source_duration_sec(text)
    attempts = [(10, 0.25), (10, 0.45), (8, 0.45), (8, 0.65), (6, 0.65), (6, 0.85), (4, 0.85)]
    best_key = (-1.0, -1.0)
    best_wav = b""
    last_error: Exception | None = None
    for max_words, pause_sec in attempts:
        expanded = expand_digits_for_tts(text)
        clauses = split_clauses_for_tts(expanded, max_words=max_words)
        try:
            parts = [
                tts.synthesize_clause(clause, language_id=source_language)
                for clause in clauses
            ]
            merged = resample_wav_to_rate(
                concat_wav_bytes(parts, pause_sec=pause_sec),
                target_rate_hz=BENCHMARK_SAMPLE_RATE_HZ,
            )
        except Exception as exc:  # Chatterbox can IndexError on some DE clauses
            last_error = exc
            continue
        duration_sec = wav_duration_sec_from_bytes(merged)
        duration_ok = 1.0 if duration_sec >= min_duration_sec * 0.85 else 0.0
        recall = 1.0
        if asr is not None:
            asr_text = asr.transcribe(merged, language=source_language)
            recall = transcript_word_recall(asr_text, text)
            if duration_ok and recall >= 0.8:
                return merged
        elif duration_ok:
            return merged
        if (duration_ok, recall) > best_key:
            best_key = (duration_ok, recall)
            best_wav = merged
    if best_wav:
        return best_wav
    detail = f" last_error={last_error!r}" if last_error is not None else ""
    raise ApiChallengeError(f"Failed local TTS render for text: {text[:80]!r}{detail}")


def build_utterance_entry_local(
    *,
    item: dict[str, Any],
    challenge_uid: str,
    flat_index: int,
    source_language: str,
    target_language: str,
    translator: LocalTranslator,
    tts: LocalChatterboxTTS,
    asr: LocalWhisperASR | None,
) -> dict[str, Any]:
    ground_truth = str(item.get("ground_truth") or "").strip()
    if not ground_truth:
        raise ApiChallengeError(
            f"Missing ground_truth dialogue_uid={item.get('dialogue_uid')} "
            f"utterance_number={item.get('utterance_number')}"
        )
    source_text = translator.translate(ground_truth, source_language=source_language)
    wav_bytes = render_source_tts_wav_local(
        source_text,
        source_language=source_language,
        tts=tts,
        asr=asr,
    )
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
                        {"language": target_language, "text": ground_truth}
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


def prepare_fixtures_local(
    *,
    challenge_uid: str,
    items: list[dict[str, Any]],
    out_dir: Path,
    source_language: str,
    target_language: str,
    translator: LocalTranslator,
    tts: LocalChatterboxTTS,
    asr: LocalWhisperASR | None,
    max_utterances: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    unique_items = dedupe_dialogue_score_items(items)
    if max_utterances is not None:
        unique_items = unique_items[: max(0, max_utterances)]
    if not unique_items:
        raise ApiChallengeError(f"No dialogue-scores utterances for {challenge_uid}")

    entries: list[dict[str, Any]] = []
    for index, item in enumerate(unique_items):
        entry = build_utterance_entry_local(
            item=item,
            challenge_uid=challenge_uid,
            flat_index=index,
            source_language=source_language,
            target_language=target_language,
            translator=translator,
            tts=tts,
            asr=asr,
        )
        entries.append(entry)
        if on_progress is not None:
            on_progress(
                f"{source_language} {out_dir.name} "
                f"{index + 1}/{len(unique_items)} uid={entry['utterance_id']}"
            )

    challenge_doc = build_challenge_doc(
        challenge_uid=challenge_uid,
        source_language=source_language,
        target_language=target_language,
        utterance_entries=entries,
        derived_from=DERIVED_FROM_LOCAL,
    )
    return materialize_challenge_fixtures(
        challenge_doc=challenge_doc,
        utterance_entries=entries,
        out_dir=out_dir,
    )


def wipe_api_challenges(data_root: Path) -> list[str]:
    root = data_root / "api_challenges"
    removed: list[str] = []
    if not root.is_dir():
        return removed
    for child in sorted(root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
            removed.append(child.name)
    return removed


def resolve_challenge_uid(explicit: str | None) -> str:
    if explicit:
        return explicit
    return get_challenge_uid(get_latest_challenge(status="completed"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-uid", default="")
    parser.add_argument("--locales", default="fr,de", help="Comma-separated source locales")
    parser.add_argument("--target-language", default="en")
    parser.add_argument("--max-utterances", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--translate-model",
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--whisper-size", default="medium")
    parser.add_argument("--skip-asr-gate", action="store_true")
    parser.add_argument("--wipe-old", action="store_true", help="Delete all api_challenges/* first")
    parser.add_argument("--miner-env", type=Path, default=None)
    args = parser.parse_args()

    subnet_root = Path(__file__).resolve().parents[1]
    workspace_root = workspace_root_from(subnet_root)
    data_root = miner_test_data_root(workspace_root)
    miner_env = args.miner_env or (workspace_root / "babelbit_miner" / ".env")
    _load_env(miner_env)

    if args.wipe_old:
        removed = wipe_api_challenges(data_root)
        print(f"wiped api_challenges: {removed}")

    challenge_uid = resolve_challenge_uid(args.challenge_uid.strip() or None)
    locales = [part.strip().lower() for part in args.locales.split(",") if part.strip()]
    for locale in locales:
        if locale not in SOURCE_LANGUAGE_LABELS:
            raise SystemExit(f"Unsupported locale {locale!r}")

    print(f"challenge_uid={challenge_uid}")
    print("loading translator + chatterbox (+ whisper)…")
    translator = LocalTranslator(args.translate_model, device=args.device)
    tts = LocalChatterboxTTS(device=args.device)
    asr = None if args.skip_asr_gate else LocalWhisperASR(args.whisper_size, device=args.device)

    stage_map = (("qualifying", "main"), ("arena", "arena"))
    written: list[str] = []
    for source_language in locales:
        locale_uid = challenge_uid_for_locale(challenge_uid, source_language=source_language)
        for stage_name, dialogue_stage in stage_map:
            items = fetch_dialogue_scores(
                challenge_uid,
                stage=dialogue_stage,
                limit=max(50, args.max_utterances * 5),
            )
            out_dir = (
                api_challenge_fixtures_dir(workspace_root, challenge_uid=locale_uid)
                / "stages"
                / stage_name
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            path = prepare_fixtures_local(
                challenge_uid=locale_uid,
                items=items,
                out_dir=out_dir,
                source_language=source_language,
                target_language=args.target_language,
                translator=translator,
                tts=tts,
                asr=asr,
                max_utterances=args.max_utterances,
                on_progress=print,
            )
            written.append(str(path))

    manifest = {
        "base_challenge_uid": challenge_uid,
        "locales": locales,
        "paths": written,
        "derived_from": DERIVED_FROM_LOCAL,
    }
    manifest_path = data_root / "api_challenges" / "LOCAL_SYNTH_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
