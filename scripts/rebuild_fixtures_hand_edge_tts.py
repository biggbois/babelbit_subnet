#!/usr/bin/env python3
"""Rebuild api_challenge fixtures from hand EN↔FR/DE texts + Microsoft edge-tts.

Wipes existing challenge dirs for the target UID (+ optional -de) first.
No GPU required; needs ffmpeg + edge-tts on PATH/venv.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from babelbit.benchmarks.api_challenge import (
    BENCHMARK_SAMPLE_RATE_HZ,
    build_challenge_doc,
    materialize_challenge_fixtures,
)
from babelbit.benchmarks.edge_fixture_tts import synthesize_edge_tts_mp3_sync
from babelbit.benchmarks.local_fixture_synth import (
    DERIVED_FROM_HAND_EDGE_TTS,
    DEFAULT_EDGE_RATE,
    challenge_uid_for_locale,
    edge_voice_for_locale,
    mp3_bytes_to_wav_bytes,
)
from babelbit.benchmarks.miner_test_data import (
    api_challenge_fixtures_dir,
    miner_test_data_root,
    workspace_root_from,
)
from babelbit.scoring.reference_metadata import resolve_audio_reference_metadata


class EdgeTTS:
    def __init__(self, *, rate: str = DEFAULT_EDGE_RATE, ffmpeg_bin: str = "ffmpeg") -> None:
        self.rate = rate
        self.ffmpeg_bin = ffmpeg_bin

    def synthesize(self, text: str, *, language_id: str) -> bytes:
        voice = edge_voice_for_locale(language_id)
        mp3 = synthesize_edge_tts_mp3_sync(text, voice=voice, rate=self.rate)
        return mp3_bytes_to_wav_bytes(
            mp3,
            target_rate_hz=BENCHMARK_SAMPLE_RATE_HZ,
            ffmpeg_bin=self.ffmpeg_bin,
        )


def load_hand_translations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("utterances")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"No utterances in {path}")
    return [item for item in items if isinstance(item, dict)]


def build_entries(
    *,
    challenge_uid: str,
    items: list[dict[str, Any]],
    source_language: str,
    tts: EdgeTTS,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        ground_truth = str(item.get("en") or "").strip()
        source_text = str(item.get(source_language) or "").strip()
        if not ground_truth or not source_text:
            raise SystemExit(f"Missing en/{source_language} for utterance {index}")
        print(f"{source_language} u{index}: edge-tts…", flush=True)
        wav_bytes = tts.synthesize(source_text, language_id=source_language)
        metadata = resolve_audio_reference_metadata(
            challenge_uid=challenge_uid,
            utterance_id=str(index),
            target_lang="en",
            challenge_doc={
                "challenge_uid": challenge_uid,
                "utterances": [
                    {
                        "utterance_id": index,
                        "utterance_translations": [
                            {"language": "en", "text": ground_truth}
                        ],
                    }
                ],
            },
            metadata_source="hand-translations+edge-tts",
        )
        entries.append(
            {
                "utterance_id": str(index),
                "utterance_index": index,
                "dialogue_index": index,
                "dialogue_utterance_index": index,
                "source_text": source_text,
                "production_ground_truth": ground_truth,
                "production_transcript": "",
                "production_accuracy": None,
                "utterance_translations": [
                    {
                        "language": "en",
                        "text": metadata.reference_text,
                        "reference_wps": metadata.reference_wps,
                        "words": metadata.reference_words,
                    }
                ],
                "_source_wav_bytes": wav_bytes,
            }
        )
    return entries


def wipe_challenge(data_root: Path, challenge_uid: str) -> None:
    target = data_root / "api_challenges" / challenge_uid
    if target.is_dir():
        shutil.rmtree(target)
        print(f"wiped {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--locales", default="fr,de")
    parser.add_argument("--rate", default=DEFAULT_EDGE_RATE, help="edge-tts rate, e.g. -10%")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--stages",
        default="qualifying,arena",
        help="Comma-separated stage folder names",
    )
    args = parser.parse_args()

    subnet_root = Path(__file__).resolve().parents[1]
    workspace_root = workspace_root_from(subnet_root)
    data_root = miner_test_data_root(workspace_root)
    items = load_hand_translations(args.translations)
    base_uid = str(
        json.loads(args.translations.read_text(encoding="utf-8")).get("challenge_uid")
        or ""
    ).strip()
    if not base_uid:
        raise SystemExit("translations JSON missing challenge_uid")

    locales = [part.strip().lower() for part in args.locales.split(",") if part.strip()]
    stages = [part.strip() for part in args.stages.split(",") if part.strip()]

    for locale in locales:
        wipe_challenge(data_root, challenge_uid_for_locale(base_uid, source_language=locale))

    tts = EdgeTTS(rate=args.rate, ffmpeg_bin=args.ffmpeg)
    written: list[str] = []
    for locale in locales:
        locale_uid = challenge_uid_for_locale(base_uid, source_language=locale)
        for stage in stages:
            entries = build_entries(
                challenge_uid=locale_uid,
                items=items,
                source_language=locale,
                tts=tts,
            )
            out_dir = (
                api_challenge_fixtures_dir(workspace_root, challenge_uid=locale_uid)
                / "stages"
                / stage
            )
            doc = build_challenge_doc(
                challenge_uid=locale_uid,
                source_language=locale,
                target_language="en",
                utterance_entries=entries,
                derived_from=DERIVED_FROM_HAND_EDGE_TTS,
            )
            path = materialize_challenge_fixtures(
                challenge_doc=doc,
                utterance_entries=entries,
                out_dir=out_dir,
            )
            written.append(str(path))
            print(f"wrote {path}", flush=True)

    manifest = {
        "base_challenge_uid": base_uid,
        "locales": locales,
        "stages": stages,
        "paths": written,
        "derived_from": DERIVED_FROM_HAND_EDGE_TTS,
        "edge_rate": args.rate,
        "translations_file": str(args.translations),
    }
    manifest_path = data_root / "api_challenges" / "LOCAL_SYNTH_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
