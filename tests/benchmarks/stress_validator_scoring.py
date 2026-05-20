from __future__ import annotations

import argparse
import io
import json
import math
import resource
import statistics
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

from babelbit.scoring.utterance_scoring import score_audio_utterance_batch


def _sine_wav_bytes(
    *,
    duration_sec: float,
    sample_rate_hz: int,
    frequency_hz: float,
    amplitude: float = 0.2,
) -> bytes:
    frame_count = int(duration_sec * sample_rate_hz)
    frames = bytearray()
    for i in range(frame_count):
        value = amplitude * math.sin(
            2.0 * math.pi * frequency_hz * i / sample_rate_hz
        )
        sample = int(
            max(-1.0, min(1.0, value)) * 32767
        )
        frames.extend(sample.to_bytes(2, "little", signed=True))

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(bytes(frames))
    return output.getvalue()


def _challenge_metadata(challenge_uid: str, utterance_id: str) -> dict[str, Any]:
    text = "hello world this is a validator scoring stress test"
    words = []
    for idx, word in enumerate(text.split()):
        start = idx * 0.35
        words.append({"word": word, "start": start, "end": start + 0.25})
    return {
        "challenge_uid": challenge_uid,
        "utterances": [
            {
                "utterance_id": utterance_id,
                "utterance_translations": [
                    {
                        "language": "en",
                        "text": text,
                        "reference_wps": 2.8,
                        "words": words,
                    }
                ],
            }
        ],
    }


def _build_predictions(
    *,
    miner_count: int,
    wav_bytes_by_miner: list[bytes],
    source_duration_sec: float,
) -> list[dict[str, Any]]:
    return [
        {
            "predicted_wav_bytes": wav_bytes_by_miner[i],
            "first_output_frame": i % 3,
            "frame_rate_hz": 12.5,
            "source_duration_sec": source_duration_sec,
        }
        for i in range(miner_count)
    ]


def _synthetic_transcribe(items: list[dict[str, Any]], **_: Any) -> list[dict[str, Any]]:
    text = "hello world this is a validator scoring stress test"
    words = [
        {"word": word, "start": idx * 0.35, "end": idx * 0.35 + 0.25}
        for idx, word in enumerate(text.split())
    ]
    return [{"text": text, "words": words, "error": None} for _item in items]


def _synthetic_accuracy(texts: list[str], *_args: Any, **_kwargs: Any) -> list[float]:
    return [0.95 if text.strip() else 0.0 for text in texts]


@contextmanager
def _scoring_mode(mode: str) -> Iterator[None]:
    if mode == "real":
        yield
        return

    with (
        patch(
            "babelbit.scoring.utterance_scoring.transcribe_wav_bytes_batch",
            side_effect=_synthetic_transcribe,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.compute_accuracy_batch",
            side_effect=_synthetic_accuracy,
        ),
    ):
        yield


def _max_rss_mb() -> float:
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_kb / 1024.0


def _run_once(
    *,
    miner_count: int,
    duration_sec: float,
    sample_rate_hz: int,
    source_duration_sec: float,
    challenge_uid: str,
    utterance_id: str,
    stt_model: str,
    stt_device: str,
    embedder_model: str,
    stt_cache_path: Path,
    mode: str,
) -> dict[str, Any]:
    wav_bytes_by_miner = [
        _sine_wav_bytes(
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
            frequency_hz=330.0 + float(miner_count * 10) + float(i),
        )
        for i in range(miner_count)
    ]
    predictions = _build_predictions(
        miner_count=miner_count,
        wav_bytes_by_miner=wav_bytes_by_miner,
        source_duration_sec=source_duration_sec,
    )
    metadata = _challenge_metadata(challenge_uid, utterance_id)

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    with _scoring_mode(mode):
        scores = score_audio_utterance_batch(
            predictions=predictions,
            challenge_uid=challenge_uid,
            utterance_id=utterance_id,
            source_duration_sec=source_duration_sec,
            challenge_metadata=metadata,
            metadata_source="stress-validator-scoring",
            stt_model=stt_model,
            stt_device=stt_device,
            embedder_model=embedder_model,
            stt_cache_path=stt_cache_path,
        )
    wall_sec = time.perf_counter() - wall_started
    cpu_sec = time.process_time() - cpu_started

    fallback_count = len([score for score in scores if score.get("score_is_fallback")])
    return {
        "miners": miner_count,
        "wall_sec": wall_sec,
        "cpu_sec": cpu_sec,
        "cpu_to_wall": cpu_sec / wall_sec if wall_sec > 0 else None,
        "miners_per_wall_sec": miner_count / wall_sec if wall_sec > 0 else None,
        "miners_per_cpu_sec": miner_count / cpu_sec if cpu_sec > 0 else None,
        "fallback_count": fallback_count,
        "max_rss_mb": _max_rss_mb(),
    }


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(samples[-1])
    metric_keys = (
        "wall_sec",
        "cpu_sec",
        "cpu_to_wall",
        "miners_per_wall_sec",
        "miners_per_cpu_sec",
        "max_rss_mb",
    )
    for key in metric_keys:
        values = [float(sample[key]) for sample in samples if sample[key] is not None]
        summary[f"{key}_median"] = statistics.median(values)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test validator audio scoring at larger miner counts."
    )
    parser.add_argument("--miners", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--sample-rate-hz", type=int, default=24_000)
    parser.add_argument("--source-duration-sec", type=float, default=2.0)
    parser.add_argument("--stt-model", default="faster-whisper-tiny")
    parser.add_argument("--stt-device", default="cpu")
    parser.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--stt-cache-path",
        type=Path,
        default=Path("/tmp/babelbit_scoring_stress_stt_cache.jsonl"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    challenge_uid = "stress-validator-scoring"
    utterance_id = "0"
    rows = []
    if not args.json:
        print(
            "mode miners wall_sec_med cpu_sec_med cpu/wall_med "
            "miners/wall_sec_med miners/cpu_sec_med fallback max_rss_mb_med",
            flush=True,
        )

    for miner_count in args.miners:
        samples = []
        for run_index in range(args.warmup + args.repeat):
            sample = _run_once(
                miner_count=miner_count,
                duration_sec=args.duration_sec,
                sample_rate_hz=args.sample_rate_hz,
                source_duration_sec=args.source_duration_sec,
                challenge_uid=challenge_uid,
                utterance_id=utterance_id,
                stt_model=args.stt_model,
                stt_device=args.stt_device,
                embedder_model=args.embedder_model,
                stt_cache_path=args.stt_cache_path,
                mode=args.mode,
            )
            if run_index >= args.warmup:
                samples.append(sample)
        row = _summarize(samples)
        rows.append(row)
        if not args.json:
            print(
                f"{args.mode} {row['miners']} "
                f"{row['wall_sec_median']:.4f} {row['cpu_sec_median']:.4f} "
                f"{row['cpu_to_wall_median']:.3f} "
                f"{row['miners_per_wall_sec_median']:.2f} "
                f"{row['miners_per_cpu_sec_median']:.2f} "
                f"{row['fallback_count']} {row['max_rss_mb_median']:.1f}",
                flush=True,
            )

    if args.json:
        print(json.dumps({"mode": args.mode, "results": rows}, indent=2))
        return


if __name__ == "__main__":
    main()
