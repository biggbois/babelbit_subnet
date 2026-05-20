from __future__ import annotations

import argparse
import io
import json
import math
import random
import resource
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from babelbit.scoring.stt import (
    STT_MODEL_TABLE,
    _wav_bytes_to_whisper_input,
    validate_stt_model,
)
from babelbit.utils.hf_runtime import ensure_hf_transfer_available


_CPU_COMPUTE_TYPE = {"float16": "int8"}
_MODEL_CACHE: dict[str, Any] = {}


def _write_wav(
    *,
    samples: np.ndarray,
    sample_rate_hz: int,
) -> bytes:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


def _audio_samples(
    *,
    kind: str,
    duration_sec: float,
    sample_rate_hz: int,
    seed: int,
) -> np.ndarray:
    frame_count = max(1, int(duration_sec * sample_rate_hz))
    rng = random.Random(seed)
    t = np.arange(frame_count, dtype=np.float32) / float(sample_rate_hz)

    if kind == "silence":
        return np.zeros(frame_count, dtype=np.float32)

    if kind == "noise":
        return np.asarray(
            [rng.uniform(-0.18, 0.18) for _ in range(frame_count)],
            dtype=np.float32,
        )

    if kind == "tone":
        frequency_hz = 220.0 + float(seed % 400)
        return (0.2 * np.sin(2.0 * math.pi * frequency_hz * t)).astype(np.float32)

    if kind == "chirp":
        start_hz = 180.0 + float(seed % 100)
        end_hz = 900.0 + float(seed % 300)
        sweep = start_hz + (end_hz - start_hz) * (t / max(duration_sec, 0.001))
        return (0.18 * np.sin(2.0 * math.pi * sweep * t)).astype(np.float32)

    if kind == "mixed":
        tone = _audio_samples(
            kind="chirp",
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
        )
        noise = _audio_samples(
            kind="noise",
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
            seed=seed + 100_000,
        )
        return np.clip(tone + (noise * 0.35), -1.0, 1.0).astype(np.float32)

    raise ValueError(f"Unknown audio kind: {kind}")


def _load_wavs_from_dir(path: Path) -> list[bytes]:
    wav_paths = sorted(path.glob("*.wav"))
    if not wav_paths:
        raise ValueError(f"No .wav files found in {path}")
    return [wav_path.read_bytes() for wav_path in wav_paths]


def _build_wavs(
    *,
    wav_dir: Path | None,
    miner_count: int,
    duration_sec: float,
    sample_rate_hz: int,
    audio_kind: str,
) -> list[bytes]:
    if wav_dir is not None:
        source_wavs = _load_wavs_from_dir(wav_dir)
        return [source_wavs[i % len(source_wavs)] for i in range(miner_count)]

    return [
        _write_wav(
            samples=_audio_samples(
                kind=audio_kind,
                duration_sec=duration_sec,
                sample_rate_hz=sample_rate_hz,
                seed=(miner_count * 10_000) + i,
            ),
            sample_rate_hz=sample_rate_hz,
        )
        for i in range(miner_count)
    ]


def _max_rss_mb() -> float:
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_kb / 1024.0


def _transcribe_one(
    wav_bytes: bytes,
    *,
    stt_model: str,
    language: str,
    device: str,
    cpu_threads: int,
) -> dict[str, Any]:
    validate_stt_model(stt_model)
    backend, model_id, compute_type = STT_MODEL_TABLE[stt_model]
    if backend != "faster-whisper":
        raise ValueError(f"Unsupported STT benchmark backend: {backend}")
    if device == "cpu" and compute_type in _CPU_COMPUTE_TYPE:
        compute_type = _CPU_COMPUTE_TYPE[compute_type]

    decode_started = time.perf_counter()
    audio_input = _wav_bytes_to_whisper_input(wav_bytes)
    decode_wall_sec = time.perf_counter() - decode_started

    cache_key = f"{model_id}:{device}:{compute_type}:{cpu_threads}"
    if cache_key not in _MODEL_CACHE:
        ensure_hf_transfer_available()
        from faster_whisper import WhisperModel

        kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
        }
        if cpu_threads > 0:
            kwargs["cpu_threads"] = cpu_threads
        _MODEL_CACHE[cache_key] = WhisperModel(model_id, **kwargs)

    model = _MODEL_CACHE[cache_key]
    stt_started = time.perf_counter()
    segments, _info = model.transcribe(
        audio_input,
        language=language,
        word_timestamps=True,
    )
    words = []
    text_parts = []
    for segment in segments:
        text_parts.append(str(segment.text).strip())
        if not getattr(segment, "words", None):
            continue
        for word in segment.words:
            words.append(word)
    text = " ".join(part for part in text_parts if part).strip()
    stt_wall_sec = time.perf_counter() - stt_started
    audio_sec = float(audio_input.shape[0]) / 16_000.0

    return {
        "audio_sec": audio_sec,
        "decode_wall_sec": decode_wall_sec,
        "stt_wall_sec": stt_wall_sec,
        "text_chars": len(text),
        "word_count": len(words),
    }


def _run_once(
    *,
    miner_count: int,
    duration_sec: float,
    sample_rate_hz: int,
    audio_kind: str,
    wav_dir: Path | None,
    stt_model: str,
    language: str,
    device: str,
    cpu_threads: int,
) -> dict[str, Any]:
    wavs = _build_wavs(
        wav_dir=wav_dir,
        miner_count=miner_count,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        audio_kind=audio_kind,
    )

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    item_results = [
        _transcribe_one(
            wav_bytes,
            stt_model=stt_model,
            language=language,
            device=device,
            cpu_threads=cpu_threads,
        )
        for wav_bytes in wavs
    ]
    wall_sec = time.perf_counter() - wall_started
    cpu_sec = time.process_time() - cpu_started

    audio_sec = sum(float(row["audio_sec"]) for row in item_results)
    decode_wall_sec = sum(float(row["decode_wall_sec"]) for row in item_results)
    stt_wall_sec = sum(float(row["stt_wall_sec"]) for row in item_results)
    word_count = sum(int(row["word_count"]) for row in item_results)
    text_chars = sum(int(row["text_chars"]) for row in item_results)

    return {
        "miners": miner_count,
        "audio_kind": audio_kind if wav_dir is None else "wav_dir",
        "duration_sec": duration_sec,
        "total_audio_sec": audio_sec,
        "wall_sec": wall_sec,
        "cpu_sec": cpu_sec,
        "decode_wall_sec": decode_wall_sec,
        "stt_wall_sec": stt_wall_sec,
        "cpu_to_wall": cpu_sec / wall_sec if wall_sec > 0 else None,
        "stt_realtime_factor": stt_wall_sec / audio_sec if audio_sec > 0 else None,
        "audio_sec_per_wall_sec": audio_sec / wall_sec if wall_sec > 0 else None,
        "miners_per_wall_sec": miner_count / wall_sec if wall_sec > 0 else None,
        "word_count": word_count,
        "text_chars": text_chars,
        "max_rss_mb": _max_rss_mb(),
        "cpu_threads": cpu_threads,
    }


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(samples[-1])
    for key in (
        "total_audio_sec",
        "wall_sec",
        "cpu_sec",
        "decode_wall_sec",
        "stt_wall_sec",
        "cpu_to_wall",
        "stt_realtime_factor",
        "audio_sec_per_wall_sec",
        "miners_per_wall_sec",
        "max_rss_mb",
    ):
        values = [float(sample[key]) for sample in samples if sample[key] is not None]
        summary[f"{key}_median"] = statistics.median(values)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    return summary


def _print_header() -> None:
    print(
        "miners audio_kind duration_sec audio_sec wall_sec cpu_sec "
        "stt_sec stt_x_realtime audio_sec_per_wall_sec miners_per_wall_sec "
        "cpu_threads words text_chars max_rss_mb",
        flush=True,
    )


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['miners']} {row['audio_kind']} {row['duration_sec']:.2f} "
        f"{row['total_audio_sec_median']:.2f} {row['wall_sec_median']:.2f} "
        f"{row['cpu_sec_median']:.2f} {row['stt_wall_sec_median']:.2f} "
        f"{row['stt_realtime_factor_median']:.2f} "
        f"{row['audio_sec_per_wall_sec_median']:.2f} "
        f"{row['miners_per_wall_sec_median']:.2f} "
        f"{row['cpu_threads']} {row['word_count']} {row['text_chars']} "
        f"{row['max_rss_mb_median']:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test faster-whisper STT throughput without scoring."
    )
    parser.add_argument("--miners", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--durations-sec", nargs="+", type=float, default=[2.0])
    parser.add_argument(
        "--audio-kind",
        choices=["silence", "noise", "tone", "chirp", "mixed"],
        default="mixed",
    )
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--sample-rate-hz", type=int, default=24_000)
    parser.add_argument("--stt-model", default="faster-whisper-tiny")
    parser.add_argument("--stt-device", default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="faster-whisper CPU threads. 0 uses the library default.",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = []
    if not args.json:
        _print_header()

    for duration_sec in args.durations_sec:
        for miner_count in args.miners:
            samples = []
            for run_index in range(args.warmup + args.repeat):
                sample = _run_once(
                    miner_count=miner_count,
                    duration_sec=duration_sec,
                    sample_rate_hz=args.sample_rate_hz,
                    audio_kind=args.audio_kind,
                    wav_dir=args.wav_dir,
                    stt_model=args.stt_model,
                    language=args.language,
                    device=args.stt_device,
                    cpu_threads=args.cpu_threads,
                )
                if run_index >= args.warmup:
                    samples.append(sample)
            row = _summarize(samples)
            rows.append(row)
            if not args.json:
                _print_row(row)

    if args.json:
        print(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    main()
