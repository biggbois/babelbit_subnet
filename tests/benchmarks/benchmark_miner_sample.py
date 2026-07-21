#!/usr/bin/env python3
"""End-to-end miner benchmark with validator-accurate scoring.

Reads locale-specific miner-test-data JSON plus cached source WAV sidecars.
Does not translate EN references at runtime.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import os
import struct
import sys
import time
import wave
from pathlib import Path
from typing import Any

import httpx
import requests

from babelbit.benchmarks.miner_test_data import (
    estimate_min_source_duration_sec,
    load_miner_test_utterance,
    locale_sample_path,
    validate_source_audio_duration,
    wav_duration_sec,
    workspace_root_from,
)
from babelbit.benchmarks.s2s_client import (
    S2sClientConfig,
    S2sUtteranceRequest,
    run_s2s_concurrent_validators,
)
from babelbit.scoring.utterance_scoring import score_audio_utterance_batch

SAMPLE_RATE_HZ = 24_000
FRAME_SAMPLES = 1_920
FRAME_RATE_HZ = SAMPLE_RATE_HZ / FRAME_SAMPLES
MINER_SAMPLE_WIDTH_BYTES = 4
MINER_CHANNELS = 1
MINER_BYTES_PER_SEC = SAMPLE_RATE_HZ * MINER_CHANNELS * MINER_SAMPLE_WIDTH_BYTES


def playback_completion_sec(
    output_chunk_frames: list[int],
    output_chunks: list[bytes],
    frame_rate_hz: float,
    bytes_per_sec: float,
    eos_frame: int | None = None,
) -> float:
    """Mirrors predict_audio._playback_completion_sec for benchmark use."""
    if frame_rate_hz <= 0:
        return 0.0
    eos_sec = float(eos_frame or 0) / frame_rate_hz if eos_frame else 0.0
    if bytes_per_sec <= 0:
        return eos_sec
    playback_end_sec = 0.0
    for arrival_frame, chunk in zip(output_chunk_frames, output_chunks):
        arrival_sec = float(arrival_frame) / frame_rate_hz
        playback_end_sec = max(playback_end_sec, arrival_sec) + (
            len(chunk) / float(bytes_per_sec)
        )
    return max(playback_end_sec, eos_sec)


def subnet_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_miner_env_path() -> Path:
    return workspace_root_from(subnet_root()) / "babelbit_miner/.env"


def default_output_dir(*, locale: str, sample_stem: str) -> Path:
    return subnet_root() / "benchmark_outputs" / f"locale_{locale}" / sample_stem


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


def _asr_text(wav_bytes: bytes, *, language: str) -> str:
    response = requests.post(
        "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3",
        headers=_deepinfra_headers(),
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        data={"language": language},
        timeout=240,
    )
    response.raise_for_status()
    return str(response.json().get("text") or "").strip()


def _wav_to_float32_frames(wav_bytes: bytes) -> tuple[list[bytes], float]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if channels != 1 or width != 2 or rate != SAMPLE_RATE_HZ:
        raise RuntimeError(
            f"unexpected source wav format channels={channels} width={width} rate={rate}"
        )
    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    data = b"".join(
        struct.pack("<f", max(-1.0, min(1.0, sample / 32768.0)))
        for sample in samples
    )
    bytes_per_frame = FRAME_SAMPLES * 4
    frames = [data[i : i + bytes_per_frame] for i in range(0, len(data), bytes_per_frame)]
    if frames and len(frames[-1]) < bytes_per_frame:
        frames[-1] += b"\x00" * (bytes_per_frame - len(frames[-1]))
    return frames, len(samples) / SAMPLE_RATE_HZ


def _float32le_to_wav(raw: bytes) -> bytes:
    values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
    pcm = bytearray()
    for value in values:
        pcm.extend(
            int(max(-1.0, min(1.0, value)) * 32767).to_bytes(
                2, "little", signed=True
            )
        )
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(bytes(pcm))
    return out.getvalue()


def _run_miner_legacy(
    base_url: str,
    frames: list[bytes],
    *,
    challenge_uid: str,
    utterance_id: str,
) -> tuple[bytes, int | None, float, bool, str | None, list[int], float]:
    predict_url = base_url.rstrip("/") + "/v1/predict"
    started = time.perf_counter()
    out_raw = bytearray()
    first_output_frame: int | None = None
    out_eos = False
    with httpx.Client(timeout=240) as client:
        init = client.post(
            predict_url,
            json={
                "kind": "init",
                "challenge_uid": challenge_uid,
                "utterance_id": utterance_id,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "frame_rate_hz": FRAME_RATE_HZ,
                "frame_samples": FRAME_SAMPLES,
                "dtype": "float32le",
                "channels": 1,
            },
        )
        init.raise_for_status()
        session_id = str(init.json()["session_id"])
        for index, frame in enumerate(frames):
            body = {
                "kind": "predict",
                "session_id": session_id,
                "audio_b64": base64.b64encode(frame).decode("ascii"),
                "in_eos": index == len(frames) - 1,
            }
            response = client.post(predict_url, json=body)
            response.raise_for_status()
            result = response.json()
            if int(result.get("n_bytes") or 0) > 0 and first_output_frame is None:
                first_output_frame = index + 1
            out_raw.extend(base64.b64decode(result.get("audio_b64") or ""))
            out_eos = bool(result.get("out_eos"))
            if out_eos:
                break
        for _ in range(24):
            if out_eos:
                break
            response = client.post(
                predict_url,
                json={
                    "kind": "predict",
                    "session_id": session_id,
                    "audio_b64": "",
                    "in_eos": True,
                },
            )
            response.raise_for_status()
            result = response.json()
            if int(result.get("n_bytes") or 0) > 0 and first_output_frame is None:
                first_output_frame = len(frames)
            out_raw.extend(base64.b64decode(result.get("audio_b64") or ""))
            out_eos = bool(result.get("out_eos"))
    return bytes(out_raw), first_output_frame, time.perf_counter() - started, out_eos, None, [], 0.0


def _validator_challenge_uids(challenge_uid: str, concurrent_validators: int) -> list[str]:
    if concurrent_validators <= 1:
        return [challenge_uid]
    prefixes = ("challenge", "solo-challenge", "qualifying-challenge")
    uids = [f"{prefixes[idx % len(prefixes)]}-{challenge_uid}" for idx in range(concurrent_validators)]
    uids[0] = challenge_uid
    return uids


def _run_miner_validator(
    base_url: str,
    frames: list[bytes],
    *,
    challenge_uid: str,
    utterance_id: str,
    concurrent_validators: int,
    s2s_config: S2sClientConfig,
) -> tuple[bytes, int | None, float, bool, str | None, list[int], float, list[dict[str, Any]], list[int], list[bytes], int | None]:
    challenge_uids = _validator_challenge_uids(challenge_uid, concurrent_validators)
    requests = [
        S2sUtteranceRequest(
            frames=frames,
            challenge_uid=uid,
            utterance_id=utterance_id,
            validator_id=f"validator-{idx}",
        )
        for idx, uid in enumerate(challenge_uids)
    ]
    results = run_s2s_concurrent_validators(
        base_url,
        requests,
        config=s2s_config,
        max_workers=concurrent_validators,
    )
    concurrent_payload = [
        {
            "validator_id": item.validator_id,
            "challenge_uid": item.challenge_uid,
            "completed": item.completed,
            "prediction_error": item.prediction_error,
            "timed_out_frames": item.timed_out_frames,
            "max_frame_latency_sec": round(item.max_frame_latency_sec, 3),
            "wall_latency_sec": round(item.wall_latency_sec, 3),
            "out_eos": item.out_eos,
        }
        for item in results
    ]
    primary = results[0]
    return (
        bytes(primary.out_raw),
        primary.first_output_frame,
        primary.wall_latency_sec,
        primary.out_eos,
        primary.prediction_error,
        primary.timed_out_frames,
        primary.max_frame_latency_sec,
        concurrent_payload,
        list(primary.output_chunk_frames),
        list(primary.output_chunks),
        primary.out_eos_frame,
    )


def _run_miner(
    base_url: str,
    frames: list[bytes],
    *,
    challenge_uid: str,
    utterance_id: str,
    validator_mode: bool = False,
    concurrent_validators: int = 1,
    s2s_config: S2sClientConfig | None = None,
) -> dict[str, Any]:
    if validator_mode:
        (
            out_raw,
            first_output_frame,
            wall_latency_sec,
            out_eos,
            prediction_error,
            timed_out_frames,
            max_frame_latency_sec,
            concurrent_payload,
            output_chunk_frames,
            output_chunks,
            out_eos_frame,
        ) = _run_miner_validator(
            base_url,
            frames,
            challenge_uid=challenge_uid,
            utterance_id=utterance_id,
            concurrent_validators=concurrent_validators,
            s2s_config=s2s_config or S2sClientConfig(),
        )
        return {
            "out_raw": out_raw,
            "first_output_frame": first_output_frame,
            "output_chunk_frames": output_chunk_frames,
            "output_chunks": output_chunks,
            "out_eos_frame": out_eos_frame,
            "wall_latency_sec": wall_latency_sec,
            "out_eos": out_eos,
            "prediction_error": prediction_error,
            "timed_out_frames": timed_out_frames,
            "max_frame_latency_sec": max_frame_latency_sec,
            "concurrent_validators": concurrent_payload,
        }

    out_raw, first_output_frame, wall_latency_sec, out_eos, prediction_error, timed_out_frames, max_frame_latency_sec = (
        _run_miner_legacy(
            base_url,
            frames,
            challenge_uid=challenge_uid,
            utterance_id=utterance_id,
        )
    )
    return {
        "out_raw": out_raw,
        "first_output_frame": first_output_frame,
        "wall_latency_sec": wall_latency_sec,
        "out_eos": out_eos,
        "prediction_error": prediction_error,
        "timed_out_frames": timed_out_frames,
        "max_frame_latency_sec": max_frame_latency_sec,
        "concurrent_validators": [],
    }


def _validator_score(
    *,
    predicted_wav_bytes: bytes,
    utterance: Any,
    source_duration_sec: float,
    first_output_frame: int | None,
    completion_sec: float,
) -> dict[str, Any]:
    try:
        return score_audio_utterance_batch(
            predictions=[
                {
                    "predicted_wav_bytes": predicted_wav_bytes,
                    "first_output_frame": int(
                        first_output_frame or len(utterance.reference_text.split())
                    ),
                    "frame_rate_hz": FRAME_RATE_HZ,
                    "source_duration_sec": source_duration_sec,
                    "completion_sec": completion_sec,
                }
            ],
            challenge_uid=utterance.challenge_uid,
            utterance_id=utterance.utterance_id,
            source_duration_sec=source_duration_sec,
            target_lang=utterance.target_language,
            challenge_metadata=utterance.challenge_doc,
            metadata_source=str(utterance.sample_path),
            stt_model=os.environ.get("BB_AUDIO_SCORING_STT_MODEL", "faster-whisper-small"),
            stt_device=os.environ.get("BB_AUDIO_SCORING_STT_DEVICE", "cpu"),
        )[0]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_benchmark(
    *,
    base_url: str,
    sample_path: Path,
    out_dir: Path,
    skip_validator_score: bool,
    utterance_id: str = "0",
    validator_mode: bool = False,
    concurrent_validators: int = 1,
    s2s_config: S2sClientConfig | None = None,
) -> dict[str, Any]:
    utterance = load_miner_test_utterance(sample_path, utterance_id=utterance_id)
    ok, message = validate_source_audio_duration(
        utterance.source_text,
        utterance.source_audio_path,
    )
    if not ok:
        print(f"warning: {message}", file=sys.stderr, flush=True)
    source_wav = utterance.source_audio_path.read_bytes()
    frames, source_duration_sec = _wav_to_float32_frames(source_wav)
    print(
        f"[benchmark] u{utterance.utterance_id} start "
        f"source={source_duration_sec:.2f}s frames={len(frames)} "
        f"mode={'validator' if validator_mode else 'legacy'} "
        f"validators={concurrent_validators} → {base_url}",
        file=sys.stderr,
        flush=True,
    )
    miner_run = _run_miner(
        base_url,
        frames,
        challenge_uid=utterance.challenge_uid,
        utterance_id=utterance.utterance_id,
        validator_mode=validator_mode,
        concurrent_validators=concurrent_validators,
        s2s_config=s2s_config,
    )
    predicted_float32le = miner_run["out_raw"]
    first_output_frame = miner_run["first_output_frame"]
    wall_latency_sec = miner_run["wall_latency_sec"]
    out_eos = miner_run["out_eos"]
    prediction_error = miner_run.get("prediction_error")
    predicted_wav = _float32le_to_wav(predicted_float32le)
    output_asr = ""
    if predicted_wav and os.environ.get("DEEPINFRA_TOKEN"):
        try:
            output_asr = _asr_text(predicted_wav, language=utterance.target_language)
        except Exception:
            output_asr = ""
    rough_similarity = difflib.SequenceMatcher(
        None, output_asr.lower(), utterance.reference_text.lower()
    ).ratio()

    out_dir.mkdir(parents=True, exist_ok=True)
    source_wav_path = out_dir / "source.wav"
    output_wav_path = out_dir / "miner_output.wav"
    source_wav_path.write_bytes(source_wav)
    output_wav_path.write_bytes(predicted_wav)

    result: dict[str, Any] = {
        "sample": str(sample_path),
        "challenge_uid": utterance.challenge_uid,
        "utterance_id": utterance.utterance_id,
        "source_language": utterance.source_language,
        "target_language": utterance.target_language,
        "reference_en": utterance.reference_text,
        "source_text": utterance.source_text,
        "source_audio": str(utterance.source_audio_path),
        "output_asr": output_asr,
        "rough_similarity_not_validator_score": round(rough_similarity, 4),
        "source_duration_sec": round(source_duration_sec, 3),
        "wall_latency_sec": round(wall_latency_sec, 3),
        "frames_sent": len(frames),
        "first_output_frame": first_output_frame,
        "out_eos": out_eos,
        "prediction_error": prediction_error,
        "timed_out_frames": miner_run.get("timed_out_frames") or [],
        "max_frame_latency_sec": miner_run.get("max_frame_latency_sec"),
        "validator_mode": validator_mode,
        "concurrent_validators": miner_run.get("concurrent_validators") or [],
        "source_wav": str(source_wav_path),
        "output_wav": str(output_wav_path),
    }
    if prediction_error:
        result["validator_score"] = {
            "error": prediction_error,
            "accuracy": 0.0,
            "score": 0.0,
            "completed": False,
            "score_method": "prediction_error",
        }
        print(
            f"[benchmark] u{utterance.utterance_id} failed prediction_error={prediction_error}",
            file=sys.stderr,
            flush=True,
        )
    elif not skip_validator_score:
        output_chunk_frames = miner_run.get("output_chunk_frames")
        output_chunks = miner_run.get("output_chunks")
        out_eos_frame = miner_run.get("out_eos_frame")
        completion_sec = playback_completion_sec(
            output_chunk_frames=output_chunk_frames or [1],
            output_chunks=output_chunks or [predicted_float32le],
            frame_rate_hz=FRAME_RATE_HZ,
            bytes_per_sec=MINER_BYTES_PER_SEC,
            eos_frame=out_eos_frame,
        )
        result["validator_score"] = _validator_score(
            predicted_wav_bytes=predicted_wav,
            utterance=utterance,
            source_duration_sec=source_duration_sec,
            first_output_frame=first_output_frame,
            completion_sec=completion_sec,
        )
        vs = result.get("validator_score") if isinstance(result.get("validator_score"), dict) else {}
        lat = vs.get("latency") if isinstance(vs.get("latency"), dict) else {}
        print(
            f"[benchmark] u{utterance.utterance_id} done "
            f"wall={wall_latency_sec:.1f}s acc={vs.get('accuracy')} "
            f"prod={vs.get('score')} overshoot={lat.get('overshoot_sec')} eos={out_eos}",
            file=sys.stderr,
            flush=True,
        )

    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a miner using locale miner-test-data and cached source audio."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--locale", choices=["en", "fr", "de"], default="fr")
    parser.add_argument("--sample", type=Path, default=None, help="Locale JSON path or EN-relative path")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--miner-env", type=Path, default=None)
    parser.add_argument("--utterance-id", default="0")
    parser.add_argument("--skip-validator-score", action="store_true")
    parser.add_argument(
        "--validator-mode",
        action="store_true",
        help="Use validator S2S envelope: 3s chunk timeout, drain loop, optional concurrent validators.",
    )
    parser.add_argument(
        "--concurrent-validators",
        type=int,
        default=1,
        help="Parallel validator sessions per utterance (default 1). Use 3 to mimic production load.",
    )
    parser.add_argument(
        "--chunk-timeout-sec",
        type=float,
        default=3.0,
        help="Per-frame response timeout in validator mode (matches BB_S2S_CHUNK_TIMEOUT_SEC).",
    )
    args = parser.parse_args()

    workspace_root = workspace_root_from(subnet_root())
    miner_env = args.miner_env or default_miner_env_path()
    _load_env(miner_env)

    if args.sample is None:
        sample_path = locale_sample_path(workspace_root, locale=args.locale)
    else:
        sample_path = args.sample
        if not sample_path.is_file():
            sample_path = locale_sample_path(
                workspace_root,
                locale=args.locale,
                relative_en_path=str(args.sample),
            )

    out_dir = args.out_dir or default_output_dir(
        locale=args.locale,
        sample_stem=sample_path.stem,
    )
    result = run_benchmark(
        base_url=args.base_url,
        sample_path=sample_path,
        out_dir=out_dir,
        skip_validator_score=args.skip_validator_score,
        utterance_id=args.utterance_id,
        validator_mode=args.validator_mode,
        concurrent_validators=max(1, args.concurrent_validators),
        s2s_config=S2sClientConfig(chunk_timeout_sec=max(0.001, args.chunk_timeout_sec)),
    )
    print(json.dumps(result, indent=2))
    if result.get("prediction_error"):
        return 2
    return 0 if result.get("out_eos") else 2


if __name__ == "__main__":
    raise SystemExit(main())
