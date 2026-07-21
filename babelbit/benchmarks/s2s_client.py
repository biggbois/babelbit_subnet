"""Validator-shaped S2S HTTP client for miner benchmarks."""

from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import httpx

SAMPLE_RATE_HZ = 24_000
FRAME_SAMPLES = 1_920
FRAME_RATE_HZ = SAMPLE_RATE_HZ / FRAME_SAMPLES


@dataclass(frozen=True)
class S2sClientConfig:
    chunk_timeout_sec: float = 3.0
    drain_timeout_sec: float = 10.0
    drain_max_requests: int = 8
    final_drain_min_timeout_sec: float = 5.0
    init_timeout_sec: float = 600.0
    # Real validators pace ~1/frame_rate_hz. ASAP sends starve mid-stream ASR.
    pace_realtime: bool = False


@dataclass(frozen=True)
class S2sUtteranceRequest:
    frames: list[bytes]
    challenge_uid: str
    utterance_id: str
    validator_id: str = "validator-0"


@dataclass
class S2sUtteranceResult:
    validator_id: str
    challenge_uid: str
    utterance_id: str
    out_raw: bytearray = field(default_factory=bytearray)
    first_output_frame: int | None = None
    output_chunk_frames: list[int] = field(default_factory=list)
    output_chunks: list[bytes] = field(default_factory=list)
    wall_latency_sec: float = 0.0
    out_eos: bool = False
    out_eos_frame: int | None = None
    prediction_error: str | None = None
    timed_out_frames: list[int] = field(default_factory=list)
    max_frame_latency_sec: float = 0.0
    frames_sent: int = 0

    @property
    def completed(self) -> bool:
        return self.prediction_error is None and self.out_eos


def frame_response_timeout_sec(
    *,
    frame_index: int,
    total_frames: int,
    config: S2sClientConfig,
) -> float:
    if frame_index == total_frames - 1:
        return max(config.chunk_timeout_sec, config.drain_timeout_sec)
    return config.chunk_timeout_sec


def drain_deadline_sec(
    *,
    last_input_sent_at: float,
    drain_started_at: float,
    config: S2sClientConfig,
) -> float:
    return max(
        last_input_sent_at + config.drain_timeout_sec,
        drain_started_at + max(0.0, config.final_drain_min_timeout_sec),
    )


def format_chunk_timeout_error(*, frame_index: int, total_frames: int, timeout_sec: float) -> str:
    return (
        "AudioChallengeError:audio chunk response for frame "
        f"{frame_index + 1}/{total_frames} timed out after {timeout_sec:.2f}s"
    )


def run_s2s_utterance(
    base_url: str,
    request: S2sUtteranceRequest,
    *,
    config: S2sClientConfig | None = None,
    client: httpx.Client | None = None,
) -> S2sUtteranceResult:
    cfg = config or S2sClientConfig()
    predict_url = base_url.rstrip("/") + "/v1/predict"
    started = time.perf_counter()
    result = S2sUtteranceResult(
        validator_id=request.validator_id,
        challenge_uid=request.challenge_uid,
        utterance_id=request.utterance_id,
    )
    owns_client = client is None
    http = client or httpx.Client()
    try:
        init_timeout = httpx.Timeout(cfg.init_timeout_sec)
        init = http.post(
            predict_url,
            json={
                "kind": "init",
                "challenge_uid": request.challenge_uid,
                "utterance_id": request.utterance_id,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "frame_rate_hz": FRAME_RATE_HZ,
                "frame_samples": FRAME_SAMPLES,
                "dtype": "float32le",
                "channels": 1,
            },
            timeout=init_timeout,
        )
        if init.status_code >= 400:
            result.prediction_error = f"init failed: HTTP {init.status_code} {init.text[:200]}"
            result.wall_latency_sec = time.perf_counter() - started
            return result
        session_id = str(init.json()["session_id"])
        total_frames = len(request.frames)
        last_input_sent_at: float | None = None
        frame_period_sec = 1.0 / FRAME_RATE_HZ
        paced_t0 = time.perf_counter()

        for index, frame in enumerate(request.frames):
            if cfg.pace_realtime:
                target = paced_t0 + (index * frame_period_sec)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            in_eos = index == total_frames - 1
            timeout_sec = frame_response_timeout_sec(
                frame_index=index,
                total_frames=total_frames,
                config=cfg,
            )
            frame_started = time.perf_counter()
            try:
                response = http.post(
                    predict_url,
                    json={
                        "kind": "predict",
                        "session_id": session_id,
                        "audio_b64": base64.b64encode(frame).decode("ascii"),
                        "in_eos": in_eos,
                    },
                    timeout=httpx.Timeout(timeout_sec),
                )
            except httpx.TimeoutException:
                result.timed_out_frames.append(index + 1)
                result.prediction_error = format_chunk_timeout_error(
                    frame_index=index,
                    total_frames=total_frames,
                    timeout_sec=timeout_sec,
                )
                result.wall_latency_sec = time.perf_counter() - started
                result.frames_sent = index + 1
                return result

            frame_latency = time.perf_counter() - frame_started
            result.max_frame_latency_sec = max(result.max_frame_latency_sec, frame_latency)
            if response.status_code >= 400:
                result.prediction_error = (
                    f"predict failed frame {index + 1}/{total_frames}: "
                    f"HTTP {response.status_code} {response.text[:200]}"
                )
                result.wall_latency_sec = time.perf_counter() - started
                result.frames_sent = index + 1
                return result

            payload = response.json()
            n_bytes = int(payload.get("n_bytes") or 0)
            chunk = base64.b64decode(payload.get("audio_b64") or "")
            if n_bytes > 0:
                if result.first_output_frame is None:
                    result.first_output_frame = index + 1
                result.output_chunk_frames.append(index + 1)
                result.output_chunks.append(chunk)
            result.out_raw.extend(chunk)
            result.out_eos = bool(payload.get("out_eos"))
            result.frames_sent = index + 1
            if in_eos:
                last_input_sent_at = time.perf_counter()
            if result.out_eos:
                result.out_eos_frame = index + 1
                result.wall_latency_sec = time.perf_counter() - started
                return result

        if last_input_sent_at is None:
            result.prediction_error = "AudioChallengeError:missing final input chunk timestamp"
            result.wall_latency_sec = time.perf_counter() - started
            return result

        drain_started_at = time.perf_counter()
        deadline = drain_deadline_sec(
            last_input_sent_at=last_input_sent_at,
            drain_started_at=drain_started_at,
            config=cfg,
        )
        for drain_index in range(cfg.drain_max_requests):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                result.prediction_error = (
                    "AudioChallengeError:Miner did not signal out_eos before drain timeout was exhausted"
                )
                result.wall_latency_sec = time.perf_counter() - started
                return result
            try:
                response = http.post(
                    predict_url,
                    json={
                        "kind": "predict",
                        "session_id": session_id,
                        "audio_b64": "",
                        "in_eos": True,
                    },
                    timeout=httpx.Timeout(remaining),
                )
            except httpx.TimeoutException:
                result.prediction_error = (
                    "AudioChallengeError:drain response after final audio chunk timed out"
                )
                result.wall_latency_sec = time.perf_counter() - started
                return result

            if response.status_code >= 400:
                result.prediction_error = (
                    f"drain failed {drain_index + 1}/{cfg.drain_max_requests}: "
                    f"HTTP {response.status_code} {response.text[:200]}"
                )
                result.wall_latency_sec = time.perf_counter() - started
                return result

            payload = response.json()
            n_bytes = int(payload.get("n_bytes") or 0)
            chunk = base64.b64decode(payload.get("audio_b64") or "")
            if n_bytes > 0:
                if result.first_output_frame is None:
                    result.first_output_frame = total_frames + drain_index + 1
                result.output_chunk_frames.append(total_frames + drain_index + 1)
                result.output_chunks.append(chunk)
            result.out_raw.extend(chunk)
            result.out_eos = bool(payload.get("out_eos"))
            if result.out_eos:
                result.out_eos_frame = total_frames + drain_index + 1
                result.wall_latency_sec = time.perf_counter() - started
                return result

        result.prediction_error = (
            "AudioChallengeError:Miner did not signal out_eos before drain budget was exhausted"
        )
        result.wall_latency_sec = time.perf_counter() - started
        return result
    finally:
        if owns_client:
            http.close()


def run_s2s_concurrent_validators(
    base_url: str,
    requests: list[S2sUtteranceRequest],
    *,
    config: S2sClientConfig | None = None,
    max_workers: int | None = None,
) -> list[S2sUtteranceResult]:
    if not requests:
        return []
    worker_count = max(1, min(len(requests), max_workers or len(requests)))
    results: list[S2sUtteranceResult | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(run_s2s_utterance, base_url, req, config=config): idx
            for idx, req in enumerate(requests)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return [item for item in results if item is not None]
