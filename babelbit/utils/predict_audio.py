import asyncio
import base64
import io
import wave
from dataclasses import dataclass
from logging import getLogger
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from babelbit.scoring.utterance_scoring import score_audio_utterance_batch
from babelbit.schemas.audio_prediction import (
    BBAudioChallengeResult,
    BBAudioMinerInitPayload,
    BBAudioMinerInitResponse,
    BBAudioMinerPredictPayload,
    BBAudioMinerPredictResponse,
    BBAudioUEUtterance,
    BBAudioUtteranceResult,
)
from babelbit.scoring.reference_metadata import (
    AudioReferenceMetadata,
    resolve_audio_reference_metadata,
)
from babelbit.utils.async_clients import get_async_client
from babelbit.utils.settings import get_settings
from babelbit.utils.utterance_auth import (
    authenticate_utterance_engine,
    get_auth_headers,
)

logger = getLogger(__name__)

_MINER_AUDIO_DTYPE = "float32le"
_MINER_AUDIO_SAMPLE_RATE_HZ = 24_000
_MINER_AUDIO_SAMPLE_WIDTH_BYTES = 4
_MINER_AUDIO_FRAME_RATE_HZ = 12.5
_MINER_AUDIO_FRAME_SAMPLES = int(
    _MINER_AUDIO_SAMPLE_RATE_HZ / _MINER_AUDIO_FRAME_RATE_HZ
)


class AudioChallengeError(Exception):
    pass


@dataclass
class _DecodedWav:
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    pcm_bytes: bytes


@dataclass
class _MinerAudio:
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    dtype: str
    frame_count: int
    pcm_bytes: bytes


def _resample_audio_frames(
    audio: np.ndarray,
    *,
    source_sample_rate_hz: int,
    target_sample_rate_hz: int,
) -> np.ndarray:
    if source_sample_rate_hz == target_sample_rate_hz:
        return np.ascontiguousarray(audio.astype(np.float32, copy=False))
    if audio.size == 0:
        return np.zeros((0, audio.shape[1]), dtype=np.float32)

    source_frames = audio.shape[0]
    target_frames = max(
        1,
        int(
            round(
                source_frames
                * float(target_sample_rate_hz)
                / float(source_sample_rate_hz)
            )
        ),
    )
    if source_frames == 1:
        return np.repeat(audio.astype(np.float32, copy=False), target_frames, axis=0)

    source_positions = np.linspace(0.0, source_frames - 1, num=source_frames)
    target_positions = np.linspace(0.0, source_frames - 1, num=target_frames)
    channels = [
        np.interp(target_positions, source_positions, audio[:, channel])
        for channel in range(audio.shape[1])
    ]
    resampled = np.stack(channels, axis=1)
    return np.ascontiguousarray(resampled.astype(np.float32, copy=False))


def _miner_log_label(miner: Any) -> str:
    uid = int(getattr(miner, "uid", -1))
    hotkey = str(getattr(miner, "hotkey", ""))
    if hotkey:
        return f"uid={uid} hotkey={hotkey}"
    return f"uid={uid}"


@dataclass
class _RawPrediction:
    miner: Any
    ue_utterance: BBAudioUEUtterance
    decoded_audio: _DecodedWav
    source_audio_bytes: bytes
    predicted_wav: bytes
    first_output_frame: int
    frame_rate_hz: float
    frame_samples: int
    frame_count_in: int
    frame_count_out: int
    session_id: str
    latency_ms: float
    completion_sec: float
    error: Optional[str] = None


@dataclass
class _MinerUtteranceSession:
    miner: Any
    ue_utterance: BBAudioUEUtterance
    decoded_audio: _DecodedWav
    miner_audio: _MinerAudio
    source_audio_bytes: bytes
    started_at: float
    session_id: str
    frame_rate_hz: float
    frame_samples: int
    input_frames: List[bytes]
    output_chunks: List[bytes]
    first_output_frame: Optional[int] = None
    saw_out_eos: bool = False
    last_input_chunk_sent_at: Optional[float] = None
    completed_frame: Optional[int] = None


def _build_miner_init_payload(
    *,
    ue_utterance: BBAudioUEUtterance,
    miner_audio: _MinerAudio,
) -> BBAudioMinerInitPayload:
    frame_samples = _default_frame_samples(miner_audio.sample_rate_hz)
    return BBAudioMinerInitPayload(
        challenge_uid=ue_utterance.challenge_uid,
        utterance_id=ue_utterance.utterance_id,
        language=ue_utterance.language,
        sample_rate_hz=miner_audio.sample_rate_hz,
        frame_samples=frame_samples,
        frame_rate_hz=_default_frame_rate_hz(
            miner_audio.sample_rate_hz,
            frame_samples,
        ),
        dtype=miner_audio.dtype,
        channels=miner_audio.channels,
    )


async def _await_predict_response(
    *,
    miner: Any,
    payload: BBAudioMinerPredictPayload,
    predict_callback: Callable[
        [Any, BBAudioMinerPredictPayload], Awaitable[Dict[str, Any]]
    ],
    timeout_seconds: float,
    timeout_label: str,
) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(
            predict_callback(miner, payload), timeout=timeout_seconds
        )
    except asyncio.TimeoutError as exc:
        raise AudioChallengeError(
            f"{timeout_label} timed out after {timeout_seconds:.2f}s"
        ) from exc


async def _request_with_reauth(
    session,
    method: str,
    url: str,
    *,
    json_payload: Optional[dict] = None,
    allow_retry: bool = True,
) -> tuple[int, dict[str, Any] | str]:
    headers = await get_auth_headers()
    request_kwargs = {"headers": headers}
    if json_payload is not None:
        request_kwargs["json"] = json_payload

    caller = getattr(session, method.lower(), None)
    if caller is None:
        caller = session.request

    async with (
        caller(method, url, **request_kwargs)
        if caller is session.request
        else caller(url, **request_kwargs) as response
    ):
        if response.status == 401 and allow_retry:
            logger.warning(
                "Utterance engine returned 401 during audio flow; refreshing auth and retrying once."
            )
            await authenticate_utterance_engine()
            return await _request_with_reauth(
                session,
                method,
                url,
                json_payload=json_payload,
                allow_retry=False,
            )

        try:
            data = await response.json()
        except Exception:
            data = await response.text()
        return response.status, data


async def _call_engine_json(
    session,
    method: str,
    url: str,
    *,
    payload: Optional[dict] = None,
    error_label: str,
) -> Dict[str, Any]:
    status, data = await _request_with_reauth(
        session, method, url, json_payload=payload
    )
    if status != 200:
        raise AudioChallengeError(f"Failed to {error_label}: HTTP {status}")
    if not isinstance(data, dict):
        raise AudioChallengeError(f"Failed to {error_label}: invalid payload type")
    return data


async def start_source_audio_session(utterance_engine_url: str) -> Dict[str, Any]:
    session = await get_async_client()
    return await _call_engine_json(
        session,
        "POST",
        f"{utterance_engine_url}/source-audio/start",
        error_label="start source-audio session",
    )


async def start_solo_audio_session(utterance_engine_url: str) -> Dict[str, Any]:
    session = await get_async_client()
    return await _call_engine_json(
        session,
        "GET",
        f"{utterance_engine_url}/solo/start",
        error_label="start solo-audio session",
    )


async def next_source_audio_utterance(
    utterance_engine_url: str, session_id: str
) -> Dict[str, Any]:
    session = await get_async_client()
    return await _call_engine_json(
        session,
        "POST",
        f"{utterance_engine_url}/source-audio/next",
        payload={"session_id": session_id},
        error_label="advance source-audio session",
    )


async def next_solo_audio_utterance(
    utterance_engine_url: str, session_id: str
) -> Dict[str, Any]:
    session = await get_async_client()
    return await _call_engine_json(
        session,
        "POST",
        f"{utterance_engine_url}/solo/next",
        payload={"session_id": session_id},
        error_label="advance solo-audio session",
    )


async def fetch_transcription_ground_truth(utterance_engine_url: str) -> Dict[str, Any]:
    session = await get_async_client()
    return await _call_engine_json(
        session,
        "GET",
        f"{utterance_engine_url}/transcription",
        error_label="fetch transcription ground truth",
    )


def _response_to_ue_utterance(
    response_data: Dict[str, Any],
) -> Optional[BBAudioUEUtterance]:
    audio_b64 = str(response_data.get("audio_b64") or "")
    if not audio_b64:
        if bool(response_data.get("done", False)):
            return None
        raise AudioChallengeError(
            "Utterance engine returned an empty audio payload for an active utterance"
        )
    return BBAudioUEUtterance.model_validate(response_data)


def _inline_metadata_from_ue_payloads(
    *, challenge_uid: str, payloads: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    utterances: List[Dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict) or not payload.get("audio_b64"):
            continue

        raw_utterance: Dict[str, Any] = {
            "utterance_id": payload.get(
                "utterance_id", payload.get("utterance_index", index)
            ),
            "utterance_index": int(payload.get("utterance_index", index) or index),
        }
        if isinstance(payload.get("target"), dict):
            raw_utterance["target"] = payload["target"]
        if "reference_text" in payload:
            raw_utterance["reference_text"] = payload.get("reference_text")
            if "reference_words" in payload:
                raw_utterance["reference_words"] = payload.get("reference_words")
            if "reference_wps" in payload:
                raw_utterance["reference_wps"] = payload.get("reference_wps")
        if isinstance(payload.get("utterance_translations"), list):
            raw_utterance["utterance_translations"] = payload[
                "utterance_translations"
            ]

        if (
            "target" in raw_utterance
            or "reference_text" in raw_utterance
            or "utterance_translations" in raw_utterance
        ):
            utterances.append(raw_utterance)

    if not utterances:
        return None
    return {"challenge_uid": challenge_uid, "utterances": utterances}


def _sample_width_to_dtype(sample_width_bytes: int) -> str:
    mapping = {
        1: "int8",
        2: "int16",
        4: "int32",
    }
    dtype = mapping.get(int(sample_width_bytes))
    if not dtype:
        raise AudioChallengeError(f"Unsupported sample width: {sample_width_bytes}")
    return dtype


def _decoded_audio_to_miner_audio(decoded_audio: _DecodedWav) -> _MinerAudio:
    dtype_map: dict[int, np.dtype] = {
        1: np.dtype("u1"),
        2: np.dtype("<i2"),
        4: np.dtype("<i4"),
    }
    dtype = dtype_map.get(decoded_audio.sample_width_bytes)
    if dtype is None:
        raise AudioChallengeError(
            f"Unsupported sample width: {decoded_audio.sample_width_bytes}"
        )

    samples = np.frombuffer(decoded_audio.pcm_bytes, dtype=dtype)
    if samples.size % decoded_audio.channels != 0:
        raise AudioChallengeError(
            "Decoded UE audio sample count was not divisible by channel count"
        )
    if decoded_audio.sample_width_bytes == 1:
        normalized = (samples.astype(np.float32) - 128.0) / 128.0
    elif decoded_audio.sample_width_bytes == 2:
        normalized = samples.astype(np.float32) / 32768.0
    else:
        normalized = samples.astype(np.float32) / 2147483648.0

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    normalized = np.clip(normalized, -1.0, 1.0)
    normalized_frames = normalized.reshape(-1, decoded_audio.channels)
    miner_frames = _resample_audio_frames(
        normalized_frames,
        source_sample_rate_hz=decoded_audio.sample_rate_hz,
        target_sample_rate_hz=_MINER_AUDIO_SAMPLE_RATE_HZ,
    )
    float32_bytes = np.ascontiguousarray(
        miner_frames.reshape(-1).astype("<f4", copy=False)
    ).tobytes()
    return _MinerAudio(
        sample_rate_hz=_MINER_AUDIO_SAMPLE_RATE_HZ,
        channels=decoded_audio.channels,
        sample_width_bytes=_MINER_AUDIO_SAMPLE_WIDTH_BYTES,
        dtype=_MINER_AUDIO_DTYPE,
        frame_count=miner_frames.shape[0],
        pcm_bytes=float32_bytes,
    )


def _float32_pcm_to_int16_pcm(float32_pcm_bytes: bytes) -> bytes:
    if not float32_pcm_bytes:
        return b""
    if len(float32_pcm_bytes) % _MINER_AUDIO_SAMPLE_WIDTH_BYTES != 0:
        raise AudioChallengeError(
            "Miner returned float32 audio with a non-sample-aligned byte length"
        )

    samples = np.frombuffer(float32_pcm_bytes, dtype=np.dtype("<f4"))
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    samples = np.clip(samples, -1.0, 1.0)
    scaled = np.where(samples < 0, samples * 32768.0, samples * 32767.0)
    return np.ascontiguousarray(scaled.astype("<i2")).tobytes()


def _decode_audio_bytes(
    audio_bytes: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
    sample_width_bytes: int,
    utterance_frames: int,
) -> _DecodedWav:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            decoded = _DecodedWav(
                sample_rate_hz=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                sample_width_bytes=wav_file.getsampwidth(),
                frame_count=wav_file.getnframes(),
                pcm_bytes=wav_file.readframes(wav_file.getnframes()),
            )
    except Exception:
        expected_num_bytes = utterance_frames * channels * sample_width_bytes
        if len(audio_bytes) != expected_num_bytes:
            raise AudioChallengeError(
                "Failed to decode UE audio payload as WAV and raw PCM size did not match "
                f"metadata: expected {expected_num_bytes} bytes, got {len(audio_bytes)}"
            )
        decoded = _DecodedWav(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_width_bytes=sample_width_bytes,
            frame_count=utterance_frames,
            pcm_bytes=audio_bytes,
        )

    if decoded.sample_rate_hz != sample_rate_hz:
        raise AudioChallengeError(
            f"UE audio payload sample_rate_hz {decoded.sample_rate_hz} did not match metadata {sample_rate_hz}"
        )
    if decoded.channels != channels:
        raise AudioChallengeError(
            f"UE audio payload channels {decoded.channels} did not match metadata {channels}"
        )
    if decoded.sample_width_bytes != sample_width_bytes:
        raise AudioChallengeError(
            "UE audio payload sample_width_bytes "
            f"{decoded.sample_width_bytes} did not match metadata {sample_width_bytes}"
        )
    if decoded.frame_count != utterance_frames:
        raise AudioChallengeError(
            f"UE audio payload frames {decoded.frame_count} did not match metadata {utterance_frames}"
        )

    return decoded


def _pcm_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
    sample_width_bytes: int,
) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width_bytes)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm_bytes)
    return output.getvalue()


def _default_frame_samples(sample_rate_hz: int) -> int:
    return _MINER_AUDIO_FRAME_SAMPLES


def _default_frame_rate_hz(sample_rate_hz: int, frame_samples: int) -> float:
    return _MINER_AUDIO_FRAME_RATE_HZ


def _split_pcm_into_frames(
    pcm_bytes: bytes,
    *,
    frame_samples: int,
    channels: int,
    sample_width_bytes: int,
) -> List[bytes]:
    bytes_per_sample_frame = channels * sample_width_bytes
    bytes_per_chunk = frame_samples * bytes_per_sample_frame
    if bytes_per_chunk <= 0:
        raise AudioChallengeError("Computed non-positive bytes_per_chunk")

    chunks: List[bytes] = []
    for offset in range(0, len(pcm_bytes), bytes_per_chunk):
        chunk = pcm_bytes[offset : offset + bytes_per_chunk]
        if len(chunk) < bytes_per_chunk:
            chunk = chunk + (b"\x00" * (bytes_per_chunk - len(chunk)))
        chunks.append(chunk)
    if not chunks:
        chunks.append(b"\x00" * bytes_per_chunk)
    return chunks


def _validate_init_response(
    response_data: Dict[str, Any],
    *,
    challenge_uid: str,
    utterance_id: str,
    sample_rate_hz: int,
    channels: int,
    dtype: str,
) -> BBAudioMinerInitResponse:
    response = BBAudioMinerInitResponse.model_validate(response_data)
    if not response.ready:
        raise AudioChallengeError("Miner init returned ready=false")
    if response.challenge_uid != challenge_uid:
        raise AudioChallengeError("Miner init returned mismatched challenge_uid")
    if response.utterance_id != utterance_id:
        raise AudioChallengeError("Miner init returned mismatched utterance_id")
    if response.sample_rate_hz != sample_rate_hz:
        raise AudioChallengeError(
            f"Miner requested unsupported sample_rate_hz {response.sample_rate_hz}; expected {sample_rate_hz}"
        )
    if response.channels != channels:
        raise AudioChallengeError(
            f"Miner requested unsupported channel count {response.channels}; expected {channels}"
        )
    if response.dtype != dtype:
        raise AudioChallengeError(
            f"Miner requested unsupported dtype {response.dtype}; expected {dtype}"
        )
    if response.frame_samples <= 0:
        raise AudioChallengeError("Miner returned non-positive frame_samples")
    if response.frame_rate_hz <= 0:
        raise AudioChallengeError("Miner returned non-positive frame_rate_hz")
    return response


def _prediction_failure(
    *,
    miner: Any,
    ue_utterance: BBAudioUEUtterance,
    decoded_audio: _DecodedWav,
    source_audio_bytes: bytes,
    started_at: float,
    exc: BaseException,
) -> _RawPrediction:
    latency_ms = (perf_counter() - started_at) * 1000.0
    logger.warning(
        "S2S audio miner failed: challenge=%s utterance=%s %s error=%s latency_ms=%.2f",
        ue_utterance.challenge_uid,
        ue_utterance.utterance_id,
        _miner_log_label(miner),
        f"{type(exc).__name__}:{exc}",
        latency_ms,
    )
    return _RawPrediction(
        miner=miner,
        ue_utterance=ue_utterance,
        decoded_audio=decoded_audio,
        source_audio_bytes=source_audio_bytes,
        predicted_wav=b"",
        first_output_frame=0,
        frame_rate_hz=0.0,
        frame_samples=0,
        frame_count_in=0,
        frame_count_out=0,
        session_id="",
        latency_ms=latency_ms,
        completion_sec=latency_ms / 1000.0,
        error=f"{type(exc).__name__}:{exc}",
    )


def _miner_output_frame_count(
    output_chunks: List[bytes],
    *,
    frame_samples: int,
    channels: int,
    sample_width_bytes: int,
) -> int:
    if frame_samples <= 0 or channels <= 0 or sample_width_bytes <= 0:
        return 0
    bytes_per_frame = frame_samples * channels * sample_width_bytes
    if bytes_per_frame <= 0:
        return 0
    return sum(len(chunk) for chunk in output_chunks) // bytes_per_frame


def _prediction_success(session: _MinerUtteranceSession) -> _RawPrediction:
    if not session.saw_out_eos:
        raise AudioChallengeError("Miner never signaled out_eos=true")
    if session.completed_frame is None:
        raise AudioChallengeError("Miner completion frame was not recorded")

    predicted_pcm = _float32_pcm_to_int16_pcm(b"".join(session.output_chunks))
    predicted_wav = _pcm_to_wav_bytes(
        predicted_pcm,
        sample_rate_hz=session.miner_audio.sample_rate_hz,
        channels=session.miner_audio.channels,
        sample_width_bytes=2,
    )
    completion_sec = (
        float(session.completed_frame) / session.frame_rate_hz
        if session.frame_rate_hz > 0
        else 0.0
    )
    latency_ms = completion_sec * 1000.0
    output_frame_count = _miner_output_frame_count(
        session.output_chunks,
        frame_samples=session.frame_samples,
        channels=session.miner_audio.channels,
        sample_width_bytes=session.miner_audio.sample_width_bytes,
    )
    logger.info(
        "S2S audio miner completed: challenge=%s utterance=%s %s frames_in=%d frames_out=%d predicted_bytes=%d latency_ms=%.2f",
        session.ue_utterance.challenge_uid,
        session.ue_utterance.utterance_id,
        _miner_log_label(session.miner),
        len(session.input_frames),
        output_frame_count,
        len(predicted_wav),
        latency_ms,
    )
    return _RawPrediction(
        miner=session.miner,
        ue_utterance=session.ue_utterance,
        decoded_audio=session.decoded_audio,
        source_audio_bytes=session.source_audio_bytes,
        predicted_wav=predicted_wav,
        first_output_frame=session.first_output_frame or 0,
        frame_rate_hz=session.frame_rate_hz,
        frame_samples=session.frame_samples,
        frame_count_in=len(session.input_frames),
        frame_count_out=output_frame_count,
        session_id=session.session_id,
        latency_ms=latency_ms,
        completion_sec=completion_sec,
    )


async def _start_utterance_for_miner(
    *,
    miner: Any,
    ue_utterance: BBAudioUEUtterance,
    decoded_audio: _DecodedWav,
    source_audio_bytes: bytes,
    init_callback: Callable[[Any, BBAudioMinerInitPayload], Awaitable[Dict[str, Any]]],
) -> _MinerUtteranceSession:
    miner_audio = _decoded_audio_to_miner_audio(decoded_audio)
    init_payload = _build_miner_init_payload(
        ue_utterance=ue_utterance,
        miner_audio=miner_audio,
    )
    started_at = perf_counter()
    init_response_data = await init_callback(miner, init_payload)
    init_response = _validate_init_response(
        init_response_data,
        challenge_uid=ue_utterance.challenge_uid,
        utterance_id=ue_utterance.utterance_id,
        sample_rate_hz=miner_audio.sample_rate_hz,
        channels=miner_audio.channels,
        dtype=miner_audio.dtype,
    )
    logger.info(
        "S2S audio miner init accepted: challenge=%s utterance=%s %s session=%s frame_samples=%d frame_rate_hz=%.2f",
        ue_utterance.challenge_uid,
        ue_utterance.utterance_id,
        _miner_log_label(miner),
        init_response.session_id,
        init_response.frame_samples,
        init_response.frame_rate_hz,
    )
    if (
        init_response.frame_samples != init_payload.frame_samples
        or abs(init_response.frame_rate_hz - init_payload.frame_rate_hz) > 1e-6
    ):
        logger.warning(
            "S2S audio miner init cadence override ignored: challenge=%s utterance=%s %s requested_frame_samples=%d requested_frame_rate_hz=%.2f response_frame_samples=%d response_frame_rate_hz=%.2f",
            ue_utterance.challenge_uid,
            ue_utterance.utterance_id,
            _miner_log_label(miner),
            init_payload.frame_samples,
            init_payload.frame_rate_hz,
            init_response.frame_samples,
            init_response.frame_rate_hz,
        )
    return _MinerUtteranceSession(
        miner=miner,
        ue_utterance=ue_utterance,
        decoded_audio=decoded_audio,
        miner_audio=miner_audio,
        source_audio_bytes=source_audio_bytes,
        started_at=started_at,
        session_id=init_response.session_id,
        frame_rate_hz=init_payload.frame_rate_hz,
        frame_samples=init_payload.frame_samples,
        input_frames=_split_pcm_into_frames(
            miner_audio.pcm_bytes,
            frame_samples=init_payload.frame_samples,
            channels=miner_audio.channels,
            sample_width_bytes=miner_audio.sample_width_bytes,
        ),
        output_chunks=[],
    )


async def _start_utterances_with_keepalive(
    *,
    miners: List[Any],
    ue_utterance: BBAudioUEUtterance,
    decoded_audio: _DecodedWav,
    source_audio_bytes: bytes,
    init_callback: Callable[[Any, BBAudioMinerInitPayload], Awaitable[Dict[str, Any]]],
    keepalive_enabled: bool,
    keepalive_interval_seconds: float,
    init_barrier_timeout_seconds: float | None = None,
) -> List[Any]:
    tasks: Dict[asyncio.Task, Any] = {
        asyncio.create_task(
            _start_utterance_for_miner(
                miner=miner,
                ue_utterance=ue_utterance,
                decoded_audio=decoded_audio,
                source_audio_bytes=source_audio_bytes,
                init_callback=init_callback,
            )
        ): miner
        for miner in miners
    }
    results_by_hotkey: Dict[str, Any] = {}
    ready_sessions: Dict[str, _MinerUtteranceSession] = {}
    started_at = perf_counter()
    last_keepalive_at = perf_counter()
    interval = max(1.0, keepalive_interval_seconds)
    barrier_timeout = (
        max(0.0, float(init_barrier_timeout_seconds))
        if init_barrier_timeout_seconds is not None
        else 0.0
    )

    while tasks:
        wait_timeout = interval if keepalive_enabled and ready_sessions else None
        if ready_sessions:
            ready_grace_remaining = interval - (perf_counter() - started_at)
            if ready_grace_remaining <= 0.0:
                wait_timeout = 0.0
            elif wait_timeout is None:
                wait_timeout = ready_grace_remaining
            else:
                wait_timeout = min(wait_timeout, ready_grace_remaining)
        if barrier_timeout > 0.0:
            remaining_barrier = barrier_timeout - (perf_counter() - started_at)
            if remaining_barrier <= 0.0:
                logger.info(
                    "S2S audio init barrier reached: challenge=%s utterance=%s pending_miners=%d timeout_s=%.2f",
                    ue_utterance.challenge_uid,
                    ue_utterance.utterance_id,
                    len(tasks),
                    barrier_timeout,
                )
                barrier_exc = AudioChallengeError(
                    f"Arena init barrier exceeded after {barrier_timeout:.2f}s"
                )
                pending_items = list(tasks.items())
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks.keys(), return_exceptions=True)
                for task, miner in pending_items:
                    tasks.pop(task, None)
                    hotkey = str(getattr(miner, "hotkey", ""))
                    results_by_hotkey[hotkey] = barrier_exc
                break
            wait_timeout = (
                remaining_barrier
                if wait_timeout is None
                else min(wait_timeout, remaining_barrier)
            )
        done, pending = await asyncio.wait(
            tasks.keys(),
            timeout=wait_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            miner = tasks.pop(task)
            hotkey = str(getattr(miner, "hotkey", ""))
            try:
                result = task.result()
            except BaseException as exc:  # noqa: BLE001
                results_by_hotkey[hotkey] = exc
                continue
            results_by_hotkey[hotkey] = result
            if isinstance(result, _MinerUtteranceSession):
                ready_sessions[hotkey] = result

        if pending and ready_sessions:
            ready_grace_remaining = interval - (perf_counter() - started_at)
            if ready_grace_remaining <= 0.0:
                logger.info(
                    "S2S audio init ready-miner grace reached: challenge=%s utterance=%s ready_miners=%d pending_miners=%d grace_s=%.2f",
                    ue_utterance.challenge_uid,
                    ue_utterance.utterance_id,
                    len(ready_sessions),
                    len(pending),
                    interval,
                )
                grace_exc = AudioChallengeError(
                    f"Arena ready-miner grace exceeded after {interval:.2f}s"
                )
                pending_items = list(tasks.items())
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks.keys(), return_exceptions=True)
                for task, miner in pending_items:
                    tasks.pop(task, None)
                    hotkey = str(getattr(miner, "hotkey", ""))
                    if hotkey in ready_sessions:
                        continue
                    results_by_hotkey[hotkey] = grace_exc
                break

        if not pending or not keepalive_enabled or not ready_sessions:
            continue
        now = perf_counter()
        if now - last_keepalive_at < interval:
            continue
        last_keepalive_at = now
        for session in ready_sessions.values():
            logger.info(
                "S2S audio init keepalive skipped for ready session while waiting for cold miners: challenge=%s utterance=%s %s session=%s pending_miners=%d",
                session.ue_utterance.challenge_uid,
                session.ue_utterance.utterance_id,
                _miner_log_label(session.miner),
                session.session_id,
                len(pending),
            )

    return [results_by_hotkey.get(str(getattr(miner, "hotkey", ""))) for miner in miners]


async def _predict_frame_for_miner(
    session: _MinerUtteranceSession,
    *,
    frame_index: int,
    response_timeout_seconds: float,
    startup_response_timeout_seconds: float | None = None,
    startup_frame_count: int = 0,
    final_response_timeout_seconds: float | None = None,
    predict_callback: Callable[
        [Any, BBAudioMinerPredictPayload], Awaitable[Dict[str, Any]]
    ],
) -> None:
    frame_bytes = session.input_frames[frame_index]
    total_frames = len(session.input_frames)
    in_eos = frame_index == total_frames - 1
    logger.info(
        "S2S audio frame send: challenge=%s utterance=%s %s frame=%d/%d bytes_in=%d in_eos=%s",
        session.ue_utterance.challenge_uid,
        session.ue_utterance.utterance_id,
        _miner_log_label(session.miner),
        frame_index + 1,
        total_frames,
        len(frame_bytes),
        in_eos,
    )
    predict_payload = BBAudioMinerPredictPayload(
        session_id=session.session_id,
        audio_b64=base64.b64encode(frame_bytes).decode("ascii"),
        in_eos=in_eos,
    )
    if in_eos:
        session.last_input_chunk_sent_at = perf_counter()
    timeout_seconds = response_timeout_seconds
    if (
        startup_response_timeout_seconds is not None
        and startup_frame_count > 0
        and frame_index < startup_frame_count
    ):
        timeout_seconds = max(timeout_seconds, startup_response_timeout_seconds)
    if in_eos and final_response_timeout_seconds is not None:
        timeout_seconds = final_response_timeout_seconds
    predict_response_data = await _await_predict_response(
        miner=session.miner,
        payload=predict_payload,
        predict_callback=predict_callback,
        timeout_seconds=timeout_seconds,
        timeout_label=(
            f"audio chunk response for frame {frame_index + 1}/{total_frames}"
        ),
    )
    predict_response = BBAudioMinerPredictResponse.model_validate(predict_response_data)
    out_bytes = 0
    if predict_response.audio_b64:
        if session.first_output_frame is None:
            session.first_output_frame = frame_index + 1
        decoded_out = base64.b64decode(predict_response.audio_b64)
        session.output_chunks.append(decoded_out)
        out_bytes = len(decoded_out)
    logger.info(
        "S2S audio frame recv: challenge=%s utterance=%s %s frame=%d/%d bytes_out=%d out_eos=%s",
        session.ue_utterance.challenge_uid,
        session.ue_utterance.utterance_id,
        _miner_log_label(session.miner),
        frame_index + 1,
        total_frames,
        out_bytes,
        predict_response.out_eos,
    )
    if predict_response.out_eos:
        session.saw_out_eos = True
        session.completed_frame = frame_index + 1


async def _drain_miner_until_eos(
    session: _MinerUtteranceSession,
    *,
    max_requests: int,
    global_timeout_seconds: float,
    min_timeout_seconds: float,
    predict_callback: Callable[
        [Any, BBAudioMinerPredictPayload], Awaitable[Dict[str, Any]]
    ],
) -> None:
    if session.saw_out_eos:
        return
    if session.last_input_chunk_sent_at is None:
        raise AudioChallengeError(
            "Drain started before final source chunk timestamp was recorded"
        )
    if max_requests <= 0:
        raise AudioChallengeError(
            "Miner did not signal out_eos before drain budget was exhausted"
        )

    total_frames = len(session.input_frames)
    drain_started_at = perf_counter()
    drain_deadline = max(
        session.last_input_chunk_sent_at + global_timeout_seconds,
        drain_started_at + max(0.0, min_timeout_seconds),
    )
    for drain_index in range(max_requests):
        remaining_timeout = drain_deadline - perf_counter()
        if remaining_timeout <= 0:
            raise AudioChallengeError(
                "Miner did not signal out_eos before drain timeout was exhausted"
            )
        logger.info(
            "S2S audio drain send: challenge=%s utterance=%s %s drain=%d/%d in_eos=true",
            session.ue_utterance.challenge_uid,
            session.ue_utterance.utterance_id,
            _miner_log_label(session.miner),
            drain_index + 1,
            max_requests,
        )
        predict_response_data = await _await_predict_response(
            miner=session.miner,
            payload=BBAudioMinerPredictPayload(
                session_id=session.session_id,
                audio_b64="",
                in_eos=True,
            ),
            predict_callback=predict_callback,
            timeout_seconds=remaining_timeout,
            timeout_label="drain response after final audio chunk",
        )
        predict_response = BBAudioMinerPredictResponse.model_validate(
            predict_response_data
        )
        out_bytes = 0
        if predict_response.audio_b64:
            if session.first_output_frame is None:
                session.first_output_frame = total_frames + drain_index + 1
            decoded_out = base64.b64decode(predict_response.audio_b64)
            session.output_chunks.append(decoded_out)
            out_bytes = len(decoded_out)
        logger.info(
            "S2S audio drain recv: challenge=%s utterance=%s %s drain=%d/%d bytes_out=%d out_eos=%s",
            session.ue_utterance.challenge_uid,
            session.ue_utterance.utterance_id,
            _miner_log_label(session.miner),
            drain_index + 1,
            max_requests,
            out_bytes,
            predict_response.out_eos,
        )
        if predict_response.out_eos:
            session.saw_out_eos = True
            session.completed_frame = total_frames + drain_index + 1
            return

    raise AudioChallengeError(
        "Miner did not signal out_eos before drain budget was exhausted"
    )


def _build_utterance_result(
    pred: _RawPrediction,
    score_result: Optional[Dict[str, Any]] = None,
    reference_metadata: Optional[AudioReferenceMetadata] = None,
) -> BBAudioUtteranceResult:
    dtype = _sample_width_to_dtype(pred.decoded_audio.sample_width_bytes)
    source_duration_sec = (
        float(pred.decoded_audio.frame_count) / float(pred.decoded_audio.sample_rate_hz)
        if pred.decoded_audio.sample_rate_hz > 0
        else 0.0
    )

    if pred.error is not None or score_result is None:
        return BBAudioUtteranceResult(
            challenge_uid=pred.ue_utterance.challenge_uid,
            utterance_index=pred.ue_utterance.utterance_index,
            utterance_id=pred.ue_utterance.utterance_id,
            language=pred.ue_utterance.language,
            miner_uid=int(getattr(pred.miner, "uid", -1)),
            miner_hotkey=str(getattr(pred.miner, "hotkey", "")),
            sample_rate_hz=pred.decoded_audio.sample_rate_hz,
            channels=pred.decoded_audio.channels,
            sample_width_bytes=pred.decoded_audio.sample_width_bytes,
            dtype=dtype,
            frame_rate_hz=0.0,
            frame_samples=0,
            frame_count_in=0,
            frame_count_out=0,
            source_num_bytes=len(pred.source_audio_bytes),
            predicted_num_bytes=0,
            completed=False,
            error=pred.error,
            latency_ms=pred.latency_ms,
            score=0.0,
            score_is_fallback=False,
            score_method="prediction_error" if pred.error else "not_scored",
            source_duration_sec=source_duration_sec,
            reference_text=(
                reference_metadata.reference_text
                if reference_metadata is not None
                else ""
            ),
            scoring_metadata_source=(
                reference_metadata.metadata_source
                if reference_metadata is not None
                else None
            ),
            score_breakdown={
                "prediction_error": pred.error,
            }
            if pred.error is not None
            else {},
            source_audio_bytes=pred.source_audio_bytes,
            predicted_audio_bytes=b"",
        )

    duplication_breakdown = score_result.get("duplicate_penalty", {})
    duplication_score_breakdown = (
        {
            "raw_score": float(
                duplication_breakdown.get(
                    "raw_score", score_result.get("raw_score", score_result.get("score", 0.0))
                )
            ),
            "final_score": float(
                duplication_breakdown.get("final_score", score_result.get("score", 0.0))
            ),
            "penalty_factor": float(
                duplication_breakdown.get(
                    "penalty_factor",
                    duplication_breakdown.get("penalty", 1.0),
                )
            ),
            "duplicate_pressure": float(
                duplication_breakdown.get("duplicate_pressure", 1.0)
            ),
            "max_peer_similarity": float(
                duplication_breakdown.get(
                    "max_peer_similarity",
                    duplication_breakdown.get("max_similarity", 0.0),
                )
            ),
            "similarity_threshold": float(
                duplication_breakdown.get("similarity_threshold", 0.0)
            ),
            "gamma": float(duplication_breakdown.get("gamma", 0.0)),
            "min_score_for_pressure": float(
                duplication_breakdown.get("min_score_for_pressure", 0.0)
            ),
            "score_epsilon": float(duplication_breakdown.get("score_epsilon", 0.0)),
        }
        if duplication_breakdown
        else {}
    )

    return BBAudioUtteranceResult(
        challenge_uid=pred.ue_utterance.challenge_uid,
        utterance_index=pred.ue_utterance.utterance_index,
        utterance_id=pred.ue_utterance.utterance_id,
        language=pred.ue_utterance.language,
        miner_uid=int(getattr(pred.miner, "uid", -1)),
        miner_hotkey=str(getattr(pred.miner, "hotkey", "")),
        miner_session_id=pred.session_id,
        sample_rate_hz=pred.decoded_audio.sample_rate_hz,
        channels=pred.decoded_audio.channels,
        sample_width_bytes=pred.decoded_audio.sample_width_bytes,
        dtype=dtype,
        frame_rate_hz=pred.frame_rate_hz,
        frame_samples=pred.frame_samples,
        frame_count_in=pred.frame_count_in,
        frame_count_out=pred.frame_count_out,
        source_num_bytes=len(pred.source_audio_bytes),
        predicted_num_bytes=len(pred.predicted_wav),
        completed=True,
        error=None,
        latency_ms=pred.latency_ms,
        score=float(score_result.get("score", 0.0)),
        score_is_fallback=bool(score_result.get("score_is_fallback", False)),
        score_method=str(score_result.get("score_method", "not_scored")),
        accuracy=float(score_result.get("accuracy", 0.0)),
        source_duration_sec=float(score_result.get("source_duration_sec", 0.0)),
        predicted_duration_sec=float(score_result.get("predicted_duration_sec", 0.0)),
        effective_completion_sec=float(
            score_result.get("effective_completion_sec", 0.0)
        ),
        reference_text=str(score_result.get("gt_text", "")),
        transcript_text=str(score_result.get("stt_text", "")),
        scoring_metadata_source=score_result.get("scoring_metadata_source"),
        score_breakdown={
            "speech_rate": score_result.get("speech_rate", {}),
            "latency": score_result.get("latency", {}),
            **(
                {"duplication": duplication_score_breakdown}
                if duplication_score_breakdown
                else {}
            ),
            **(
                {"score_error": score_result["score_error"]}
                if "score_error" in score_result
                else {}
            ),
        },
        source_audio_bytes=pred.source_audio_bytes,
        predicted_audio_bytes=pred.predicted_wav,
    )


def _resolve_reference_metadata(
    *,
    challenge_uid: str,
    utterance_id: str,
    challenge_metadata: Optional[Dict[str, Any]],
    metadata_source: Optional[str],
) -> Optional[AudioReferenceMetadata]:
    try:
        return resolve_audio_reference_metadata(
            challenge_uid=challenge_uid,
            utterance_id=utterance_id,
            challenge_doc=challenge_metadata,
            metadata_source=metadata_source,
        )
    except Exception as exc:
        logger.info(
            "Reference metadata unavailable for challenge=%s utterance=%s: %s:%s",
            challenge_uid,
            utterance_id,
            type(exc).__name__,
            exc,
        )
        return None


async def predict_source_audio_multi_miner(
    *,
    utterance_engine_url: str,
    miners: List[Any],
    init_callback: Callable[[Any, BBAudioMinerInitPayload], Awaitable[Dict[str, Any]]],
    predict_callback: Callable[
        [Any, BBAudioMinerPredictPayload], Awaitable[Dict[str, Any]]
    ],
    challenge_type: str = "main",
    profile: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[str], Dict[str, BBAudioChallengeResult]]:
    is_solo_challenge = challenge_type == "solo"
    start_data = (
        await start_solo_audio_session(utterance_engine_url)
        if is_solo_challenge
        else await start_source_audio_session(utterance_engine_url)
    )
    current_utterance = _response_to_ue_utterance(start_data)
    challenge_uid = str(start_data.get("challenge_uid") or "") or None
    ue_utterance_payloads: List[Dict[str, Any]] = [start_data]

    if current_utterance is None or challenge_uid is None:
        logger.info("S2S audio session completed immediately with no utterances")
        return challenge_uid, {}

    session_id = current_utterance.session_id
    prefetched_utterances: List[BBAudioUEUtterance] = [current_utterance]
    challenge_error: str | None = None

    transcription_metadata: Optional[Dict[str, Any]] = None
    transcription_metadata_source = f"{utterance_engine_url.rstrip('/')}/transcription"
    if not is_solo_challenge:
        try:
            transcription_payload = await fetch_transcription_ground_truth(
                utterance_engine_url
            )
            payload_challenge_uid = str(transcription_payload.get("challenge_uid") or "")
            payload_metadata = transcription_payload.get("metadata")
            if payload_challenge_uid == challenge_uid and isinstance(
                payload_metadata, dict
            ):
                transcription_metadata = payload_metadata
            else:
                logger.warning(
                    "Ignoring transcription ground truth for scoring: expected challenge=%s got challenge=%s metadata_type=%s",
                    challenge_uid,
                    payload_challenge_uid,
                    type(payload_metadata).__name__,
                )
        except Exception as exc:
            logger.info(
                "Transcription ground truth unavailable for challenge=%s; falling back to local scoring metadata: %s:%s",
                challenge_uid,
                type(exc).__name__,
                exc,
            )

    if transcription_metadata is None and challenge_uid is not None:
        transcription_metadata = _inline_metadata_from_ue_payloads(
            challenge_uid=challenge_uid,
            payloads=ue_utterance_payloads,
        )
        if transcription_metadata is not None:
            transcription_metadata_source = (
                f"{utterance_engine_url.rstrip('/')}/source-audio"
            )
            logger.info(
                "Using inline source-audio reference metadata for challenge=%s",
                challenge_uid,
            )
        else:
            first_payload_keys = sorted(str(k) for k in start_data.keys())
            logger.warning(
                "No inline source-audio reference metadata found for challenge=%s first_utterance=%s payload_keys=%s",
                challenge_uid,
                str(start_data.get("utterance_id") or start_data.get("utterance_index") or ""),
                first_payload_keys,
            )

    prefetch_cursor = current_utterance
    while prefetch_cursor is not None and not prefetch_cursor.done:
        try:
            next_data = (
                await next_solo_audio_utterance(utterance_engine_url, session_id)
                if is_solo_challenge
                else await next_source_audio_utterance(utterance_engine_url, session_id)
            )
            ue_utterance_payloads.append(next_data)
            if (
                transcription_metadata_source
                == f"{utterance_engine_url.rstrip('/')}/source-audio"
            ):
                updated_inline_metadata = _inline_metadata_from_ue_payloads(
                    challenge_uid=challenge_uid,
                    payloads=ue_utterance_payloads,
                )
                if updated_inline_metadata is not None:
                    transcription_metadata = updated_inline_metadata
            prefetch_cursor = _response_to_ue_utterance(next_data)
            if prefetch_cursor is not None:
                prefetched_utterances.append(prefetch_cursor)
        except Exception as exc:
            challenge_error = f"{type(exc).__name__}:{exc}"
            logger.warning(
                "S2S audio session aborted while prefetching source utterances: challenge=%s session=%s utterances_prefetched=%d error=%s",
                challenge_uid,
                session_id,
                len(prefetched_utterances),
                challenge_error,
            )
            break

    logger.info(
        "S2S audio session started: challenge=%s session=%s miners=%d first_utterance=%s prefetched_utterances=%d",
        challenge_uid,
        session_id,
        len(miners),
        current_utterance.utterance_id,
        len(prefetched_utterances),
    )

    results: Dict[str, BBAudioChallengeResult] = {
        str(getattr(miner, "hotkey", "")): BBAudioChallengeResult(
            challenge_uid=challenge_uid,
            challenge_type=challenge_type,
            miner_uid=int(getattr(miner, "uid", -1)),
            miner_hotkey=str(getattr(miner, "hotkey", "")),
        )
        for miner in miners
    }

    total_miner_serving_seconds = 0.0
    total_scoring_seconds = 0.0
    active_miners = list(miners)
    arena_consecutive_failures = {
        str(getattr(miner, "hotkey", "")): 0 for miner in active_miners
    }
    settings = get_settings()
    max_drain_requests = max(0, int(getattr(settings, "BB_S2S_DRAIN_MAX_REQUESTS", 8)))
    chunk_response_timeout_seconds = max(
        0.001, float(getattr(settings, "BB_S2S_CHUNK_TIMEOUT_SEC", 3.0))
    )
    arena_startup_chunk_timeout_seconds = max(
        chunk_response_timeout_seconds,
        float(
            getattr(
                settings,
                "BB_ARENA_STARTUP_CHUNK_TIMEOUT_SEC",
                60.0,
            )
        ),
    )
    arena_startup_chunk_count = max(
        0,
        int(getattr(settings, "BB_ARENA_STARTUP_CHUNK_COUNT", 4)),
    )
    arena_startup_utterance_count = max(
        1,
        int(getattr(settings, "BB_ARENA_STARTUP_UTTERANCE_COUNT", 3)),
    )
    arena_max_consecutive_failures = max(
        1,
        int(
            getattr(
                settings,
                "BB_ARENA_MAX_CONSECUTIVE_UTTERANCE_FAILURES",
                2,
            )
        ),
    )
    drain_timeout_seconds = max(
        0.001, float(getattr(settings, "BB_S2S_DRAIN_TIMEOUT_SEC", 10.0))
    )
    min_final_drain_timeout_seconds = max(
        0.0,
        float(getattr(settings, "BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC", 5.0)),
    )
    init_keepalive_enabled = str(
        getattr(settings, "BB_ARENA_INIT_KEEPALIVE_ENABLED", "true")
    ).strip().lower() not in {"0", "false", "no", "off"}
    init_keepalive_interval_seconds = max(
        1.0,
        float(getattr(settings, "BB_ARENA_INIT_KEEPALIVE_INTERVAL_SEC", 30.0)),
    )
    arena_init_barrier_timeout_seconds = max(
        0.0,
        float(getattr(settings, "BB_ARENA_INIT_BARRIER_TIMEOUT_SEC", 600.0)),
    )

    for utterance_position, current_utterance in enumerate(prefetched_utterances):
        if not active_miners:
            logger.info(
                "S2S audio session stopping early: challenge=%s session=%s no_active_miners=True",
                challenge_uid,
                session_id,
            )
            break
        raw_audio_bytes = base64.b64decode(current_utterance.audio_b64)
        decoded_audio = _decode_audio_bytes(
            raw_audio_bytes,
            sample_rate_hz=current_utterance.sample_rate_hz,
            channels=current_utterance.channels,
            sample_width_bytes=current_utterance.sample_width_bytes,
            utterance_frames=current_utterance.utterance_frames,
        )
        logger.info(
            "S2S audio utterance received: challenge=%s session=%s utterance_index=%d utterance=%s language=%s frames=%d sample_rate_hz=%d channels=%d sample_width_bytes=%d payload_bytes=%d",
            current_utterance.challenge_uid,
            session_id,
            current_utterance.utterance_index,
            current_utterance.utterance_id,
            current_utterance.language,
            decoded_audio.frame_count,
            decoded_audio.sample_rate_hz,
            decoded_audio.channels,
            decoded_audio.sample_width_bytes,
            len(raw_audio_bytes),
        )
        source_audio_bytes = _pcm_to_wav_bytes(
            decoded_audio.pcm_bytes,
            sample_rate_hz=decoded_audio.sample_rate_hz,
            channels=decoded_audio.channels,
            sample_width_bytes=decoded_audio.sample_width_bytes,
        )
        serving_started_at = perf_counter()
        miner_started_at = {
            str(getattr(miner, "hotkey", "")): perf_counter()
            for miner in active_miners
        }
        start_results = await _start_utterances_with_keepalive(
            miners=active_miners,
            ue_utterance=current_utterance,
            decoded_audio=decoded_audio,
            source_audio_bytes=source_audio_bytes,
            init_callback=init_callback,
            keepalive_enabled=(challenge_type == "arena" and init_keepalive_enabled),
            keepalive_interval_seconds=init_keepalive_interval_seconds,
            init_barrier_timeout_seconds=(
                arena_init_barrier_timeout_seconds
                if challenge_type == "arena"
                else None
            ),
        )

        active_sessions: List[_MinerUtteranceSession] = []
        prediction_by_hotkey: Dict[str, _RawPrediction] = {}
        for miner, start_result in zip(active_miners, start_results):
            if isinstance(start_result, BaseException):
                prediction_by_hotkey[str(getattr(miner, "hotkey", ""))] = (
                    _prediction_failure(
                        miner=miner,
                        ue_utterance=current_utterance,
                        decoded_audio=decoded_audio,
                        source_audio_bytes=source_audio_bytes,
                        started_at=miner_started_at[str(getattr(miner, "hotkey", ""))],
                        exc=start_result,
                    )
                )
                continue
            assert isinstance(start_result, _MinerUtteranceSession)
            active_sessions.append(start_result)

        max_frame_count = max(
            (len(session.input_frames) for session in active_sessions), default=0
        )
        for frame_index in range(max_frame_count):
            frame_sessions = [
                session
                for session in active_sessions
                if frame_index < len(session.input_frames) and not session.saw_out_eos
            ]
            if not frame_sessions:
                continue

            frame_results = await asyncio.gather(
                *(
                    _predict_frame_for_miner(
                        session,
                        frame_index=frame_index,
                        response_timeout_seconds=chunk_response_timeout_seconds,
                        startup_response_timeout_seconds=(
                            arena_startup_chunk_timeout_seconds
                            if (
                                challenge_type == "arena"
                                and utterance_position < arena_startup_utterance_count
                            )
                            else None
                        ),
                        startup_frame_count=(
                            arena_startup_chunk_count
                            if (
                                challenge_type == "arena"
                                and utterance_position < arena_startup_utterance_count
                            )
                            else 0
                        ),
                        final_response_timeout_seconds=drain_timeout_seconds,
                        predict_callback=predict_callback,
                    )
                    for session in frame_sessions
                ),
                return_exceptions=True,
            )

            failed_sessions: set[int] = set()
            for session, frame_result in zip(frame_sessions, frame_results):
                if isinstance(frame_result, Exception):
                    prediction_by_hotkey[str(getattr(session.miner, "hotkey", ""))] = (
                        _prediction_failure(
                            miner=session.miner,
                            ue_utterance=session.ue_utterance,
                            decoded_audio=session.decoded_audio,
                            source_audio_bytes=session.source_audio_bytes,
                            started_at=session.started_at,
                            exc=frame_result,
                        )
                    )
                    failed_sessions.add(id(session))

            if failed_sessions:
                active_sessions = [
                    session
                    for session in active_sessions
                    if id(session) not in failed_sessions
                ]

        drain_sessions = [
            session for session in active_sessions if not session.saw_out_eos
        ]
        if drain_sessions:
            drain_results = await asyncio.gather(
                *(
                    _drain_miner_until_eos(
                        session,
                        max_requests=max_drain_requests,
                        global_timeout_seconds=drain_timeout_seconds,
                        min_timeout_seconds=min_final_drain_timeout_seconds,
                        predict_callback=predict_callback,
                    )
                    for session in drain_sessions
                ),
                return_exceptions=True,
            )

            failed_sessions: set[int] = set()
            for session, drain_result in zip(drain_sessions, drain_results):
                if isinstance(drain_result, Exception):
                    prediction_by_hotkey[str(getattr(session.miner, "hotkey", ""))] = (
                        _prediction_failure(
                            miner=session.miner,
                            ue_utterance=session.ue_utterance,
                            decoded_audio=session.decoded_audio,
                            source_audio_bytes=session.source_audio_bytes,
                            started_at=session.started_at,
                            exc=drain_result,
                        )
                    )
                    failed_sessions.add(id(session))

            if failed_sessions:
                active_sessions = [
                    session
                    for session in active_sessions
                    if id(session) not in failed_sessions
                ]

        for session in active_sessions:
            try:
                prediction_by_hotkey[str(getattr(session.miner, "hotkey", ""))] = (
                    _prediction_success(session)
                )
            except Exception as exc:
                prediction_by_hotkey[str(getattr(session.miner, "hotkey", ""))] = (
                    _prediction_failure(
                        miner=session.miner,
                        ue_utterance=session.ue_utterance,
                        decoded_audio=session.decoded_audio,
                        source_audio_bytes=session.source_audio_bytes,
                        started_at=session.started_at,
                        exc=exc,
                    )
                )

        raw_predictions = [
            prediction_by_hotkey[str(getattr(miner, "hotkey", ""))]
            for miner in active_miners
        ]
        total_miner_serving_seconds += perf_counter() - serving_started_at

        source_duration_sec = (
            float(decoded_audio.frame_count) / float(decoded_audio.sample_rate_hz)
            if decoded_audio.sample_rate_hz > 0
            else 0.0
        )
        reference_metadata = _resolve_reference_metadata(
            challenge_uid=current_utterance.challenge_uid,
            utterance_id=current_utterance.utterance_id,
            challenge_metadata=transcription_metadata,
            metadata_source=(
                transcription_metadata_source
                if transcription_metadata is not None
                else None
            ),
        )

        successful_indices = [
            i
            for i, p in enumerate(raw_predictions)
            if p.error is None and p.predicted_wav
        ]

        batch_scores: Dict[int, Dict[str, Any]] = {}
        if successful_indices:
            scoring_started_at = perf_counter()
            try:
                scored = score_audio_utterance_batch(
                    predictions=[
                        {
                            "predicted_wav_bytes": raw_predictions[i].predicted_wav,
                            "first_output_frame": raw_predictions[i].first_output_frame,
                            "frame_rate_hz": raw_predictions[i].frame_rate_hz,
                            "source_duration_sec": source_duration_sec,
                            "completion_sec": raw_predictions[i].completion_sec,
                        }
                        for i in successful_indices
                    ],
                    challenge_uid=current_utterance.challenge_uid,
                    utterance_id=current_utterance.utterance_id,
                    source_duration_sec=source_duration_sec,
                    challenge_metadata=transcription_metadata,
                    metadata_source=(
                        transcription_metadata_source
                        if transcription_metadata is not None
                        else None
                    ),
                )
                for j, orig_idx in enumerate(successful_indices):
                    batch_scores[orig_idx] = scored[j]
            except Exception as exc:
                logger.info(
                    "Batch scoring failed for challenge=%s utterance=%s: %s:%s",
                    current_utterance.challenge_uid,
                    current_utterance.utterance_id,
                    type(exc).__name__,
                    exc,
                )
                for idx in successful_indices:
                    pred = raw_predictions[idx]
                    predicted_duration_sec = (
                        float(len(pred.predicted_wav))
                        / float(
                            pred.decoded_audio.sample_rate_hz
                            * pred.decoded_audio.channels
                            * pred.decoded_audio.sample_width_bytes
                        )
                        if pred.decoded_audio.sample_rate_hz > 0
                        else 0.0
                    )
                    effective_completion_sec = pred.completion_sec
                    batch_scores[idx] = {
                        "score": 0.0,
                        "accuracy": 0.0,
                        "speech_rate": {"penalty": 0.0, "reason": "scoring_error"},
                        "latency": {
                            "score": 1.0,
                            "completion_sec": round(effective_completion_sec, 4),
                            "source_duration_sec": round(source_duration_sec, 4),
                            "overshoot_sec": round(
                                max(
                                    0.0,
                                    effective_completion_sec - source_duration_sec,
                                ),
                                4,
                            ),
                        },
                        "stt_text": "",
                        "gt_text": (
                            reference_metadata.reference_text
                            if reference_metadata is not None
                            else ""
                        ),
                        "predicted_duration_sec": round(predicted_duration_sec, 4),
                        "effective_completion_sec": round(effective_completion_sec, 4),
                        "source_duration_sec": round(source_duration_sec, 4),
                        "score_is_fallback": False,
                        "score_method": "semantic_audio_v1_error",
                        "scoring_metadata_source": (
                            reference_metadata.metadata_source
                            if reference_metadata is not None
                            else None
                        ),
                        "score_error": f"{type(exc).__name__}:{exc}",
                    }
            finally:
                total_scoring_seconds += perf_counter() - scoring_started_at

        utterance_results: List[BBAudioUtteranceResult] = []
        for i, pred in enumerate(raw_predictions):
            score_result = batch_scores.get(i)
            utterance_result = _build_utterance_result(
                pred,
                score_result,
                reference_metadata=reference_metadata,
            )
            utterance_results.append(utterance_result)
            duplication = utterance_result.score_breakdown.get("duplication", {})
            logger.info(
                "S2S audio score summary: challenge=%s utterance=%s %s score=%.6f raw_score=%.6f accuracy=%.6f latency=%.6f dup_pressure=%.6f dup_penalty=%.6f max_peer_similarity=%.6f score_method=%s fallback=%s",
                utterance_result.challenge_uid,
                utterance_result.utterance_id,
                _miner_log_label(pred.miner),
                utterance_result.score,
                float(duplication.get("raw_score", utterance_result.score)),
                utterance_result.accuracy,
                float(
                    utterance_result.score_breakdown.get("latency", {}).get(
                        "score", 0.0
                    )
                ),
                float(duplication.get("duplicate_pressure", 1.0)),
                float(duplication.get("penalty_factor", 1.0)),
                float(duplication.get("max_peer_similarity", 0.0)),
                utterance_result.score_method,
                utterance_result.score_is_fallback,
            )
            hotkey = utterance_result.miner_hotkey
            if hotkey in results:
                results[hotkey].utterances.append(utterance_result)

        successful_miners = sum(1 for result in utterance_results if result.completed)
        logger.info(
            "S2S audio utterance processed: challenge=%s utterance=%s miners_ok=%d miners_failed=%d",
            current_utterance.challenge_uid,
            current_utterance.utterance_id,
            successful_miners,
            len(utterance_results) - successful_miners,
        )
        if challenge_type == "arena":
            successful_hotkeys = {
                result.miner_hotkey for result in utterance_results if result.completed
            }
            for miner in active_miners:
                hotkey = str(getattr(miner, "hotkey", ""))
                if hotkey in successful_hotkeys:
                    arena_consecutive_failures[hotkey] = 0
                else:
                    arena_consecutive_failures[hotkey] = (
                        arena_consecutive_failures.get(hotkey, 0) + 1
                    )
            dropped_miners = [
                miner
                for miner in active_miners
                if arena_consecutive_failures.get(str(getattr(miner, "hotkey", "")), 0)
                >= arena_max_consecutive_failures
            ]
            if dropped_miners:
                logger.info(
                    (
                        "S2S audio arena dropped miners after repeated first utterance "
                        "failures: challenge=%s utterance=%s dropped=%d kept=%d "
                        "threshold=%d dropped_uids=%s"
                        if utterance_position == 0
                        else "S2S audio arena dropped miners after repeated utterance "
                        "failures: challenge=%s utterance=%s dropped=%d kept=%d "
                        "threshold=%d dropped_uids=%s"
                    ),
                    current_utterance.challenge_uid,
                    current_utterance.utterance_id,
                    len(dropped_miners),
                    len(active_miners) - len(dropped_miners),
                    arena_max_consecutive_failures,
                    ",".join(str(getattr(miner, "uid", -1)) for miner in dropped_miners),
                )
                active_miners = [
                    miner
                    for miner in active_miners
                    if arena_consecutive_failures.get(
                        str(getattr(miner, "hotkey", "")), 0
                    )
                    < arena_max_consecutive_failures
                ]

        if current_utterance.done:
            logger.info(
                "S2S audio utterance marked done by UE: challenge=%s session=%s utterance=%s",
                current_utterance.challenge_uid,
                session_id,
                current_utterance.utterance_id,
            )
            break

    for challenge_result in results.values():
        utterance_count = len(challenge_result.utterances)
        if utterance_count:
            score_sum = sum(
                utterance.score for utterance in challenge_result.utterances
            )
            challenge_score = score_sum / float(utterance_count)
            challenge_result.score = challenge_score
            challenge_result.completed = all(
                utterance.completed for utterance in challenge_result.utterances
            )
            utterance_methods = {
                utterance.score_method for utterance in challenge_result.utterances
            }
            challenge_result.score_is_fallback = False
            if len(utterance_methods) == 1:
                challenge_result.score_method = next(iter(utterance_methods))
            else:
                challenge_result.score_method = "mixed_semantic_audio_scores"
            if challenge_error is not None:
                challenge_result.completed = False
                challenge_result.error = challenge_error
        else:
            challenge_result.score = 0.0
            challenge_result.completed = False
            if challenge_error is not None:
                challenge_result.error = challenge_error

    completed_miners = sum(1 for result in results.values() if result.completed)
    if profile is not None:
        profile["miner_serving_seconds"] = total_miner_serving_seconds
        profile["scoring_seconds"] = total_scoring_seconds
        if challenge_error is not None:
            profile["challenge_error"] = challenge_error
    logger.info(
        "S2S audio session completed: challenge=%s session=%s miners=%d completed_miners=%d miner_serving_sec=%.3f scoring_sec=%.3f",
        challenge_uid,
        session_id,
        len(results),
        completed_miners,
        total_miner_serving_seconds,
        total_scoring_seconds,
    )

    return challenge_uid, results
