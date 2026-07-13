import asyncio
import base64
import io
import json
import logging
import numpy as np
import re
import tarfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from babelbit.cli.runner import (
    _build_axon_audio_callbacks,
    _build_round2_audio_callbacks,
    _enforce_runner_logging_level,
    _resolve_round2_routes_when_ready,
    runner,
    runner_round2,
)
from babelbit.scoring.reference_metadata import resolve_audio_reference_metadata
from babelbit.scoring.utterance_scoring import score_audio_utterance_batch
from babelbit.schemas.audio_prediction import (
    BBAudioChallengeResult,
    BBAudioUEUtterance,
    BBAudioUtteranceResult,
)
from babelbit.utils.miner_registry import Miner
from babelbit.utils.managed_container_registry import ManagedRoute
from babelbit.utils.predict_audio import (
    AudioChallengeError,
    _DecodedWav,
    _MinerAudio,
    _MinerUtteranceSession,
    _start_utterances_with_keepalive,
    _drain_miner_until_eos,
    predict_source_audio_multi_miner,
)


def _build_test_wav(
    *,
    sample_rate_hz: int = 8,
    channels: int = 1,
    sample_width_bytes: int = 2,
    frame_count: int = 10,
) -> bytes:
    pcm = b"".join(
        (i % 256).to_bytes(sample_width_bytes, "little", signed=False)
        for i in range(frame_count)
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width_bytes)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm)
    return output.getvalue()


@pytest.mark.asyncio
async def test_audio_callbacks_use_separate_init_timeout():
    calls = []

    async def fake_call_miner_axon_audio_endpoint(**kwargs):
        calls.append(("axon", kwargs["timeout"]))
        return {"ok": True}

    async def fake_call_managed_route_audio_endpoint(**kwargs):
        calls.append(("round2", kwargs["timeout"], kwargs["route"].miner_uid))
        return {"ok": True}

    miner = Miner(uid=8, hotkey="hk", block=1, axon_ip="127.0.0.1", axon_port=8000)
    route = ManagedRoute(miner_hotkey="hk", endpoint_url="http://miner")

    with patch(
        "babelbit.utils.predict_engine.call_miner_axon_audio_endpoint",
        side_effect=fake_call_miner_axon_audio_endpoint,
    ), patch(
        "babelbit.utils.predict_engine.call_managed_route_audio_endpoint",
        side_effect=fake_call_managed_route_audio_endpoint,
    ):
        axon_init, axon_predict = _build_axon_audio_callbacks(
            miner_timeout=10.0,
            init_timeout=60.0,
        )
        round2_init, round2_predict = _build_round2_audio_callbacks(
            routes_by_hotkey={"hk": route},
            miner_timeout=10.0,
            init_timeout=60.0,
        )

        await axon_init(miner, SimpleNamespace(utterance_id="challenge-audio:0"))
        await axon_predict(miner, SimpleNamespace())
        await round2_init(miner, SimpleNamespace(utterance_id="challenge-audio:0"))
        await round2_predict(miner, SimpleNamespace())

    assert calls == [
        ("axon", 60.0),
        ("axon", 10.0),
        ("round2", 60.0, 8),
        ("round2", 10.0, 8),
    ]


@pytest.mark.asyncio
async def test_audio_callbacks_use_default_timeout_after_first_utterance():
    calls = []

    async def fake_call_miner_axon_audio_endpoint(**kwargs):
        calls.append(("axon", kwargs["timeout"], kwargs["payload"].utterance_id))
        return {"ok": True}

    async def fake_call_managed_route_audio_endpoint(**kwargs):
        calls.append(
            (
                "round2",
                kwargs["timeout"],
                kwargs["payload"].utterance_id,
                kwargs["route"].miner_uid,
            )
        )
        return {"ok": True}

    miner = Miner(uid=8, hotkey="hk", block=1, axon_ip="127.0.0.1", axon_port=8000)
    route = ManagedRoute(miner_hotkey="hk", endpoint_url="http://miner")

    with patch(
        "babelbit.utils.predict_engine.call_miner_axon_audio_endpoint",
        side_effect=fake_call_miner_axon_audio_endpoint,
    ), patch(
        "babelbit.utils.predict_engine.call_managed_route_audio_endpoint",
        side_effect=fake_call_managed_route_audio_endpoint,
    ):
        axon_init, _ = _build_axon_audio_callbacks(
            miner_timeout=10.0,
            init_timeout=60.0,
        )
        round2_init, _ = _build_round2_audio_callbacks(
            routes_by_hotkey={"hk": route},
            miner_timeout=10.0,
            init_timeout=60.0,
        )

        await axon_init(miner, SimpleNamespace(utterance_id="challenge-audio:1"))
        await round2_init(miner, SimpleNamespace(utterance_id="challenge-audio:1"))

    assert calls == [
        ("axon", 10.0, "challenge-audio:1"),
        ("round2", 10.0, "challenge-audio:1", 8),
    ]


@pytest.mark.asyncio
async def test_audio_callbacks_use_init_timeout_for_arena_startup_window():
    calls = []

    async def fake_call_miner_axon_audio_endpoint(**kwargs):
        calls.append(("axon", kwargs["timeout"], kwargs["payload"].utterance_id))
        return {"ok": True}

    async def fake_call_managed_route_audio_endpoint(**kwargs):
        calls.append(
            (
                "round2",
                kwargs["timeout"],
                kwargs["payload"].utterance_id,
                kwargs["route"].miner_uid,
            )
        )
        return {"ok": True}

    miner = Miner(uid=8, hotkey="hk", block=1, axon_ip="127.0.0.1", axon_port=8000)
    route = ManagedRoute(miner_hotkey="hk", endpoint_url="http://miner")

    with patch(
        "babelbit.utils.predict_engine.call_miner_axon_audio_endpoint",
        side_effect=fake_call_miner_axon_audio_endpoint,
    ), patch(
        "babelbit.utils.predict_engine.call_managed_route_audio_endpoint",
        side_effect=fake_call_managed_route_audio_endpoint,
    ):
        axon_init, _ = _build_axon_audio_callbacks(
            miner_timeout=10.0,
            init_timeout=60.0,
            startup_utterance_count=3,
        )
        round2_init, _ = _build_round2_audio_callbacks(
            routes_by_hotkey={"hk": route},
            miner_timeout=10.0,
            init_timeout=60.0,
            startup_utterance_count=3,
        )

        await axon_init(miner, SimpleNamespace(utterance_id="challenge-audio:2"))
        await round2_init(miner, SimpleNamespace(utterance_id="challenge-audio:2"))

    assert calls == [
        ("axon", 60.0, "challenge-audio:2"),
        ("round2", 60.0, "challenge-audio:2", 8),
    ]


@pytest.mark.asyncio
async def test_start_utterances_with_keepalive_caps_pending_init_miners():
    miners = [
        Miner(uid=8, hotkey="hk-ready", block=1),
        Miner(uid=9, hotkey="hk-stuck", block=1),
    ]
    ue_utterance = SimpleNamespace(
        challenge_uid="challenge-audio",
        utterance_id="challenge-audio:0",
        language="fr",
    )
    decoded_audio = _DecodedWav(
        sample_rate_hz=24_000,
        channels=1,
        sample_width_bytes=2,
        frame_count=1,
        pcm_bytes=b"\x00\x00",
    )

    async def init_callback(miner, payload):
        if miner.hotkey == "hk-stuck":
            await asyncio.sleep(3600)
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": f"session-{miner.hotkey}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "language": payload.language,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": payload.frame_samples,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    results = await asyncio.wait_for(
        _start_utterances_with_keepalive(
            miners=miners,
            ue_utterance=ue_utterance,
            decoded_audio=decoded_audio,
            source_audio_bytes=b"RIFF",
            init_callback=init_callback,
            keepalive_enabled=False,
            keepalive_interval_seconds=30.0,
            init_barrier_timeout_seconds=0.01,
        ),
        timeout=1.0,
    )

    assert isinstance(results[0], _MinerUtteranceSession)
    assert isinstance(results[1], AudioChallengeError)
    assert "Arena init barrier exceeded" in str(results[1])


@pytest.mark.asyncio
async def test_start_utterances_with_keepalive_releases_ready_miners_after_grace():
    miners = [
        Miner(uid=8, hotkey="hk-ready", block=1),
        Miner(uid=9, hotkey="hk-stuck", block=1),
    ]
    ue_utterance = SimpleNamespace(
        challenge_uid="challenge-audio",
        utterance_id="challenge-audio:0",
        language="fr",
    )
    decoded_audio = _DecodedWav(
        sample_rate_hz=24_000,
        channels=1,
        sample_width_bytes=2,
        frame_count=1,
        pcm_bytes=b"\x00\x00",
    )
    init_calls = {"hk-ready": 0, "hk-stuck": 0}

    async def init_callback(miner, payload):
        init_calls[miner.hotkey] += 1
        if miner.hotkey == "hk-stuck":
            await asyncio.sleep(3600)
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": f"session-{miner.hotkey}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "language": payload.language,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": payload.frame_samples,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    results = await asyncio.wait_for(
        _start_utterances_with_keepalive(
            miners=miners,
            ue_utterance=ue_utterance,
            decoded_audio=decoded_audio,
            source_audio_bytes=b"RIFF",
            init_callback=init_callback,
            keepalive_enabled=True,
            keepalive_interval_seconds=0.01,
            init_barrier_timeout_seconds=3600.0,
        ),
        timeout=3.0,
    )

    assert isinstance(results[0], _MinerUtteranceSession)
    assert isinstance(results[1], AudioChallengeError)
    assert "ready-miner grace exceeded" in str(results[1])
    assert init_calls == {"hk-ready": 1, "hk-stuck": 1}


@pytest.mark.asyncio
async def test_start_utterances_with_keepalive_uses_grace_from_start_time():
    miners = [
        Miner(uid=8, hotkey="hk-ready", block=1),
        Miner(uid=9, hotkey="hk-late", block=1),
        Miner(uid=10, hotkey="hk-stuck", block=1),
    ]
    ue_utterance = SimpleNamespace(
        challenge_uid="challenge-audio",
        utterance_id="challenge-audio:0",
        language="fr",
    )
    decoded_audio = _DecodedWav(
        sample_rate_hz=24_000,
        channels=1,
        sample_width_bytes=2,
        frame_count=1,
        pcm_bytes=b"\x00\x00",
    )

    async def init_callback(miner, payload):
        if miner.hotkey == "hk-late":
            await asyncio.sleep(0.6)
        elif miner.hotkey == "hk-stuck":
            await asyncio.sleep(3600)
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": f"session-{miner.hotkey}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "language": payload.language,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": payload.frame_samples,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    results = await asyncio.wait_for(
        _start_utterances_with_keepalive(
            miners=miners,
            ue_utterance=ue_utterance,
            decoded_audio=decoded_audio,
            source_audio_bytes=b"RIFF",
            init_callback=init_callback,
            keepalive_enabled=True,
            keepalive_interval_seconds=1.0,
            init_barrier_timeout_seconds=3600.0,
        ),
        timeout=3.0,
    )
    elapsed = loop.time() - started_at

    assert elapsed < 1.5
    assert isinstance(results[0], _MinerUtteranceSession)
    assert isinstance(results[1], _MinerUtteranceSession)
    assert isinstance(results[2], AudioChallengeError)
    assert "ready-miner grace exceeded" in str(results[2])


def test_resolve_audio_reference_metadata_supports_translation_schema(tmp_path):
    challenge_uid = "challenge-audio"
    metadata_dir = tmp_path / challenge_uid
    metadata_dir.mkdir(parents=True)
    metadata_path = metadata_dir / "challenge.json"
    metadata_path.write_text(
        json.dumps(
            {
                "challenge_uid": challenge_uid,
                "utterances": [
                    {
                        "utterance_id": 0,
                        "utterance_translations": [
                            {
                                "language": "en",
                                "text": "hello world",
                                "reference_wps": 2.5,
                                "words": [
                                    {"word": "hello", "start": 0.0, "end": 0.4},
                                    {"word": "world", "start": 0.4, "end": 0.8},
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metadata = resolve_audio_reference_metadata(
        challenge_uid=challenge_uid,
        utterance_id=f"{challenge_uid}:0",
        target_lang="en",
        metadata_root=tmp_path,
    )

    assert metadata.reference_text == "hello world"
    assert metadata.reference_wps == 2.5
    assert metadata.reference_words[0]["word"] == "hello"
    assert metadata.metadata_source == str(metadata_path)


def test_resolve_audio_reference_metadata_supports_inline_transcription_payload():
    challenge_uid = "challenge-audio"
    metadata = resolve_audio_reference_metadata(
        challenge_uid=challenge_uid,
        utterance_id=f"{challenge_uid}:0",
        target_lang="en",
        challenge_doc={
            "challenge_uid": challenge_uid,
            "utterances": [
                {
                    "utterance_id": 0,
                    "utterance_translations": [
                        {
                            "language": "en",
                            "text": "hello world",
                            "reference_wps": 2.5,
                            "words": [
                                {"word": "hello", "start": 0.0, "end": 0.4},
                                {"word": "world", "start": 0.4, "end": 0.8},
                            ],
                        }
                    ],
                }
            ],
        },
        metadata_source="http://ue.test/transcription",
    )

    assert metadata.reference_text == "hello world"
    assert metadata.reference_wps == 2.5
    assert metadata.reference_words[0]["word"] == "hello"
    assert metadata.metadata_source == "http://ue.test/transcription"


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_sends_float32_audio_at_12_5hz():
    wav_bytes = _build_test_wav(sample_rate_hz=12_000, frame_count=1_920)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 12_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    init_payloads = []
    predict_chunks = []
    buffered_audio = []

    async def init_callback(_miner, payload):
        init_payloads.append(payload.model_dump())
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "language": payload.language,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": 11,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        predict_chunks.append(decoded)
        buffered_audio.append(decoded)
        if payload.in_eos:
            pcm = b"".join(buffered_audio)
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            return_value=[
                {
                    "score": 0.0,
                    "accuracy": 0.0,
                    "speech_rate": {},
                    "latency": {},
                    "stt_text": "",
                    "gt_text": "",
                    "predicted_duration_sec": 0.0,
                    "effective_completion_sec": 0.0,
                    "source_duration_sec": 0.0,
                    "score_is_fallback": False,
                    "score_method": "semantic_audio_v1",
                }
            ],
        ),
    ):
        _challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert init_payloads == [
        {
            "kind": "init",
            "challenge_uid": "challenge-audio",
            "utterance_id": "challenge-audio:0",
            "language": "fr",
            "sample_rate_hz": 24_000,
            "frame_rate_hz": 12.5,
            "frame_samples": 1_920,
            "dtype": "float32le",
            "channels": 1,
        }
    ]
    assert len(predict_chunks) == 2
    assert all(len(chunk) == 7_680 for chunk in predict_chunks)
    assert np.frombuffer(predict_chunks[0], dtype="<f4").dtype == np.float32
    assert np.frombuffer(predict_chunks[0], dtype="<f4").shape == (1_920,)
    np.testing.assert_allclose(
        np.frombuffer(predict_chunks[0], dtype="<f4")[0],
        0.0,
    )
    utterance = results["hk1"].utterances[0]
    assert utterance.frame_rate_hz == 12.5
    assert utterance.frame_samples == 1_920
    assert utterance.frame_count_in == 2


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_frames_by_miner_contract():
    wav_bytes = _build_test_wav(sample_rate_hz=24_000, frame_count=5_760)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 5_760,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []
    session_buffers = {}

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        if payload.in_eos:
            pcm = b"".join(session_buffers[payload.session_id])
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert "hk1" in results
    result = results["hk1"]
    assert result.completed is True
    assert result.score == 0.0
    assert result.score_is_fallback is False
    assert result.score_method == "semantic_audio_v1_error"
    assert len(result.utterances) == 1
    utterance = result.utterances[0]
    assert utterance.completed is True
    assert utterance.frame_samples == 1_920
    assert utterance.frame_count_in == 3
    assert utterance.predicted_audio_bytes
    assert utterance.score == 0.0
    assert utterance.score_is_fallback is False
    assert utterance.score_method == "semantic_audio_v1_error"
    assert len(predict_calls) == 3
    assert predict_calls[-1]["in_eos"] is True


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_logs_prediction_steps():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    session_buffers = {}

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        if payload.in_eos:
            pcm = b"".join(session_buffers[payload.session_id])
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch("babelbit.utils.predict_audio.logger") as mock_logger,
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
    ):
        await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    info_messages = [call.args[0] for call in mock_logger.info.call_args_list]
    assert any(
        message.startswith("S2S audio session started:") for message in info_messages
    )
    assert any(
        message.startswith("S2S audio utterance received:") for message in info_messages
    )
    assert any(
        message.startswith("S2S audio miner init accepted:")
        for message in info_messages
    )
    assert any(
        message.startswith("S2S audio miner completed:") for message in info_messages
    )
    assert any(
        message.startswith("S2S audio utterance processed:")
        for message in info_messages
    )
    assert any(
        message.startswith("S2S audio session completed:") for message in info_messages
    )


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_sends_each_frame_to_all_miners_before_next():
    wav_bytes = _build_test_wav(sample_rate_hz=24_000, frame_count=5_760)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 5_760,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miners = [
        Miner(uid=8, hotkey="hk1", block=1),
        Miner(uid=9, hotkey="hk2", block=1),
    ]
    predict_call_order = []
    session_buffers = {}

    async def init_callback(miner, payload):
        return {
            "ready": True,
            "miner_id": f"toy-{miner.hotkey}",
            "session_id": f"miner-session-{miner.hotkey}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(miner, payload):
        predict_call_order.append(miner.hotkey)
        await asyncio.sleep(0)
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        if payload.in_eos:
            pcm = b"".join(session_buffers[payload.session_id])
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
    ):
        await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=miners,
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert predict_call_order == ["hk1", "hk2", "hk1", "hk2", "hk1", "hk2"]


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_stops_sending_after_out_eos():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        return {
            "session_id": payload.session_id,
            "audio_b64": payload.audio_b64,
            "out_eos": True,
            "n_bytes": len(base64.b64decode(payload.audio_b64))
            if payload.audio_b64
            else 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
    ):
        await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert len(predict_calls) == 1
    assert predict_calls[0]["in_eos"] is False


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_uses_frame_time_for_completion():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miners = [
        Miner(uid=8, hotkey="fast", block=1),
        Miner(uid=9, hotkey="slow", block=1),
    ]
    captured_predictions = []

    async def init_callback(miner, payload):
        return {
            "ready": True,
            "miner_id": f"toy-{miner.hotkey}",
            "session_id": f"miner-session-{miner.hotkey}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": payload.frame_samples,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(miner, payload):
        if miner.hotkey == "slow":
            await asyncio.sleep(0.03)
        return {
            "session_id": payload.session_id,
            "audio_b64": payload.audio_b64,
            "out_eos": True,
            "n_bytes": len(base64.b64decode(payload.audio_b64))
            if payload.audio_b64
            else 0,
        }

    def fake_score_audio_utterance_batch(*, predictions, **_kwargs):
        captured_predictions.extend(predictions)
        return [
            {
                "score": 0.0,
                "accuracy": 0.0,
                "speech_rate": {},
                "latency": {"completion_sec": pred["completion_sec"]},
                "stt_text": "",
                "gt_text": "",
                "predicted_duration_sec": 0.0,
                "effective_completion_sec": pred["completion_sec"],
                "source_duration_sec": 0.0,
                "score_is_fallback": False,
                "score_method": "semantic_audio_v1",
            }
            for pred in predictions
        ]

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            side_effect=fake_score_audio_utterance_batch,
        ),
    ):
        _challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=miners,
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert results["fast"].utterances[0].completed is True
    assert results["slow"].utterances[0].completed is True
    # Output chunk arrives with frame 1 (0.08 s) and holds 0.08 s of audio,
    # so playback-based completion is 0.16 s.
    assert captured_predictions[0]["completion_sec"] == pytest.approx(0.16)
    assert captured_predictions[1]["completion_sec"] == pytest.approx(0.16)


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_drains_until_out_eos():
    wav_bytes = _build_test_wav(sample_rate_hz=24_000, frame_count=3_840)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 3_840,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []
    buffered_audio = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        if payload.audio_b64:
            buffered_audio.append(base64.b64decode(payload.audio_b64))
            return {
                "session_id": payload.session_id,
                "audio_b64": "",
                "out_eos": False,
                "n_bytes": 0,
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(b"".join(buffered_audio)).decode("ascii"),
            "out_eos": True,
            "n_bytes": sum(len(chunk) for chunk in buffered_audio),
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert len(predict_calls) == 3
    assert predict_calls[0]["audio_b64"]
    assert predict_calls[1]["audio_b64"]
    assert predict_calls[2]["audio_b64"] == ""
    assert predict_calls[-1]["in_eos"] is True
    assert results["hk1"].completed is True
    assert results["hk1"].utterances[0].frame_count_out == 2


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_fails_when_drain_budget_exhausted():
    wav_bytes = _build_test_wav(sample_rate_hz=24_000, frame_count=3_840)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 3_840,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.get_settings",
            return_value=SimpleNamespace(BB_S2S_DRAIN_MAX_REQUESTS=2),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert len(predict_calls) == 4
    assert predict_calls[0]["audio_b64"]
    assert predict_calls[1]["audio_b64"]
    assert predict_calls[2]["audio_b64"] == ""
    assert predict_calls[3]["audio_b64"] == ""
    assert results["hk1"].completed is False
    assert "drain budget was exhausted" in (results["hk1"].utterances[0].error or "")


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_times_out_slow_chunk_response():
    wav_bytes = _build_test_wav(frame_count=8)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        await asyncio.sleep(0.02)
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.get_settings",
            return_value=SimpleNamespace(
                BB_S2S_DRAIN_MAX_REQUESTS=8,
                BB_S2S_CHUNK_TIMEOUT_SEC=0.01,
                BB_S2S_DRAIN_TIMEOUT_SEC=10.0,
            ),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert len(predict_calls) == 1
    assert results["hk1"].completed is False
    assert "audio chunk response" in (results["hk1"].utterances[0].error or "")
    assert "timed out after 0.01s" in (results["hk1"].utterances[0].error or "")


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_allows_slow_final_eos_frame():
    wav_bytes = _build_test_wav(frame_count=8)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        if payload.in_eos:
            await asyncio.sleep(0.02)
            return {
                "session_id": payload.session_id,
                "audio_b64": payload.audio_b64,
                "out_eos": True,
                "n_bytes": len(base64.b64decode(payload.audio_b64)),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.get_settings",
            return_value=SimpleNamespace(
                BB_S2S_DRAIN_MAX_REQUESTS=8,
                BB_S2S_CHUNK_TIMEOUT_SEC=0.01,
                BB_S2S_DRAIN_TIMEOUT_SEC=0.05,
                BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC=0.05,
            ),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert len(predict_calls) > 1
    assert sum(1 for call in predict_calls if call["in_eos"]) == 1
    assert predict_calls[-1]["in_eos"] is True
    assert results["hk1"].completed is True
    assert results["hk1"].utterances[0].completed is True
    assert results["hk1"].utterances[0].error is None


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_allows_slow_arena_startup_chunk():
    wav_bytes = _build_test_wav(frame_count=8)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        if len(predict_calls) == 1:
            await asyncio.sleep(0.02)
        return {
            "session_id": payload.session_id,
            "audio_b64": payload.audio_b64 if payload.in_eos else "",
            "out_eos": payload.in_eos,
            "n_bytes": len(base64.b64decode(payload.audio_b64)) if payload.audio_b64 else 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.get_settings",
            return_value=SimpleNamespace(
                BB_S2S_DRAIN_MAX_REQUESTS=8,
                BB_S2S_CHUNK_TIMEOUT_SEC=0.01,
                BB_ARENA_STARTUP_CHUNK_TIMEOUT_SEC=0.05,
                BB_ARENA_STARTUP_CHUNK_COUNT=1,
                BB_S2S_DRAIN_TIMEOUT_SEC=0.05,
                BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC=0.05,
            ),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="arena",
        )

    assert challenge_uid == "challenge-audio"
    assert len(predict_calls) > 1
    assert results["hk1"].completed is True
    assert results["hk1"].utterances[0].completed is True
    assert results["hk1"].utterances[0].error is None


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_allows_slow_arena_startup_chunk_on_later_utterance():
    wav_first = _build_test_wav(frame_count=8)
    wav_second = _build_test_wav(frame_count=8)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_first).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 1,
        "utterance_id": "challenge-audio:1",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": True,
        "audio_b64": base64.b64encode(wav_second).decode("ascii"),
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": f"miner-session:{payload.utterance_id}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        if (
            payload.session_id.endswith("challenge-audio:1")
            and sum(1 for call in predict_calls if call["session_id"] == payload.session_id)
            == 1
        ):
            await asyncio.sleep(0.02)
        return {
            "session_id": payload.session_id,
            "audio_b64": payload.audio_b64 if payload.in_eos else "",
            "out_eos": payload.in_eos,
            "n_bytes": len(base64.b64decode(payload.audio_b64)) if payload.audio_b64 else 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.get_settings",
            return_value=SimpleNamespace(
                BB_S2S_DRAIN_MAX_REQUESTS=8,
                BB_S2S_CHUNK_TIMEOUT_SEC=0.01,
                BB_ARENA_STARTUP_CHUNK_TIMEOUT_SEC=0.05,
                BB_ARENA_STARTUP_CHUNK_COUNT=1,
                BB_ARENA_STARTUP_UTTERANCE_COUNT=2,
                BB_S2S_DRAIN_TIMEOUT_SEC=0.05,
                BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC=0.05,
            ),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="arena",
        )

    assert challenge_uid == "challenge-audio"
    assert len(predict_calls) >= 3
    assert results["hk1"].completed is True
    assert len(results["hk1"].utterances) == 2
    assert results["hk1"].utterances[1].completed is True
    assert results["hk1"].utterances[1].error is None


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_fails_when_drain_timeout_exhausted():
    wav_bytes = _build_test_wav(sample_rate_hz=24_000, frame_count=3_840)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 3_840,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    predict_calls = []

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        predict_calls.append(payload.model_dump())
        if payload.audio_b64:
            return {
                "session_id": payload.session_id,
                "audio_b64": "",
                "out_eos": False,
                "n_bytes": 0,
            }
        await asyncio.sleep(0.03)
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.get_settings",
            return_value=SimpleNamespace(
                BB_S2S_DRAIN_MAX_REQUESTS=8,
                BB_S2S_CHUNK_TIMEOUT_SEC=0.05,
                BB_S2S_DRAIN_TIMEOUT_SEC=0.02,
                BB_S2S_FINAL_DRAIN_MIN_TIMEOUT_SEC=0.0,
            ),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert len(predict_calls) == 3
    assert predict_calls[0]["audio_b64"]
    assert predict_calls[1]["audio_b64"]
    assert predict_calls[2]["audio_b64"] == ""
    assert results["hk1"].completed is False
    assert "drain response after final audio chunk" in (
        results["hk1"].utterances[0].error or ""
    )


@pytest.mark.asyncio
async def test_drain_miner_uses_minimum_timeout_after_global_budget_expires():
    miner = Miner(uid=8, hotkey="hk1", block=1)
    session = _MinerUtteranceSession(
        miner=miner,
        ue_utterance=BBAudioUEUtterance(
            challenge_uid="challenge-audio",
            session_id="ue-session",
            utterance_index=0,
            utterance_id="challenge-audio:0",
            language="fr",
            sample_rate_hz=8,
            channels=1,
            sample_width_bytes=2,
            utterance_frames=8,
            audio_b64="",
        ),
        decoded_audio=_DecodedWav(
            sample_rate_hz=8,
            channels=1,
            sample_width_bytes=2,
            frame_count=8,
            pcm_bytes=b"",
        ),
        miner_audio=_MinerAudio(
            sample_rate_hz=8,
            channels=1,
            sample_width_bytes=2,
            dtype="pcm_s16le",
            frame_count=8,
            pcm_bytes=b"",
        ),
        source_audio_bytes=b"",
        started_at=0.0,
        session_id="miner-session",
        frame_rate_hz=2.0,
        frame_samples=4,
        input_frames=[b"abcd"],
        output_chunks=[],
        last_input_chunk_sent_at=0.0,
    )

    async def predict_callback(_miner, payload):
        await asyncio.sleep(0.02)
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": True,
            "n_bytes": 0,
        }

    await _drain_miner_until_eos(
        session,
        max_requests=1,
        global_timeout_seconds=0.001,
        min_timeout_seconds=0.05,
        predict_callback=predict_callback,
    )

    assert session.saw_out_eos is True


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_processes_final_done_utterance():
    wav_first = _build_test_wav(frame_count=8)
    wav_second = _build_test_wav(frame_count=6)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_first).decode("ascii"),
    }
    final_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 1,
        "utterance_id": "challenge-audio:1",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 6,
        "end_of_utterance": True,
        "done": True,
        "audio_b64": base64.b64encode(wav_second).decode("ascii"),
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    session_buffers = {}

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        if payload.in_eos:
            pcm = b"".join(session_buffers[payload.session_id])
            session_buffers[payload.session_id] = []
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=final_payload),
        ) as mock_next,
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert mock_next.await_count == 1
    challenge_result = results["hk1"]
    assert challenge_result.completed is True
    assert len(challenge_result.utterances) == 2
    assert challenge_result.utterances[0].utterance_id == "challenge-audio:0"
    assert challenge_result.utterances[1].utterance_id == "challenge-audio:1"


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_arena_keeps_single_first_utterance_failures():
    wav_first = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    wav_second = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_first).decode("ascii"),
    }
    final_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 1,
        "utterance_id": "challenge-audio:1",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": True,
        "audio_b64": base64.b64encode(wav_second).decode("ascii"),
    }
    good_miner = Miner(uid=185, hotkey="hk-good", block=1)
    bad_miner = Miner(uid=113, hotkey="hk-bad", block=1)
    init_calls = []
    session_buffers = {}

    async def init_callback(miner, payload):
        init_calls.append((miner.hotkey, payload.utterance_id))
        if miner.hotkey == "hk-bad" and payload.utterance_id == "challenge-audio:0":
            raise AudioChallengeError("cold pod")
        return {
            "ready": True,
            "miner_id": miner.hotkey,
            "session_id": f"{miner.hotkey}:{payload.utterance_id}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": payload.frame_samples,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        pcm = b"".join(session_buffers[payload.session_id])
        return {
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
            "out_eos": payload.in_eos,
            "n_bytes": len(pcm),
        }

    def fake_score_audio_utterance_batch(**kwargs):
        return [
            {
                "score": 1.0,
                "accuracy": 1.0,
                "speech_rate": {},
                "latency": {"score": 1.0},
                "stt_text": "",
                "gt_text": "",
                "predicted_duration_sec": 0.0,
                "effective_completion_sec": 0.0,
                "source_duration_sec": 0.0,
                "score_is_fallback": False,
                "score_method": "semantic_audio_v1",
            }
            for _ in kwargs["predictions"]
        ]

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=final_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            side_effect=fake_score_audio_utterance_batch,
        ),
    ):
        _challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[good_miner, bad_miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="arena",
        )

    assert init_calls == [
        ("hk-good", "challenge-audio:0"),
        ("hk-bad", "challenge-audio:0"),
        ("hk-good", "challenge-audio:1"),
        ("hk-bad", "challenge-audio:1"),
    ]
    assert len(results["hk-good"].utterances) == 2
    assert results["hk-good"].completed is True
    assert len(results["hk-bad"].utterances) == 2
    assert results["hk-bad"].completed is False
    assert results["hk-bad"].utterances[0].completed is False
    assert results["hk-bad"].utterances[1].completed is True


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_arena_keeps_single_later_utterance_failures():
    wav_first = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    wav_second = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    wav_third = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_first).decode("ascii"),
    }
    second_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 1,
        "utterance_id": "challenge-audio:1",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_second).decode("ascii"),
    }
    final_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 2,
        "utterance_id": "challenge-audio:2",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": True,
        "audio_b64": base64.b64encode(wav_third).decode("ascii"),
    }
    miner = Miner(uid=100, hotkey="hk1", block=1)
    init_calls = []
    session_buffers = {}

    async def init_callback(miner, payload):
        init_calls.append((miner.hotkey, payload.utterance_id))
        if payload.utterance_id == "challenge-audio:1":
            raise AudioChallengeError("frame timeout")
        return {
            "ready": True,
            "miner_id": miner.hotkey,
            "session_id": f"{miner.hotkey}:{payload.utterance_id}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": payload.frame_samples,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        pcm = b"".join(session_buffers[payload.session_id])
        return {
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
            "out_eos": payload.in_eos,
            "n_bytes": len(pcm),
        }

    def fake_score_audio_utterance_batch(**kwargs):
        return [
            {
                "score": 1.0,
                "accuracy": 1.0,
                "speech_rate": {},
                "latency": {"score": 1.0},
                "stt_text": "",
                "gt_text": "",
                "predicted_duration_sec": 0.0,
                "effective_completion_sec": 0.0,
                "source_duration_sec": 0.0,
                "score_is_fallback": False,
                "score_method": "semantic_audio_v1",
            }
            for _ in kwargs["predictions"]
        ]

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(side_effect=[second_payload, final_payload]),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            side_effect=fake_score_audio_utterance_batch,
        ),
    ):
        _challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="arena",
        )

    assert init_calls == [
        ("hk1", "challenge-audio:0"),
        ("hk1", "challenge-audio:1"),
        ("hk1", "challenge-audio:2"),
    ]
    assert len(results["hk1"].utterances) == 3
    assert results["hk1"].utterances[0].completed is True
    assert results["hk1"].utterances[1].completed is False
    assert results["hk1"].utterances[1].score_method == "prediction_error"
    assert results["hk1"].utterances[2].completed is True
    assert results["hk1"].completed is False


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_arena_drops_repeated_later_utterance_failures():
    wav_first = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    wav_second = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    wav_third = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    wav_fourth = _build_test_wav(sample_rate_hz=24_000, frame_count=1_920)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_first).decode("ascii"),
    }
    second_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 1,
        "utterance_id": "challenge-audio:1",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_second).decode("ascii"),
    }
    third_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 2,
        "utterance_id": "challenge-audio:2",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_third).decode("ascii"),
    }
    final_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 3,
        "utterance_id": "challenge-audio:3",
        "language": "fr",
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 1_920,
        "end_of_utterance": True,
        "done": True,
        "audio_b64": base64.b64encode(wav_fourth).decode("ascii"),
    }
    miner = Miner(uid=100, hotkey="hk1", block=1)
    init_calls = []
    session_buffers = {}

    async def init_callback(miner, payload):
        init_calls.append((miner.hotkey, payload.utterance_id))
        if payload.utterance_id in {"challenge-audio:1", "challenge-audio:2"}:
            raise AudioChallengeError("frame timeout")
        return {
            "ready": True,
            "miner_id": miner.hotkey,
            "session_id": f"{miner.hotkey}:{payload.utterance_id}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": payload.frame_rate_hz,
            "frame_samples": payload.frame_samples,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        pcm = b"".join(session_buffers[payload.session_id])
        return {
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(pcm).decode("ascii"),
            "out_eos": payload.in_eos,
            "n_bytes": len(pcm),
        }

    def fake_score_audio_utterance_batch(**kwargs):
        return [
            {
                "score": 1.0,
                "accuracy": 1.0,
                "speech_rate": {},
                "latency": {"score": 1.0},
                "stt_text": "",
                "gt_text": "",
                "predicted_duration_sec": 0.0,
                "effective_completion_sec": 0.0,
                "source_duration_sec": 0.0,
                "score_is_fallback": False,
                "score_method": "semantic_audio_v1",
            }
            for _ in kwargs["predictions"]
        ]

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(side_effect=[second_payload, third_payload, final_payload]),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            side_effect=fake_score_audio_utterance_batch,
        ),
    ):
        _challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="arena",
        )

    assert init_calls == [
        ("hk1", "challenge-audio:0"),
        ("hk1", "challenge-audio:1"),
        ("hk1", "challenge-audio:2"),
    ]
    assert len(results["hk1"].utterances) == 3
    assert results["hk1"].utterances[0].completed is True
    assert results["hk1"].utterances[1].completed is False
    assert results["hk1"].utterances[2].completed is False
    assert results["hk1"].completed is False


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_prefetches_before_miner_init():
    wav_first = _build_test_wav(frame_count=8)
    wav_final = _build_test_wav(frame_count=8)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_first).decode("ascii"),
    }
    final_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 1,
        "utterance_id": "challenge-audio:1",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 8,
        "end_of_utterance": True,
        "done": True,
        "audio_b64": base64.b64encode(wav_final).decode("ascii"),
    }
    events: list[str] = []
    miner = Miner(uid=8, hotkey="hk1", block=1)

    async def init_callback(_miner, payload):
        events.append(f"init:{payload.utterance_id}")
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": f"miner-session-{payload.utterance_id}",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        return {
            "session_id": payload.session_id,
            "audio_b64": payload.audio_b64,
            "out_eos": payload.in_eos,
            "n_bytes": len(base64.b64decode(payload.audio_b64)) if payload.audio_b64 else 0,
        }

    async def next_callback(_url, _session_id):
        events.append("next")
        return final_payload

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(side_effect=next_callback),
        ),
    ):
        _challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert events[:2] == ["next", "init:challenge-audio:0"]
    assert len(results["hk1"].utterances) == 2


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_returns_partial_results_on_next_failure():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    session_buffers = {}
    profile = {}

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        if payload.in_eos:
            pcm = b"".join(session_buffers[payload.session_id])
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(
                side_effect=AudioChallengeError(
                    "Failed to advance source-audio session: HTTP 503"
                )
            ),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            return_value=[
                {
                    "score": 0.42,
                    "raw_score": 0.84,
                    "accuracy": 0.7,
                    "speech_rate": {"penalty": 0.8},
                    "latency": {"score": 0.75},
                    "duplicate_penalty": {
                        "raw_score": 0.84,
                        "final_score": 0.42,
                        "penalty_factor": 0.5,
                        "duplicate_pressure": 4.0,
                        "max_peer_similarity": 0.99,
                        "similarity_threshold": 0.88,
                        "gamma": 0.5,
                        "min_score_for_pressure": 0.2,
                        "score_epsilon": 0.02,
                    },
                    "stt_text": "hello there",
                    "gt_text": "hello world",
                    "predicted_duration_sec": 0.5,
                    "effective_completion_sec": 1.5,
                    "source_duration_sec": 1.25,
                    "score_is_fallback": False,
                    "score_method": "semantic_audio_v1",
                    "scoring_metadata_source": "/tmp/metadata/challenge.json",
                }
            ],
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
            profile=profile,
        )

    assert challenge_uid == "challenge-audio"
    assert profile["challenge_error"] == (
        "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
    )
    challenge_result = results["hk1"]
    assert len(challenge_result.utterances) == 1
    assert challenge_result.score == 0.42
    assert challenge_result.completed is False
    assert challenge_result.error == (
        "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
    )
    assert challenge_result.utterances[0].completed is True
    assert challenge_result.utterances[0].score == 0.42


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_applies_real_scores():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    session_buffers = {}

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        if payload.in_eos:
            pcm = b"".join(session_buffers[payload.session_id])
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            return_value=[
                {
                    "score": 0.42,
                    "raw_score": 0.84,
                    "accuracy": 0.7,
                    "speech_rate": {"penalty": 0.8},
                    "latency": {"score": 0.75},
                    "duplicate_penalty": {
                        "raw_score": 0.84,
                        "final_score": 0.42,
                        "penalty_factor": 0.5,
                        "duplicate_pressure": 4.0,
                        "max_peer_similarity": 0.99,
                        "similarity_threshold": 0.88,
                        "gamma": 0.5,
                        "min_score_for_pressure": 0.2,
                        "score_epsilon": 0.02,
                    },
                    "stt_text": "hello there",
                    "gt_text": "hello world",
                    "predicted_duration_sec": 0.5,
                    "effective_completion_sec": 1.5,
                    "source_duration_sec": 1.25,
                    "score_is_fallback": False,
                    "score_method": "semantic_audio_v1",
                    "scoring_metadata_source": "/tmp/metadata/challenge.json",
                }
            ],
        ),
        patch("babelbit.utils.predict_audio.logger") as mock_logger,
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    result = results["hk1"]
    assert result.completed is True
    assert result.score == 0.42
    assert result.score_is_fallback is False
    assert result.score_method == "semantic_audio_v1"
    utterance = result.utterances[0]
    assert utterance.score == 0.42
    assert utterance.score_is_fallback is False
    assert utterance.score_method == "semantic_audio_v1"
    assert utterance.accuracy == 0.7
    assert utterance.reference_text == "hello world"
    assert utterance.transcript_text == "hello there"
    assert utterance.scoring_metadata_source == "/tmp/metadata/challenge.json"
    assert utterance.score_breakdown["speech_rate"]["penalty"] == 0.8
    assert utterance.score_breakdown["latency"]["score"] == 0.75
    assert utterance.score_breakdown["duplication"] == {
        "raw_score": 0.84,
        "final_score": 0.42,
        "penalty_factor": 0.5,
        "duplicate_pressure": 4.0,
        "max_peer_similarity": 0.99,
        "similarity_threshold": 0.88,
        "gamma": 0.5,
        "min_score_for_pressure": 0.2,
        "score_epsilon": 0.02,
    }
    info_messages = [call.args[0] for call in mock_logger.info.call_args_list if call.args]
    assert (
        "S2S audio score summary: challenge=%s utterance=%s %s score=%.6f raw_score=%.6f "
        "accuracy=%.6f latency=%.6f dup_pressure=%.6f dup_penalty=%.6f "
        "max_peer_similarity=%.6f score_method=%s fallback=%s"
    ) in info_messages


def test_enforce_runner_logging_level_preserves_debug_when_requested():
    root_logger = logging.getLogger()
    babelbit_logger = logging.getLogger("babelbit")
    previous_root = root_logger.level
    previous_babelbit = babelbit_logger.level

    try:
        root_logger.setLevel(logging.DEBUG)
        babelbit_logger.setLevel(logging.INFO)
        _enforce_runner_logging_level()
        assert babelbit_logger.level == logging.DEBUG
    finally:
        root_logger.setLevel(previous_root)
        babelbit_logger.setLevel(previous_babelbit)


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_prefers_ue_transcription_ground_truth():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }
    transcription_payload = {
        "challenge_uid": "challenge-audio",
        "metadata": {
            "challenge_uid": "challenge-audio",
            "target_lang": "en",
            "utterances": [
                {
                    "utterance_id": 0,
                    "utterance_translations": [
                        {
                            "language": "en",
                            "text": "hello world",
                            "reference_wps": 2.5,
                            "words": [
                                {"word": "hello", "start": 0.0, "end": 0.4},
                                {"word": "world", "start": 0.4, "end": 0.8},
                            ],
                        }
                    ],
                }
            ],
        },
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)
    session_buffers = {}

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        decoded = base64.b64decode(payload.audio_b64)
        session_buffers.setdefault(payload.session_id, []).append(decoded)
        if payload.in_eos:
            pcm = b"".join(session_buffers[payload.session_id])
            return {
                "session_id": payload.session_id,
                "audio_b64": base64.b64encode(pcm).decode("ascii"),
                "out_eos": True,
                "n_bytes": len(pcm),
            }
        return {
            "session_id": payload.session_id,
            "audio_b64": "",
            "out_eos": False,
            "n_bytes": 0,
        }

    score_calls = []

    def fake_score_audio_utterance_batch(**kwargs):
        score_calls.append(kwargs)
        return [
            {
                "score": 0.42,
                "accuracy": 0.7,
                "speech_rate": {"penalty": 0.8},
                "latency": {"score": 0.75},
                "stt_text": "hello there",
                "gt_text": "hello world",
                "predicted_duration_sec": 0.5,
                "effective_completion_sec": 1.5,
                "source_duration_sec": 1.25,
                "score_is_fallback": False,
                "score_method": "semantic_audio_v1",
                "scoring_metadata_source": "http://ue.test/transcription",
            }
        ]

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(return_value=transcription_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            side_effect=fake_score_audio_utterance_batch,
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert results["hk1"].score == 0.42
    assert len(score_calls) == 1
    assert score_calls[0]["challenge_metadata"] == transcription_payload["metadata"]
    assert score_calls[0]["metadata_source"] == "http://ue.test/transcription"


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_uses_inline_source_audio_metadata_when_transcription_unavailable():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
        "reference_text": "hello world",
        "reference_words": [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.4, "end": 0.8},
        ],
        "reference_wps": 2.5,
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)

    async def init_callback(_miner, payload):
        return {
            "ready": True,
            "miner_id": "toy",
            "session_id": "miner-session",
            "challenge_uid": payload.challenge_uid,
            "utterance_id": payload.utterance_id,
            "sample_rate_hz": payload.sample_rate_hz,
            "frame_rate_hz": 2.0,
            "frame_samples": 4,
            "dtype": payload.dtype,
            "channels": payload.channels,
        }

    async def predict_callback(_miner, payload):
        return {
            "session_id": payload.session_id,
            "audio_b64": payload.audio_b64,
            "out_eos": payload.in_eos,
            "n_bytes": len(base64.b64decode(payload.audio_b64)),
        }

    score_calls = []

    def fake_score_audio_utterance_batch(**kwargs):
        score_calls.append(kwargs)
        return [
            {
                "score": 0.42,
                "accuracy": 0.7,
                "speech_rate": {"penalty": 1.0},
                "latency": {"score": 0.9},
                "stt_text": "hello world",
                "gt_text": "hello world",
                "predicted_duration_sec": 0.5,
                "effective_completion_sec": 1.5,
                "source_duration_sec": 1.25,
                "score_is_fallback": False,
                "score_method": "semantic_audio_v1",
                "scoring_metadata_source": "http://ue.test/source-audio",
            }
        ]

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(side_effect=AudioChallengeError("no transcription")),
        ),
        patch(
            "babelbit.utils.predict_audio.score_audio_utterance_batch",
            side_effect=fake_score_audio_utterance_batch,
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=predict_callback,
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    assert results["hk1"].score == 0.42
    assert len(score_calls) == 1
    assert score_calls[0]["metadata_source"] == "http://ue.test/source-audio"
    assert score_calls[0]["challenge_metadata"] == {
        "challenge_uid": "challenge-audio",
        "utterances": [
            {
                "utterance_id": "challenge-audio:0",
                "utterance_index": 0,
                "reference_text": "hello world",
                "reference_words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.4, "end": 0.8},
                ],
                "reference_wps": 2.5,
            }
        ],
    }


@pytest.mark.asyncio
async def test_runner_writes_s2s_log_tar_and_fallback_scores(tmp_path):
    logs_dir = tmp_path / "logs"
    scores_dir = tmp_path / "scores"
    miner = Miner(uid=8, hotkey="hk1", block=1, axon_ip="127.0.0.1", axon_port=8091)
    source_wav = _build_test_wav(frame_count=6)
    predicted_wav = _build_test_wav(frame_count=6)
    challenge_result = BBAudioChallengeResult(
        challenge_uid="challenge-s2s",
        challenge_type="main",
        miner_uid=miner.uid,
        miner_hotkey=miner.hotkey,
        utterances=[
            BBAudioUtteranceResult(
                challenge_uid="challenge-s2s",
                utterance_index=0,
                utterance_id="challenge-s2s:0",
                language="fr",
                miner_uid=miner.uid,
                miner_hotkey=miner.hotkey,
                miner_session_id="sess-1",
                sample_rate_hz=8,
                channels=1,
                sample_width_bytes=2,
                dtype="int16",
                frame_rate_hz=2.0,
                frame_samples=4,
                frame_count_in=2,
                frame_count_out=1,
                source_num_bytes=len(source_wav),
                predicted_num_bytes=len(predicted_wav),
                completed=True,
                score=1.0,
                source_audio_bytes=source_wav,
                predicted_audio_bytes=predicted_wav,
            )
        ],
        completed=True,
        score=1.0,
    )

    mock_settings = SimpleNamespace(
        BABELBIT_NETUID=42,
        BB_MINER_TIMEOUT_SEC=10,
        S3_BUCKET_NAME="",
        S3_ACCESS_KEY_ID="",
        S3_SECRET_ACCESS_KEY=Mock(get_secret_value=Mock(return_value="")),
        S3_ENDPOINT_URL="",
        S3_REGION="us-east-1",
        S3_ADDRESSING_STYLE="path",
        S3_SIGNATURE_VERSION="s3v4",
        S3_USE_SSL=True,
        S3_LOG_DIR="logs",
    )

    submission_client = SimpleNamespace(
        is_ready=True,
        submit_url="http://submit.test",
        submit_validation_file=AsyncMock(return_value=True),
        submit_validation_artifact=AsyncMock(return_value=True),
    )

    with (
        patch("babelbit.cli.runner.get_settings", return_value=mock_settings),
        patch(
            "babelbit.cli.runner.get_current_challenge_uid",
            AsyncMock(return_value="challenge-s2s"),
        ),
        patch(
            "babelbit.cli.runner.get_miners_from_registry",
            AsyncMock(return_value={miner.uid: miner}),
        ),
        patch(
            "babelbit.cli.runner.predict_source_audio_multi_miner",
            AsyncMock(return_value=("challenge-s2s", {miner.hotkey: challenge_result})),
        ),
        patch(
            "babelbit.cli.runner.ValidationSubmissionClient",
            return_value=submission_client,
        ),
        patch("babelbit.cli.runner.close_http_clients"),
        patch("babelbit.cli.runner.mark_challenge_processed") as mock_mark_processed,
    ):
        with patch.dict(
            "os.environ",
            {
                "BB_OUTPUT_LOGS_DIR": str(logs_dir),
                "BB_OUTPUT_SCORES_DIR": str(scores_dir),
                "BB_ENABLE_SOLO_CHALLENGE": "0",
            },
        ):
            await runner(
                utterance_engine_url="http://ue.test", output_dir=str(scores_dir)
            )

    run_log = logs_dir / "s2s" / "challenge-s2s" / "miner_8__hk_hk1" / "run.json"
    audio_tar = logs_dir / "s2s" / "challenge-s2s" / "miner_8__hk_hk1" / "audio.tar"
    assert run_log.exists()
    assert audio_tar.exists()

    log_doc = json.loads(run_log.read_text(encoding="utf-8"))
    assert log_doc["protocol"] == "s2s_audio_v1"
    assert log_doc["score_is_fallback"] is False
    assert len(log_doc["utterances"]) == 1
    assert log_doc["utterances"][0]["source_audio_path"] == "source/utt_0000.wav"
    assert log_doc["utterances"][0]["predicted_audio_path"] == "predicted/utt_0000.wav"

    with tarfile.open(audio_tar, "r") as tar_file:
        names = sorted(tar_file.getnames())
    assert names == ["predicted/utt_0000.wav", "source/utt_0000.wav"]

    challenge_runs = sorted(
        scores_dir.glob("challenge_run_challenge-s2s_type_main_*.json")
    )
    challenge_scores = sorted(
        scores_dir.glob("challenge_score_challenge-s2s_type_main_*.json")
    )
    assert challenge_runs
    assert challenge_scores

    challenge_run = json.loads(challenge_runs[0].read_text(encoding="utf-8"))
    assert len(challenge_run["utterances"]) == 1
    utterance_entry = challenge_run["utterances"][0]
    assert utterance_entry["utterance_uid"] == "challenge-s2s:0"
    assert utterance_entry["reference_text"] == ""
    assert utterance_entry["transcript"] is None
    assert utterance_entry["score"] == 1.0
    assert utterance_entry["frame_count_in"] == 2
    assert utterance_entry["frame_count_out"] == 1
    assert utterance_entry["predicted_num_bytes"] == len(predicted_wav)
    assert "ground_truth" not in utterance_entry
    assert "transcript_text" not in utterance_entry
    assert "steps" not in utterance_entry
    assert "best_step" not in utterance_entry
    assert "U_best" not in utterance_entry
    assert "total_steps" not in utterance_entry
    assert challenge_run["challenge_summary"]["average_U_best_early"] == 1.0

    challenge_summary = json.loads(challenge_scores[0].read_text(encoding="utf-8"))
    assert challenge_summary["challenge_mean_U"] == 1.0
    assert challenge_summary["protocol"] == "s2s_audio_v1"
    assert challenge_summary["score_is_fallback"] is False
    assert len(challenge_summary["utterances"]) == 1
    assert challenge_summary["utterances"][0]["utterance_score"] == 1.0
    run_timestamp = re.search(r"_run_(\d{8}_\d{6})\.json$", challenge_runs[0].name)
    score_timestamp = re.search(
        r"_score_(\d{8}_\d{6})\.json$", challenge_scores[0].name
    )
    assert run_timestamp is not None
    assert score_timestamp is not None
    assert run_timestamp.group(1) == score_timestamp.group(1)
    submission_client.submit_validation_artifact.assert_awaited_once()
    artifact_call = submission_client.submit_validation_artifact.await_args.kwargs
    assert artifact_call["kind"] == "audio_bundle"
    assert artifact_call["challenge_id"] == "challenge-s2s"
    assert artifact_call["file_path"] == audio_tar
    assert artifact_call["extra_data"]["protocol"] == "s2s_audio_v1"
    assert artifact_call["extra_data"]["score_is_fallback"] is False
    mock_mark_processed.assert_called_once()


@pytest.mark.asyncio
async def test_runner_challenge_score_aggregates_all_utterance_scores(tmp_path):
    logs_dir = tmp_path / "logs"
    scores_dir = tmp_path / "scores"
    miner = Miner(uid=8, hotkey="hk1", block=1, axon_ip="127.0.0.1", axon_port=8091)
    source_wav = _build_test_wav(frame_count=6)
    predicted_wav = _build_test_wav(frame_count=6)
    challenge_result = BBAudioChallengeResult(
        challenge_uid="challenge-s2s",
        challenge_type="main",
        miner_uid=miner.uid,
        miner_hotkey=miner.hotkey,
        utterances=[
            BBAudioUtteranceResult(
                challenge_uid="challenge-s2s",
                utterance_index=0,
                utterance_id="challenge-s2s:0",
                language="fr",
                miner_uid=miner.uid,
                miner_hotkey=miner.hotkey,
                sample_rate_hz=8,
                channels=1,
                sample_width_bytes=2,
                dtype="int16",
                frame_rate_hz=2.0,
                frame_samples=4,
                frame_count_in=2,
                frame_count_out=1,
                source_num_bytes=len(source_wav),
                predicted_num_bytes=len(predicted_wav),
                completed=True,
                score=0.2,
                accuracy=0.3,
                effective_completion_sec=1.0,
                source_audio_bytes=source_wav,
                predicted_audio_bytes=predicted_wav,
            ),
            BBAudioUtteranceResult(
                challenge_uid="challenge-s2s",
                utterance_index=1,
                utterance_id="challenge-s2s:1",
                language="fr",
                miner_uid=miner.uid,
                miner_hotkey=miner.hotkey,
                sample_rate_hz=8,
                channels=1,
                sample_width_bytes=2,
                dtype="int16",
                frame_rate_hz=2.0,
                frame_samples=4,
                frame_count_in=3,
                frame_count_out=1,
                source_num_bytes=len(source_wav),
                predicted_num_bytes=len(predicted_wav),
                completed=True,
                score=0.6,
                accuracy=0.7,
                effective_completion_sec=3.0,
                source_audio_bytes=source_wav,
                predicted_audio_bytes=predicted_wav,
            ),
        ],
        completed=True,
        score=0.99,
        score_is_fallback=False,
        score_method="semantic_audio_v1",
    )

    mock_settings = SimpleNamespace(
        BABELBIT_NETUID=42,
        BB_MINER_TIMEOUT_SEC=10,
        S3_BUCKET_NAME="",
        S3_ACCESS_KEY_ID="",
        S3_SECRET_ACCESS_KEY=Mock(get_secret_value=Mock(return_value="")),
        S3_ENDPOINT_URL="",
        S3_REGION="us-east-1",
        S3_ADDRESSING_STYLE="path",
        S3_SIGNATURE_VERSION="s3v4",
        S3_USE_SSL=True,
        S3_LOG_DIR="logs",
    )

    submission_client = SimpleNamespace(is_ready=False, submit_url="http://submit.test")

    with (
        patch("babelbit.cli.runner.get_settings", return_value=mock_settings),
        patch(
            "babelbit.cli.runner.get_current_challenge_uid",
            AsyncMock(return_value="challenge-s2s"),
        ),
        patch(
            "babelbit.cli.runner.get_miners_from_registry",
            AsyncMock(return_value={miner.uid: miner}),
        ),
        patch(
            "babelbit.cli.runner.predict_source_audio_multi_miner",
            AsyncMock(return_value=("challenge-s2s", {miner.hotkey: challenge_result})),
        ),
        patch(
            "babelbit.cli.runner.ValidationSubmissionClient",
            return_value=submission_client,
        ),
        patch("babelbit.cli.runner.close_http_clients"),
        patch("babelbit.cli.runner.mark_challenge_processed"),
    ):
        with patch.dict(
            "os.environ",
            {
                "BB_OUTPUT_LOGS_DIR": str(logs_dir),
                "BB_OUTPUT_SCORES_DIR": str(scores_dir),
                "BB_ENABLE_SOLO_CHALLENGE": "0",
            },
        ):
            await runner(
                utterance_engine_url="http://ue.test", output_dir=str(scores_dir)
            )

    challenge_runs = sorted(
        scores_dir.glob("challenge_run_challenge-s2s_type_main_*.json")
    )
    challenge_scores = sorted(
        scores_dir.glob("challenge_score_challenge-s2s_type_main_*.json")
    )
    assert challenge_runs
    assert challenge_scores

    challenge_run = json.loads(challenge_runs[0].read_text(encoding="utf-8"))
    assert len(challenge_run["utterances"]) == 2
    assert challenge_run["utterances"][0]["transcript"] is None
    assert challenge_run["challenge_summary"]["average_U_best_early"] == 0.4

    challenge_score = json.loads(challenge_scores[0].read_text(encoding="utf-8"))
    assert len(challenge_score["utterances"]) == 2
    assert challenge_score["challenge_mean_U"] == 0.4
    assert challenge_score["challenge_summary"]["average_U_best_early"] == 0.4
    run_timestamp = re.search(r"_run_(\d{8}_\d{6})\.json$", challenge_runs[0].name)
    score_timestamp = re.search(
        r"_score_(\d{8}_\d{6})\.json$", challenge_scores[0].name
    )
    assert run_timestamp is not None
    assert score_timestamp is not None
    assert run_timestamp.group(1) == score_timestamp.group(1)


@pytest.mark.asyncio
async def test_predict_source_audio_multi_miner_preserves_reference_text_on_failure():
    wav_bytes = _build_test_wav(frame_count=10)
    start_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "utterance_index": 0,
        "utterance_id": "challenge-audio:0",
        "language": "fr",
        "sample_rate_hz": 8,
        "channels": 1,
        "sample_width_bytes": 2,
        "utterance_frames": 10,
        "end_of_utterance": True,
        "done": False,
        "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
    }
    next_payload = {
        "session_id": "ue-session",
        "challenge_uid": "challenge-audio",
        "done": True,
        "end_of_utterance": True,
        "audio_b64": "",
    }
    transcription_payload = {
        "challenge_uid": "challenge-audio",
        "metadata": {
            "challenge_uid": "challenge-audio",
            "utterances": [
                {
                    "utterance_id": 0,
                    "utterance_translations": [
                        {
                            "language": "en",
                            "text": "hello world",
                            "reference_wps": 2.5,
                            "words": [
                                {"word": "hello", "start": 0.0, "end": 0.4},
                                {"word": "world", "start": 0.4, "end": 0.8},
                            ],
                        }
                    ],
                }
            ],
        },
    }

    miner = Miner(uid=8, hotkey="hk1", block=1)

    async def init_callback(_miner, _payload):
        raise RuntimeError("boom")

    with (
        patch(
            "babelbit.utils.predict_audio.start_source_audio_session",
            AsyncMock(return_value=start_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.next_source_audio_utterance",
            AsyncMock(return_value=next_payload),
        ),
        patch(
            "babelbit.utils.predict_audio.fetch_transcription_ground_truth",
            AsyncMock(return_value=transcription_payload),
        ),
    ):
        challenge_uid, results = await predict_source_audio_multi_miner(
            utterance_engine_url="http://ue.test",
            miners=[miner],
            init_callback=init_callback,
            predict_callback=AsyncMock(),
            challenge_type="main",
        )

    assert challenge_uid == "challenge-audio"
    utterance = results["hk1"].utterances[0]
    assert utterance.completed is False
    assert utterance.reference_text == "hello world"
    assert utterance.scoring_metadata_source == "http://ue.test/transcription"
    assert utterance.score_method == "prediction_error"
    assert utterance.score_breakdown["prediction_error"] == "RuntimeError:boom"
    assert utterance.error == "RuntimeError:boom"


@pytest.mark.asyncio
async def test_runner_logs_timing_profile(tmp_path):
    logs_dir = tmp_path / "logs"
    scores_dir = tmp_path / "scores"
    miner = Miner(uid=8, hotkey="hk1", block=1, axon_ip="127.0.0.1", axon_port=8091)
    challenge_result = BBAudioChallengeResult(
        challenge_uid="challenge-s2s",
        challenge_type="main",
        miner_uid=miner.uid,
        miner_hotkey=miner.hotkey,
        utterances=[],
        completed=True,
        score=0.42,
    )

    mock_settings = SimpleNamespace(
        BABELBIT_NETUID=42,
        BB_MINER_TIMEOUT_SEC=10,
        S3_BUCKET_NAME="",
        S3_ACCESS_KEY_ID="",
        S3_SECRET_ACCESS_KEY=Mock(get_secret_value=Mock(return_value="")),
        S3_ENDPOINT_URL="",
        S3_REGION="us-east-1",
        S3_ADDRESSING_STYLE="path",
        S3_SIGNATURE_VERSION="s3v4",
        S3_USE_SSL=True,
        S3_LOG_DIR="logs",
    )

    submission_client = SimpleNamespace(is_ready=False, submit_url="http://submit.test")

    async def fake_predict_source_audio_multi_miner(**kwargs):
        profile = kwargs.get("profile")
        if isinstance(profile, dict):
            profile["miner_serving_seconds"] = 4.25
            profile["scoring_seconds"] = 1.5
        return ("challenge-s2s", {miner.hotkey: challenge_result})

    with (
        patch("babelbit.cli.runner.get_settings", return_value=mock_settings),
        patch(
            "babelbit.cli.runner.get_current_challenge_uid",
            AsyncMock(return_value="challenge-s2s"),
        ),
        patch(
            "babelbit.cli.runner.get_miners_from_registry",
            AsyncMock(return_value={miner.uid: miner}),
        ),
        patch(
            "babelbit.cli.runner.predict_source_audio_multi_miner",
            AsyncMock(side_effect=fake_predict_source_audio_multi_miner),
        ),
        patch(
            "babelbit.cli.runner._score_audio_miners_for_challenge",
            AsyncMock(return_value=(1, 3, [0.42])),
        ),
        patch(
            "babelbit.cli.runner.ValidationSubmissionClient",
            return_value=submission_client,
        ),
        patch("babelbit.cli.runner.close_http_clients"),
        patch("babelbit.cli.runner.mark_challenge_processed"),
        patch("babelbit.cli.runner.logger") as mock_logger,
        patch("babelbit.cli.runner.time.perf_counter", side_effect=[20.0, 21.5]),
    ):
        with patch.dict(
            "os.environ",
            {
                "BB_OUTPUT_LOGS_DIR": str(logs_dir),
                "BB_OUTPUT_SCORES_DIR": str(scores_dir),
                "BB_ENABLE_SOLO_CHALLENGE": "0",
            },
        ):
            await runner(
                utterance_engine_url="http://ue.test", output_dir=str(scores_dir)
            )

    info_messages = [
        call.args[0] for call in mock_logger.info.call_args_list if call.args
    ]
    assert (
        "[RunnerProfile][main] challenge_uid=challenge-s2s miners=1 "
        "miners_with_utterances=0 dialogues_scored=3 "
        "miner_serving_sec=4.250 scoring_sec=1.500 persistence_sec=1.500"
    ) in info_messages


@pytest.mark.asyncio
async def test_runner_does_not_mark_processed_for_partial_challenge(tmp_path):
    logs_dir = tmp_path / "logs"
    scores_dir = tmp_path / "scores"
    miner = Miner(uid=8, hotkey="hk1", block=1, axon_ip="127.0.0.1", axon_port=8091)
    source_wav = _build_test_wav(frame_count=6)
    predicted_wav = _build_test_wav(frame_count=6)
    challenge_result = BBAudioChallengeResult(
        challenge_uid="challenge-s2s",
        challenge_type="main",
        miner_uid=miner.uid,
        miner_hotkey=miner.hotkey,
        utterances=[
            BBAudioUtteranceResult(
                challenge_uid="challenge-s2s",
                utterance_index=0,
                utterance_id="challenge-s2s:0",
                language="fr",
                miner_uid=miner.uid,
                miner_hotkey=miner.hotkey,
                sample_rate_hz=8,
                channels=1,
                sample_width_bytes=2,
                dtype="int16",
                frame_rate_hz=2.0,
                frame_samples=4,
                frame_count_in=2,
                frame_count_out=1,
                source_num_bytes=len(source_wav),
                predicted_num_bytes=len(predicted_wav),
                completed=True,
                score=0.42,
                accuracy=0.7,
                effective_completion_sec=1.5,
                reference_text="hello world",
                transcript_text="hello there",
                source_audio_bytes=source_wav,
                predicted_audio_bytes=predicted_wav,
            )
        ],
        completed=False,
        error="AudioChallengeError:Failed to advance source-audio session: HTTP 503",
        score=0.42,
        score_is_fallback=False,
        score_method="semantic_audio_v1",
    )

    mock_settings = SimpleNamespace(
        BABELBIT_NETUID=42,
        BB_MINER_TIMEOUT_SEC=10,
        S3_BUCKET_NAME="",
        S3_ACCESS_KEY_ID="",
        S3_SECRET_ACCESS_KEY=Mock(get_secret_value=Mock(return_value="")),
        S3_ENDPOINT_URL="",
        S3_REGION="us-east-1",
        S3_ADDRESSING_STYLE="path",
        S3_SIGNATURE_VERSION="s3v4",
        S3_USE_SSL=True,
        S3_LOG_DIR="logs",
    )

    submission_client = SimpleNamespace(is_ready=False, submit_url="http://submit.test")

    async def fake_predict_source_audio_multi_miner(**kwargs):
        profile = kwargs.get("profile")
        if isinstance(profile, dict):
            profile["challenge_error"] = (
                "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
            )
        return ("challenge-s2s", {miner.hotkey: challenge_result})

    with (
        patch("babelbit.cli.runner.get_settings", return_value=mock_settings),
        patch(
            "babelbit.cli.runner.get_current_challenge_uid",
            AsyncMock(return_value="challenge-s2s"),
        ),
        patch(
            "babelbit.cli.runner.get_miners_from_registry",
            AsyncMock(return_value={miner.uid: miner}),
        ),
        patch(
            "babelbit.cli.runner.predict_source_audio_multi_miner",
            AsyncMock(side_effect=fake_predict_source_audio_multi_miner),
        ),
        patch(
            "babelbit.cli.runner.ValidationSubmissionClient",
            return_value=submission_client,
        ),
        patch("babelbit.cli.runner.close_http_clients"),
        patch("babelbit.cli.runner.mark_challenge_processed") as mock_mark_processed,
    ):
        with patch.dict(
            "os.environ",
            {
                "BB_OUTPUT_LOGS_DIR": str(logs_dir),
                "BB_OUTPUT_SCORES_DIR": str(scores_dir),
                "BB_ENABLE_SOLO_CHALLENGE": "0",
            },
        ):
            await runner(
                utterance_engine_url="http://ue.test", output_dir=str(scores_dir)
            )

    challenge_runs = sorted(
        scores_dir.glob("challenge_run_challenge-s2s_type_main_*.json")
    )
    challenge_scores = sorted(
        scores_dir.glob("challenge_score_challenge-s2s_type_main_*.json")
    )
    assert challenge_runs
    assert challenge_scores
    challenge_run = json.loads(challenge_runs[0].read_text(encoding="utf-8"))
    challenge_score = json.loads(challenge_scores[0].read_text(encoding="utf-8"))
    assert challenge_run["error"] == (
        "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
    )
    assert challenge_score["error"] == (
        "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
    )
    mock_mark_processed.assert_not_called()


@pytest.mark.asyncio
async def test_runner_round2_does_not_mark_processed_for_partial_challenge(tmp_path):
    logs_dir = tmp_path / "logs"
    scores_dir = tmp_path / "scores"
    miner = Miner(uid=8, hotkey="hk1", block=1)
    route = ManagedRoute(miner_hotkey="hk1", endpoint_url="http://miner", status="running")
    source_wav = _build_test_wav(frame_count=6)
    predicted_wav = _build_test_wav(frame_count=6)
    challenge_result = BBAudioChallengeResult(
        challenge_uid="challenge-s2s",
        challenge_type="arena",
        miner_uid=miner.uid,
        miner_hotkey=miner.hotkey,
        utterances=[
            BBAudioUtteranceResult(
                challenge_uid="challenge-s2s",
                utterance_index=0,
                utterance_id="challenge-s2s:0",
                language="fr",
                miner_uid=miner.uid,
                miner_hotkey=miner.hotkey,
                sample_rate_hz=8,
                channels=1,
                sample_width_bytes=2,
                dtype="int16",
                frame_rate_hz=2.0,
                frame_samples=4,
                frame_count_in=2,
                frame_count_out=1,
                source_num_bytes=len(source_wav),
                predicted_num_bytes=len(predicted_wav),
                completed=True,
                score=0.42,
                accuracy=0.7,
                effective_completion_sec=1.5,
                reference_text="hello world",
                transcript_text="hello there",
                source_audio_bytes=source_wav,
                predicted_audio_bytes=predicted_wav,
            )
        ],
        completed=False,
        error="AudioChallengeError:Failed to advance source-audio session: HTTP 503",
        score=0.42,
        score_is_fallback=False,
        score_method="semantic_audio_v1",
    )

    mock_settings = SimpleNamespace(
        BABELBIT_NETUID=42,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        S3_BUCKET_NAME="",
        S3_ACCESS_KEY_ID="",
        S3_SECRET_ACCESS_KEY=Mock(get_secret_value=Mock(return_value="")),
        S3_ENDPOINT_URL="",
        S3_REGION="us-east-1",
        S3_ADDRESSING_STYLE="path",
        S3_SIGNATURE_VERSION="s3v4",
        S3_USE_SSL=True,
        S3_LOG_DIR="logs",
    )

    submission_client = SimpleNamespace(is_ready=False, submit_url="http://submit.test")

    async def fake_predict_source_audio_multi_miner(**kwargs):
        profile = kwargs.get("profile")
        if isinstance(profile, dict):
            profile["challenge_error"] = (
                "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
            )
        return ("challenge-s2s", {miner.hotkey: challenge_result})

    with (
        patch("babelbit.cli.runner.get_settings", return_value=mock_settings),
        patch(
            "babelbit.cli.runner.get_current_challenge_uid",
            AsyncMock(return_value="challenge-s2s"),
        ),
        patch(
            "babelbit.cli.runner.resolve_round2_routes",
            AsyncMock(return_value=([miner], {miner.hotkey: route})),
        ),
        patch(
            "babelbit.cli.runner.predict_source_audio_multi_miner",
            AsyncMock(side_effect=fake_predict_source_audio_multi_miner),
        ),
        patch(
            "babelbit.cli.runner.ValidationSubmissionClient",
            return_value=submission_client,
        ),
        patch("babelbit.cli.runner.close_http_clients"),
        patch("babelbit.cli.runner.mark_challenge_processed") as mock_mark_processed,
    ):
        with patch.dict(
            "os.environ",
            {
                "BB_OUTPUT_LOGS_DIR": str(logs_dir),
                "BB_OUTPUT_SCORES_DIR": str(scores_dir),
            },
        ):
            await runner_round2(
                utterance_engine_url="http://ue.test", output_dir=str(scores_dir)
            )

    challenge_runs = sorted(
        scores_dir.glob("challenge_run_challenge-s2s_type_arena_*.json")
    )
    challenge_scores = sorted(
        scores_dir.glob("challenge_score_challenge-s2s_type_arena_*.json")
    )
    assert challenge_runs
    assert challenge_scores
    challenge_run = json.loads(challenge_runs[0].read_text(encoding="utf-8"))
    challenge_score = json.loads(challenge_scores[0].read_text(encoding="utf-8"))
    assert challenge_run["error"] == (
        "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
    )
    assert challenge_score["error"] == (
        "AudioChallengeError:Failed to advance source-audio session: HTTP 503"
    )
    mock_mark_processed.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_round2_routes_when_ready_waits_for_all_discovered_routes():
    miners = [
        Miner(uid=1, hotkey="hk1", block=1),
        Miner(uid=2, hotkey="hk2", block=1),
    ]
    warming_routes = {
        "hk1": ManagedRoute(miner_hotkey="hk1", endpoint_url="http://miner-1", status="running"),
        "hk2": ManagedRoute(miner_hotkey="hk2", endpoint_url="http://miner-2", status="warming"),
    }
    ready_routes = {
        "hk1": ManagedRoute(miner_hotkey="hk1", endpoint_url="http://miner-1", status="running"),
        "hk2": ManagedRoute(miner_hotkey="hk2", endpoint_url="http://miner-2", status="idle"),
    }

    with (
        patch(
            "babelbit.cli.runner.resolve_round2_routes",
            AsyncMock(side_effect=[(miners, warming_routes), (miners, ready_routes)]),
        ) as mock_resolve,
        patch("babelbit.cli.runner.asyncio.sleep", AsyncMock()) as mock_sleep,
    ):
        resolved_miners, routes_by_hotkey = await _resolve_round2_routes_when_ready(
            netuid=42,
            subtensor=None,
            ready_timeout_sec=30,
            poll_sec=5,
        )

    assert resolved_miners == miners
    assert routes_by_hotkey == ready_routes
    assert mock_resolve.await_count == 2
    mock_sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_round2_routes_when_ready_timeout_keeps_live_routes_only():
    miners = [
        Miner(uid=1, hotkey="hk1", block=1),
        Miner(uid=2, hotkey="hk2", block=1),
        Miner(uid=3, hotkey="hk3", block=1),
    ]
    routes = {
        "hk1": ManagedRoute(miner_hotkey="hk1", endpoint_url="http://miner-1", status="running"),
        "hk2": ManagedRoute(miner_hotkey="hk2", endpoint_url="http://miner-2", status="warming"),
        "hk3": ManagedRoute(miner_hotkey="hk3", endpoint_url="http://miner-3", status="unavailable"),
    }

    with patch(
        "babelbit.cli.runner.resolve_round2_routes",
        AsyncMock(return_value=(miners, routes)),
    ) as mock_resolve:
        resolved_miners, routes_by_hotkey = await _resolve_round2_routes_when_ready(
            netuid=42,
            subtensor=None,
            ready_timeout_sec=0,
            poll_sec=5,
        )

    assert [miner.hotkey for miner in resolved_miners] == ["hk1"]
    assert set(routes_by_hotkey) == {"hk1"}
    assert mock_resolve.await_count == 1


@pytest.mark.asyncio
async def test_resolve_round2_routes_when_ready_waits_for_gateway_warming_routes():
    miners = [
        Miner(uid=1, hotkey="hk1", block=1),
        Miner(uid=2, hotkey="hk2", block=1),
    ]
    warming_routes = {
        "hk1": ManagedRoute(
            miner_hotkey="hk1",
            endpoint_url="https://gw.example/runsync",
            provider="gateway",
            status="warming",
        ),
        "hk2": ManagedRoute(
            miner_hotkey="hk2",
            endpoint_url="https://gw.example/runsync",
            provider="gateway",
            status="idle",
        ),
    }
    ready_routes = {
        "hk1": ManagedRoute(
            miner_hotkey="hk1",
            endpoint_url="https://gw.example/runsync",
            provider="gateway",
            status="running",
        ),
        "hk2": warming_routes["hk2"],
    }

    with (
        patch(
            "babelbit.cli.runner.resolve_round2_routes",
            AsyncMock(side_effect=[(miners, warming_routes), (miners, ready_routes)]),
        ) as mock_resolve,
        patch("babelbit.cli.runner.asyncio.sleep", AsyncMock()) as mock_sleep,
    ):
        resolved_miners, routes_by_hotkey = await _resolve_round2_routes_when_ready(
            netuid=42,
            subtensor=None,
            ready_timeout_sec=30,
            poll_sec=5,
        )

    assert resolved_miners == miners
    assert routes_by_hotkey == ready_routes
    assert mock_resolve.await_count == 2
    mock_sleep.assert_awaited_once()


def test_score_audio_utterance_batch_uses_speech_rate_penalty_key(tmp_path):
    stt_cache_path = tmp_path / "stt_cache.jsonl"
    mock_settings = SimpleNamespace(
        BB_AUDIO_SCORING_STT_MODEL="faster-whisper-small",
        BB_AUDIO_SCORING_STT_DEVICE="cpu",
        BB_AUDIO_SCORING_EMBEDDER="all-MiniLM-L6-v2",
        BB_AUDIO_SCORING_STT_CACHE_PATH=stt_cache_path,
        BB_AUDIO_SCORING_ACC_WEIGHT=1.0,
        BB_AUDIO_SCORING_SR_PENALTY_WEIGHT=1.0,
        BB_AUDIO_SCORING_LATENCY_WEIGHT=1.0,
    )
    mock_metadata = SimpleNamespace(
        reference_text="hello world",
        reference_wps=2.0,
        metadata_source="test-metadata",
    )

    with (
        patch(
            "babelbit.scoring.utterance_scoring.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.resolve_audio_reference_metadata",
            return_value=mock_metadata,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.get_reference_embedding",
            return_value=Mock(),
        ),
        patch(
            "babelbit.scoring.utterance_scoring.transcribe_wav_bytes_batch",
            return_value=[
                {
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ],
                    "error": None,
                }
            ],
        ),
        patch(
            "babelbit.scoring.utterance_scoring.compute_accuracy_batch",
            return_value=[0.9],
        ),
        patch(
            "babelbit.scoring.utterance_scoring._wav_duration_sec",
            return_value=1.0,
        ),
    ):
        scores = score_audio_utterance_batch(
            predictions=[
                {
                    "predicted_wav_bytes": _build_test_wav(frame_count=8),
                    "first_output_frame": 0,
                    "frame_rate_hz": 2.0,
                    "source_duration_sec": 1.0,
                }
            ],
            challenge_uid="challenge-s2s",
            utterance_id="challenge-s2s:0",
            source_duration_sec=1.0,
        )

    assert scores[0]["speech_rate"]["penalty"] == 1.0
    assert scores[0]["score"] == 0.9


def test_score_audio_utterance_batch_reproduces_live_clone_input_score(tmp_path):
    stt_cache_path = tmp_path / "stt_cache.jsonl"
    cloned_source_wav = _build_test_wav(sample_rate_hz=24_000, frame_count=342_000)
    mock_settings = SimpleNamespace(
        BB_AUDIO_SCORING_STT_MODEL="faster-whisper-large-v3-turbo",
        BB_AUDIO_SCORING_STT_DEVICE="cuda",
        BB_AUDIO_SCORING_EMBEDDER="all-MiniLM-L6-v2",
        BB_AUDIO_SCORING_STT_CACHE_PATH=stt_cache_path,
        BB_AUDIO_SCORING_ACCURACY_THRESHOLD=0.65,
        BB_AUDIO_SCORING_RATE_LOWER=0.3,
        BB_AUDIO_SCORING_RATE_UPPER=1.3,
        BB_AUDIO_SCORING_LATENCY_OVERSHOOT_FRACTION=0.3,
        BB_AUDIO_SCORING_LATENCY_MIN_OVERSHOOT_SEC=2.0,
        BB_AUDIO_SCORING_LATENCY_MAX_OVERSHOOT_SEC=10.0,
        BB_AUDIO_SCORING_LATENCY_POWER=2.0,
    )
    mock_metadata = SimpleNamespace(
        reference_text=(
            "The delay was not caused directly by the team but by the budget "
            "approval process taking longer than expected."
        ),
        reference_wps=2.969,
        metadata_source="live-validator-2026-06-12",
    )
    live_validator_transcript = (
        "So, on the client file, finally, on the proposal, what I wanted to say "
        "is that the delay does not really come from the team. Not directly. "
        "It's rather the budget validation that took more time than expected."
    )

    with (
        patch(
            "babelbit.scoring.utterance_scoring.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.resolve_audio_reference_metadata",
            return_value=mock_metadata,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.get_reference_embedding",
            return_value=Mock(),
        ),
        patch(
            "babelbit.scoring.utterance_scoring.transcribe_wav_bytes_batch",
            return_value=[
                {
                    "text": live_validator_transcript,
                    "words": [
                        {"word": "So", "start": 0.0, "end": 0.2},
                        {"word": "on", "start": 0.4, "end": 0.6},
                        {"word": "the", "start": 0.8, "end": 1.0},
                        {"word": "client", "start": 1.2, "end": 1.4},
                        {"word": "file", "start": 1.6, "end": 1.845},
                    ],
                    "error": None,
                }
            ],
        ),
        patch(
            "babelbit.scoring.utterance_scoring.compute_accuracy_batch",
            return_value=[0.769289],
        ),
        patch(
            "babelbit.scoring.utterance_scoring._wav_duration_sec",
            return_value=14.32,
        ),
    ):
        scores = score_audio_utterance_batch(
            predictions=[
                {
                    "predicted_wav_bytes": cloned_source_wav,
                    "first_output_frame": 1,
                    "frame_rate_hz": 12.5,
                }
            ],
            challenge_uid="challenge-1781174402-d4bb9c06",
            utterance_id="0",
            source_duration_sec=14.25,
            target_lang="en",
        )

    assert scores[0]["score"] == 0.998769
    assert scores[0]["accuracy"] == 0.769289
    assert scores[0]["accuracy_pass"] is True
    assert scores[0]["stt_text"] == live_validator_transcript
    assert scores[0]["latency"] == {
        "score": 0.998769,
        "completion_sec": 14.4,
        "source_duration_sec": 14.25,
        "overshoot_sec": 0.15,
        "allowed_overshoot_sec": 4.275,
    }


def test_score_audio_utterance_batch_bounds_measured_completion_by_audio_timeline(tmp_path):
    stt_cache_path = tmp_path / "stt_cache.jsonl"
    mock_settings = SimpleNamespace(
        BB_AUDIO_SCORING_STT_MODEL="faster-whisper-small",
        BB_AUDIO_SCORING_STT_DEVICE="cpu",
        BB_AUDIO_SCORING_EMBEDDER="all-MiniLM-L6-v2",
        BB_AUDIO_SCORING_STT_CACHE_PATH=stt_cache_path,
        BB_AUDIO_SCORING_ACC_WEIGHT=1.0,
        BB_AUDIO_SCORING_SR_PENALTY_WEIGHT=1.0,
        BB_AUDIO_SCORING_LATENCY_WEIGHT=1.0,
    )
    mock_metadata = SimpleNamespace(
        reference_text="hello world",
        reference_wps=2.0,
        metadata_source="test-metadata",
    )

    with (
        patch(
            "babelbit.scoring.utterance_scoring.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.resolve_audio_reference_metadata",
            return_value=mock_metadata,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.get_reference_embedding",
            return_value=Mock(),
        ),
        patch(
            "babelbit.scoring.utterance_scoring.transcribe_wav_bytes_batch",
            return_value=[
                {
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ],
                    "error": None,
                }
            ],
        ),
        patch(
            "babelbit.scoring.utterance_scoring.compute_accuracy_batch",
            return_value=[0.9],
        ),
        patch(
            "babelbit.scoring.utterance_scoring._wav_duration_sec",
            return_value=3.0,
        ),
    ):
        scores = score_audio_utterance_batch(
            predictions=[
                {
                    "predicted_wav_bytes": _build_test_wav(frame_count=8),
                    "first_output_frame": 99,
                    "frame_rate_hz": 1.0,
                    "source_duration_sec": 1.0,
                    "completion_sec": 1.25,
                }
            ],
            challenge_uid="challenge-s2s",
            utterance_id="challenge-s2s:0",
            source_duration_sec=1.0,
        )

    assert scores[0]["predicted_duration_sec"] == 3.0
    assert scores[0]["effective_completion_sec"] == 102.0
    assert scores[0]["latency"]["completion_sec"] == 102.0


def test_score_audio_utterance_batch_marks_stt_errors_as_fallback(tmp_path):
    stt_cache_path = tmp_path / "stt_cache.jsonl"
    mock_settings = SimpleNamespace(
        BB_AUDIO_SCORING_STT_MODEL="faster-whisper-small",
        BB_AUDIO_SCORING_STT_DEVICE="cpu",
        BB_AUDIO_SCORING_EMBEDDER="all-MiniLM-L6-v2",
        BB_AUDIO_SCORING_STT_CACHE_PATH=stt_cache_path,
        BB_AUDIO_SCORING_ACC_WEIGHT=1.0,
        BB_AUDIO_SCORING_SR_PENALTY_WEIGHT=1.0,
        BB_AUDIO_SCORING_LATENCY_WEIGHT=1.0,
    )
    mock_metadata = SimpleNamespace(
        reference_text="hello world",
        reference_wps=2.0,
        metadata_source="test-metadata",
    )

    with (
        patch(
            "babelbit.scoring.utterance_scoring.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.resolve_audio_reference_metadata",
            return_value=mock_metadata,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.get_reference_embedding",
            return_value=Mock(),
        ),
        patch(
            "babelbit.scoring.utterance_scoring.transcribe_wav_bytes_batch",
            return_value=[{"text": "", "words": [], "error": "RuntimeError:boom"}],
        ),
        patch(
            "babelbit.scoring.utterance_scoring.compute_accuracy_batch",
            return_value=[0.0],
        ),
        patch(
            "babelbit.scoring.utterance_scoring._wav_duration_sec",
            return_value=1.0,
        ),
    ):
        scores = score_audio_utterance_batch(
            predictions=[
                {
                    "predicted_wav_bytes": _build_test_wav(frame_count=8),
                    "first_output_frame": 0,
                    "frame_rate_hz": 2.0,
                    "source_duration_sec": 1.0,
                }
            ],
            challenge_uid="challenge-s2s",
            utterance_id="challenge-s2s:0",
            source_duration_sec=1.0,
        )

    assert scores[0]["score_is_fallback"] is False
    assert scores[0]["score_method"] == "semantic_audio_v1_error"
    assert scores[0]["score"] == 0.0
    assert scores[0]["score_error"] == "RuntimeError:boom"


def test_score_audio_utterance_batch_marks_empty_short_audio_as_error(tmp_path):
    stt_cache_path = tmp_path / "stt_cache.jsonl"
    mock_settings = SimpleNamespace(
        BB_AUDIO_SCORING_STT_MODEL="faster-whisper-small",
        BB_AUDIO_SCORING_STT_DEVICE="cpu",
        BB_AUDIO_SCORING_EMBEDDER="all-MiniLM-L6-v2",
        BB_AUDIO_SCORING_STT_CACHE_PATH=stt_cache_path,
        BB_AUDIO_SCORING_ACC_WEIGHT=1.0,
        BB_AUDIO_SCORING_SR_PENALTY_WEIGHT=1.0,
        BB_AUDIO_SCORING_LATENCY_WEIGHT=1.0,
    )
    mock_metadata = SimpleNamespace(
        reference_text="hello world",
        reference_wps=2.0,
        metadata_source="test-metadata",
    )

    with (
        patch(
            "babelbit.scoring.utterance_scoring.get_settings",
            return_value=mock_settings,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.resolve_audio_reference_metadata",
            return_value=mock_metadata,
        ),
        patch(
            "babelbit.scoring.utterance_scoring.get_reference_embedding",
            return_value=Mock(),
        ),
        patch(
            "babelbit.scoring.utterance_scoring.transcribe_wav_bytes_batch",
            return_value=[{"text": "", "words": [], "error": None}],
        ),
        patch(
            "babelbit.scoring.utterance_scoring.compute_accuracy_batch",
            return_value=[0.0],
        ),
        patch(
            "babelbit.scoring.utterance_scoring._wav_duration_sec",
            return_value=0.2,
        ),
    ):
        scores = score_audio_utterance_batch(
            predictions=[
                {
                    "predicted_wav_bytes": _build_test_wav(frame_count=8),
                    "first_output_frame": 0,
                    "frame_rate_hz": 2.0,
                    "source_duration_sec": 1.0,
                }
            ],
            challenge_uid="challenge-s2s",
            utterance_id="challenge-s2s:0",
            source_duration_sec=1.0,
        )

    assert scores[0]["score_method"] == "semantic_audio_v1_error"
    assert scores[0]["score"] == 0.0
    assert "short audio output" in scores[0]["score_error"]


def test_transcribe_wav_bytes_resamples_audio_for_whisper(tmp_path):
    from babelbit.scoring.stt import transcribe_wav_bytes

    wav_bytes = _build_test_wav(sample_rate_hz=8000, frame_count=8)
    stt_cache_path = tmp_path / "stt_cache.jsonl"

    with patch(
        "babelbit.scoring.stt._stt_faster_whisper",
        return_value=("hello world", [], "en"),
    ) as mock_stt:
        text, words, detected_language = transcribe_wav_bytes(
            wav_bytes,
            wav_hash="hash-1",
            stt_model="faster-whisper-small",
            language="en",
            device="cpu",
            stt_cache_path=stt_cache_path,
        )

    assert text == "hello world"
    assert words == []
    assert detected_language == "en"
    audio_input = mock_stt.call_args.args[0]
    assert isinstance(audio_input, np.ndarray)
    assert audio_input.dtype == np.float32
    assert len(audio_input) == 16


@pytest.mark.asyncio
async def test_runner_writes_real_audio_score_metadata(tmp_path):
    logs_dir = tmp_path / "logs"
    scores_dir = tmp_path / "scores"
    miner = Miner(uid=8, hotkey="hk1", block=1, axon_ip="127.0.0.1", axon_port=8091)
    source_wav = _build_test_wav(frame_count=6)
    predicted_wav = _build_test_wav(frame_count=6)
    challenge_result = BBAudioChallengeResult(
        challenge_uid="challenge-s2s",
        challenge_type="main",
        miner_uid=miner.uid,
        miner_hotkey=miner.hotkey,
        utterances=[
            BBAudioUtteranceResult(
                challenge_uid="challenge-s2s",
                utterance_index=0,
                utterance_id="challenge-s2s:0",
                language="fr",
                miner_uid=miner.uid,
                miner_hotkey=miner.hotkey,
                miner_session_id="sess-1",
                sample_rate_hz=8,
                channels=1,
                sample_width_bytes=2,
                dtype="int16",
                frame_rate_hz=2.0,
                frame_samples=4,
                frame_count_in=2,
                frame_count_out=1,
                source_num_bytes=len(source_wav),
                predicted_num_bytes=len(predicted_wav),
                completed=True,
                score=0.42,
                score_is_fallback=False,
                score_method="semantic_audio_v1",
                accuracy=0.7,
                reference_text="hello world",
                transcript_text="hello there",
                score_breakdown={
                    "speech_rate": {"penalty": 0.8},
                    "latency": {"score": 0.75},
                },
                source_audio_bytes=source_wav,
                predicted_audio_bytes=predicted_wav,
            )
        ],
        completed=True,
        score=0.42,
        score_is_fallback=False,
        score_method="semantic_audio_v1",
    )

    mock_settings = SimpleNamespace(
        BABELBIT_NETUID=42,
        BB_MINER_TIMEOUT_SEC=10,
        S3_BUCKET_NAME="",
        S3_ACCESS_KEY_ID="",
        S3_SECRET_ACCESS_KEY=Mock(get_secret_value=Mock(return_value="")),
        S3_ENDPOINT_URL="",
        S3_REGION="us-east-1",
        S3_ADDRESSING_STYLE="path",
        S3_SIGNATURE_VERSION="s3v4",
        S3_USE_SSL=True,
        S3_LOG_DIR="logs",
    )

    submission_client = SimpleNamespace(
        is_ready=True,
        submit_url="http://submit.test",
        submit_validation_file=AsyncMock(return_value=True),
        submit_validation_artifact=AsyncMock(return_value=True),
    )

    with (
        patch("babelbit.cli.runner.get_settings", return_value=mock_settings),
        patch(
            "babelbit.cli.runner.get_current_challenge_uid",
            AsyncMock(return_value="challenge-s2s"),
        ),
        patch(
            "babelbit.cli.runner.get_miners_from_registry",
            AsyncMock(return_value={miner.uid: miner}),
        ),
        patch(
            "babelbit.cli.runner.predict_source_audio_multi_miner",
            AsyncMock(return_value=("challenge-s2s", {miner.hotkey: challenge_result})),
        ),
        patch(
            "babelbit.cli.runner.ValidationSubmissionClient",
            return_value=submission_client,
        ),
        patch("babelbit.cli.runner.close_http_clients"),
        patch("babelbit.cli.runner.mark_challenge_processed"),
    ):
        with patch.dict(
            "os.environ",
            {
                "BB_OUTPUT_LOGS_DIR": str(logs_dir),
                "BB_OUTPUT_SCORES_DIR": str(scores_dir),
                "BB_ENABLE_SOLO_CHALLENGE": "0",
            },
        ):
            await runner(
                utterance_engine_url="http://ue.test", output_dir=str(scores_dir)
            )

    challenge_scores = sorted(
        scores_dir.glob("challenge_score_challenge-s2s_type_main_*.json")
    )
    assert challenge_scores
    challenge_summary = json.loads(challenge_scores[0].read_text(encoding="utf-8"))
    assert challenge_summary["challenge_mean_U"] == 0.42
    assert challenge_summary["score_is_fallback"] is False
    assert challenge_summary["score_method"] == "semantic_audio_v1"
    assert challenge_summary["utterances"][0]["utterance_score"] == 0.42

    challenge_runs = sorted(
        scores_dir.glob("challenge_run_challenge-s2s_type_main_*.json")
    )
    challenge_run = json.loads(challenge_runs[0].read_text(encoding="utf-8"))
    utterance_entry = challenge_run["utterances"][0]
    assert utterance_entry["reference_text"] == "hello world"
    assert utterance_entry["transcript"] == "hello there"
    assert utterance_entry["score"] == 0.42
    assert utterance_entry["frame_count_in"] == 2
    assert utterance_entry["frame_count_out"] == 1
    artifact_call = submission_client.submit_validation_artifact.await_args.kwargs
    assert artifact_call["extra_data"]["score_is_fallback"] is False
    assert artifact_call["extra_data"]["score_method"] == "semantic_audio_v1"
