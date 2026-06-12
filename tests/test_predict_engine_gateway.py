from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from babelbit.schemas.audio_prediction import BBAudioMinerInitPayload
from babelbit.utils.predict_engine import (
    _GATEWAY_AUTH_TOKEN_CACHE,
    call_gateway_runsync_audio_endpoint,
    call_managed_container_audio_endpoint,
)


class _MockResponse:
    def __init__(self, status: int, text: str):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text


class _MockSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []

    def post(self, *args, **kwargs):
        self.posts.append({"args": args, "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("unexpected extra POST")
        return self._responses.pop(0)


class _MockMixedSession:
    def __init__(self, gets=None, posts=None):
        self._gets = list(gets or [])
        self._posts = list(posts or [])
        self.gets = []
        self.posts = []

    def get(self, *args, **kwargs):
        self.gets.append({"args": args, "kwargs": kwargs})
        if not self._gets:
            raise AssertionError("unexpected extra GET")
        return self._gets.pop(0)

    def post(self, *args, **kwargs):
        self.posts.append({"args": args, "kwargs": kwargs})
        if not self._posts:
            raise AssertionError("unexpected extra POST")
        return self._posts.pop(0)


@pytest.mark.asyncio
async def test_gateway_runsync_unwraps_vsubmit_output_payload():
    session = _MockSession(
        [
            _MockResponse(200, '{"auth_token":"tok","expires_in":300}'),
            _MockResponse(
                200,
                '{"delay_ms":null,"exec_ms":null,"output":{"ready":true,"session_id":"s1","challenge_uid":"c1","utterance_id":"u1","sample_rate_hz":24000,"frame_rate_hz":12.5,"frame_samples":1920,"dtype":"float32le","channels":1}}',
            ),
        ]
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_GATEWAY_AUTH_API_PATH="/auth/token",
        BB_ARENA_RUNSYNC_API_PATH="/runsync",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_gateway_runsync_audio_endpoint(
            "http://gateway.test/runsync",
            payload,
            miner_hotkey="miner-hk",
            miner_uid=8,
            timeout=5,
        )

    assert response["ready"] is True
    assert response["session_id"] == "s1"
    assert len(session.posts) == 2
    assert session.posts[0]["args"][0] == "http://gateway.test/auth/token"
    assert session.posts[1]["args"][0] == "http://gateway.test/runsync"
    assert session.posts[1]["kwargs"]["json"]["request_id"].startswith("gw-audio:BBAudioMinerInitPayload:8:1:")
    assert session.posts[1]["kwargs"]["json"]["input"]["predict_payload"]["kind"] == "init"
    assert session.posts[1]["kwargs"]["json"]["input"]["bt_headers"]["bt_header_axon_hotkey"] == "miner-hk"


@pytest.mark.asyncio
async def test_gateway_runsync_retries_warmup_error_before_success():
    _GATEWAY_AUTH_TOKEN_CACHE.clear()
    session = _MockSession(
        [
            _MockResponse(200, '{"auth_token":"tok","expires_in":300}'),
            _MockResponse(
                503,
                '{"error":{"code":"pod_capacity_exhausted","message":"Cold pod could not be started"}}',
            ),
            _MockResponse(
                200,
                '{"output":{"ready":true,"session_id":"s1","challenge_uid":"c1","utterance_id":"u1","sample_rate_hz":24000,"frame_rate_hz":12.5,"frame_samples":1920,"dtype":"float32le","channels":1}}',
            ),
        ]
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_GATEWAY_AUTH_API_PATH="/auth/token",
        BB_ARENA_RUNSYNC_API_PATH="/runsync",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch("babelbit.utils.predict_engine.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_gateway_runsync_audio_endpoint(
            "http://gateway.test/runsync",
            payload,
            miner_hotkey="miner-hk",
            miner_uid=8,
            timeout=20,
        )

    assert response["ready"] is True
    assert len(session.posts) == 3
    first_runsync = session.posts[1]["kwargs"]["json"]["request_id"]
    second_runsync = session.posts[2]["kwargs"]["json"]["request_id"]
    assert first_runsync.startswith("gw-audio:BBAudioMinerInitPayload:8:1:")
    assert second_runsync.startswith("gw-audio:BBAudioMinerInitPayload:8:2:")
    assert first_runsync != second_runsync


@pytest.mark.asyncio
async def test_gateway_runsync_uses_auth_token_and_unwraps_json():
    _GATEWAY_AUTH_TOKEN_CACHE.clear()
    session = _MockSession(
        [
            _MockResponse(200, '{"auth_token":"tok","expires_in":300}'),
            _MockResponse(
                200,
                '{"output":{"ready":true,"session_id":"s1","challenge_uid":"c1","utterance_id":"u1","sample_rate_hz":24000,"frame_rate_hz":12.5,"frame_samples":1920,"dtype":"float32le","channels":1}}',
            ),
        ]
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_GATEWAY_AUTH_API_PATH="/auth/token",
        BB_ARENA_RUNSYNC_API_PATH="/runsync",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch("babelbit.utils.predict_engine.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_gateway_runsync_audio_endpoint(
            "http://gateway.test/runsync",
            payload,
            miner_hotkey="miner-hk",
            miner_uid=8,
            timeout=20,
        )

    assert response["ready"] is True
    assert len(session.posts) == 2


@pytest.mark.asyncio
async def test_gateway_runsync_retries_pod_recreating_error_before_success():
    _GATEWAY_AUTH_TOKEN_CACHE.clear()
    session = _MockSession(
        [
            _MockResponse(200, '{"auth_token":"tok","expires_in":300}'),
            _MockResponse(
                503,
                '{"error":{"code":"pod_recreating","message":"Miner pod is being recreated"}}',
            ),
            _MockResponse(
                200,
                '{"output":{"ready":true,"session_id":"s1","challenge_uid":"c1","utterance_id":"u1","sample_rate_hz":24000,"frame_rate_hz":12.5,"frame_samples":1920,"dtype":"float32le","channels":1}}',
            ),
        ]
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_GATEWAY_AUTH_API_PATH="/auth/token",
        BB_ARENA_RUNSYNC_API_PATH="/runsync",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch("babelbit.utils.predict_engine.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_gateway_runsync_audio_endpoint(
            "http://gateway.test/runsync",
            payload,
            miner_hotkey="miner-hk",
            miner_uid=8,
            timeout=20,
        )

    assert response["ready"] is True
    assert len(session.posts) == 3


@pytest.mark.asyncio
async def test_gateway_runsync_does_not_retry_miner_app_unavailable():
    _GATEWAY_AUTH_TOKEN_CACHE.clear()
    session = _MockSession(
        [
            _MockResponse(200, '{"auth_token":"tok","expires_in":300}'),
            _MockResponse(
                503,
                '{"error":{"code":"miner_app_unavailable","message":"Miner app unavailable","details":{"detail":"Miner not initialized"}}}',
            ),
        ]
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_GATEWAY_AUTH_API_PATH="/auth/token",
        BB_ARENA_RUNSYNC_API_PATH="/runsync",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch("babelbit.utils.predict_engine.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_gateway_runsync_audio_endpoint(
            "http://gateway.test/runsync",
            payload,
            miner_hotkey="miner-hk",
            miner_uid=8,
            timeout=20,
        )

    assert "miner_app_unavailable" in response["error"]
    assert len(session.posts) == 2
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_runsync_retries_miner_unavailable_before_success():
    _GATEWAY_AUTH_TOKEN_CACHE.clear()
    session = _MockSession(
        [
            _MockResponse(200, '{"auth_token":"tok","expires_in":300}'),
            _MockResponse(
                410,
                '{"error":{"code":"miner_unavailable","message":"Miner endpoint unavailable"}}',
            ),
            _MockResponse(
                200,
                '{"output":{"ready":true,"session_id":"s1","challenge_uid":"c1","utterance_id":"u1","sample_rate_hz":24000,"frame_rate_hz":12.5,"frame_samples":1920,"dtype":"float32le","channels":1}}',
            ),
        ]
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_GATEWAY_AUTH_API_PATH="/auth/token",
        BB_ARENA_RUNSYNC_API_PATH="/runsync",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch("babelbit.utils.predict_engine.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_gateway_runsync_audio_endpoint(
            "http://gateway.test/runsync",
            payload,
            miner_hotkey="miner-hk",
            miner_uid=8,
            timeout=20,
        )

    assert response["ready"] is True
    assert len(session.posts) == 3


@pytest.mark.asyncio
async def test_gateway_runsync_retries_upstream_error_before_success():
    _GATEWAY_AUTH_TOKEN_CACHE.clear()
    session = _MockSession(
        [
            _MockResponse(200, '{"auth_token":"tok","expires_in":300}'),
            _MockResponse(
                409,
                '{"error":{"code":"upstream_error","message":"Upstream request failed"}}',
            ),
            _MockResponse(
                200,
                '{"output":{"ready":true,"session_id":"s1","challenge_uid":"c1","utterance_id":"u1","sample_rate_hz":24000,"frame_rate_hz":12.5,"frame_samples":1920,"dtype":"float32le","channels":1}}',
            ),
        ]
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_ARENA_GATEWAY_AUTH_API_PATH="/auth/token",
        BB_ARENA_RUNSYNC_API_PATH="/runsync",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch("babelbit.utils.predict_engine.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_gateway_runsync_audio_endpoint(
            "http://gateway.test/runsync",
            payload,
            miner_hotkey="miner-hk",
            miner_uid=8,
            timeout=20,
        )

    assert response["ready"] is True
    assert len(session.posts) == 3


@pytest.mark.asyncio
async def test_managed_container_pod_route_starts_and_waits_before_predict():
    session = _MockMixedSession(
        gets=[_MockResponse(200, '{"status":"ok"}')],
        posts=[
            _MockResponse(200, '{"started":true}'),
            _MockResponse(
                200,
                '{"ready":true,"session_id":"s1","challenge_uid":"c1","utterance_id":"u1","sample_rate_hz":24000,"frame_rate_hz":12.5,"frame_samples":1920,"dtype":"float32le","channels":1}',
            ),
        ],
    )
    keypair = Mock()
    keypair.sign.return_value = b"sig"
    settings = SimpleNamespace(
        BB_ARENA_MINER_TIMEOUT_SEC=10,
        BB_MINER_TIMEOUT_SEC=10,
        BB_MINER_PREDICT_ENDPOINT="v1/predict",
    )
    payload = BBAudioMinerInitPayload(
        challenge_uid="c1",
        utterance_id="u1",
        sample_rate_hz=24000,
        frame_rate_hz=12.5,
        frame_samples=1920,
        dtype="float32le",
        channels=1,
    )

    with (
        patch("babelbit.utils.predict_engine.get_settings", return_value=settings),
        patch("babelbit.utils.predict_engine.get_async_client", new_callable=AsyncMock, return_value=session),
        patch("babelbit.utils.predict_engine.getenv", side_effect=lambda name, default=None: "rp-key" if name == "RUNPOD_API_KEY" else default),
        patch(
            "babelbit.utils.predict_engine._get_validator_identity",
            return_value={
                "keypair": keypair,
                "hotkey": "validator-hk",
                "external_ip": "127.0.0.1",
                "uuid": "validator-uuid",
            },
        ),
    ):
        response = await call_managed_container_audio_endpoint(
            "http://149.36.1.29:13426",
            payload,
            miner_hotkey="miner-hk",
            endpoint_id="zrhjcj1p2df7fd",
            endpoint_type="POD",
            status="unhealthy",
            timeout=5,
        )

    assert response["ready"] is True
    assert session.posts[0]["args"][0] == "https://rest.runpod.io/v1/pods/zrhjcj1p2df7fd/start"
    assert session.gets[0]["args"][0] == "http://149.36.1.29:13426/healthz"
    assert session.posts[1]["args"][0] == "http://149.36.1.29:13426/v1/predict"
