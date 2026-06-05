from json import dumps, loads
import hashlib
from logging import getLogger
from os import getenv
from time import monotonic
import asyncio
import time
from threading import Lock
import uuid
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from asyncio import TimeoutError
from aiohttp import ClientTimeout
from bittensor.utils import networking

from babelbit.schemas.audio_prediction import (
    BBAudioMinerInitPayload,
    BBAudioMinerPredictPayload,
)
from babelbit.utils.async_clients import get_async_client
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.settings import get_settings

logger = getLogger(__name__)

_VALIDATOR_IDENTITY_CACHE = None
_GATEWAY_AUTH_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_GATEWAY_AUTH_TOKEN_LOCK = Lock()
_GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S = 1800


def _env_float(name: str, default: float) -> float:
    try:
        return float(getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: str) -> str:
    value = str(getenv(name, default) or default).strip()
    if not value.startswith("/"):
        value = f"/{value}"
    return value


def _gateway_retry_delay_seconds(attempt: int) -> float:
    base = _env_float("BB_ARENA_GATEWAY_RETRY_BASE_DELAY_SEC", 2.0)
    cap = _env_float("BB_ARENA_GATEWAY_RETRY_MAX_DELAY_SEC", 10.0)
    return max(0.1, min(cap, base * (2 ** max(0, attempt))))


def _is_retryable_gateway_error(status: int, body: str) -> bool:
    text = str(body or "").lower()
    if any(
        marker in text
        for marker in (
            "miner_app_unavailable",
            "miner not initialized",
            "not initialized",
        )
    ):
        return False
    if status in {404, 429, 503, 504}:
        return True
    return any(
        marker in text
        for marker in (
            "endpoint_not_found",
            "pod_capacity_exhausted",
            "pod_recreating",
            "miner pod is being recreated",
            "cold pod could not be started",
            "warming",
            "rate_limited",
            "upstream_unavailable",
        )
    )


def _build_url(base_url: str, path: str) -> str:
    normalized_base = str(base_url or "").strip().rstrip("/")
    normalized_path = str(path or "").strip()
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return f"{normalized_base}{normalized_path}"


def _is_active_pod_status(status: str | None) -> bool:
    return str(status or "").strip().lower() in {"running", "warming", "idle"}


async def _start_runpod_pod(*, endpoint_id: str, runpod_api_key: str, timeout: float) -> None:
    session = await get_async_client()
    async with session.post(
        f"https://rest.runpod.io/v1/pods/{endpoint_id}/start",
        headers={"Authorization": f"Bearer {runpod_api_key}"},
        timeout=ClientTimeout(total=max(timeout, 10.0)),
    ) as response:
        if response.status not in {200, 201}:
            text = await response.text()
            raise RuntimeError(f"RunPod pod start failed: status={response.status} body={text[:300]}")


async def _wait_for_managed_pod_health(*, base_url: str, timeout: float) -> None:
    session = await get_async_client()
    deadline = monotonic() + max(1.0, timeout)
    last_error = ""

    while monotonic() < deadline:
        for path in (_env_path("POD_HEALTH_PATH", "/healthz"), "/health"):
            try:
                async with session.get(
                    _build_url(base_url, path),
                    timeout=ClientTimeout(total=min(max(timeout, 1.0), 10.0)),
                ) as response:
                    if response.status == 200:
                        return
                    last_error = f"status={response.status} path={path}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{exc}"
        await asyncio.sleep(max(0.2, _env_float("POD_WAKE_POLL_INTERVAL_SECONDS", 2.0)))

    raise RuntimeError(f"RunPod pod health check timed out: {last_error or 'unknown'}")


def _build_bt_predict_headers(
    *,
    validator_identity: dict[str, Any],
    miner_hotkey: str,
    axon_ip: str,
    axon_port: str,
    timeout: float,
    payload: BBAudioMinerInitPayload | BBAudioMinerPredictPayload,
) -> dict[str, str]:
    nonce = time.time_ns()
    predict_payload = payload.model_dump(mode="json")
    body_hash = hashlib.sha256(
        dumps(predict_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    message = (
        f"{nonce}.{validator_identity['hotkey']}."
        f"{miner_hotkey}.{validator_identity['uuid']}.{body_hash}"
    )
    signature = f"0x{validator_identity['keypair'].sign(message).hex()}"
    return {
        "Content-Type": "application/json",
        "bt_header_dendrite_nonce": str(nonce),
        "bt_header_dendrite_hotkey": validator_identity["hotkey"],
        "bt_header_dendrite_signature": signature,
        "bt_header_dendrite_uuid": validator_identity["uuid"],
        "bt_header_dendrite_ip": validator_identity["external_ip"],
        "bt_header_dendrite_version": "7002000",
        "bt_header_axon_hotkey": miner_hotkey,
        "bt_header_axon_ip": axon_ip,
        "bt_header_axon_port": str(axon_port),
        "timeout": str(timeout),
        "name": type(payload).__name__,
        "computed_body_hash": body_hash,
    }


def _normalize_managed_predict_url(endpoint_url: str, predict_endpoint: str) -> str:
    url = (endpoint_url or "").strip()
    if not url:
        return ""

    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"

    endpoint = str(predict_endpoint or "predict").strip().lstrip("/")
    if not endpoint:
        return url

    parsed = urlparse(url)
    path = parsed.path or ""

    normalized_path = path.rstrip("/")
    if not normalized_path:
        normalized_path = f"/{endpoint}"
    elif normalized_path.split("/")[-1] != endpoint:
        normalized_path = f"{normalized_path}/{endpoint}"

    return urlunparse(parsed._replace(path=normalized_path))


def _normalize_gateway_runsync_url(endpoint_url: str) -> str:
    url = (endpoint_url or "").strip()
    if not url:
        return ""
    if not url.startswith("http://") and not url.startswith("https://"):
        return f"http://{url}"
    return url


def _derive_gateway_auth_url(
    gateway_runsync_url: str, auth_path: str, runsync_path: str
) -> str:
    normalized_runsync_url = _normalize_gateway_runsync_url(gateway_runsync_url)
    if not normalized_runsync_url:
        return ""

    parsed = urlparse(normalized_runsync_url)
    current_path = parsed.path or ""

    normalized_auth_path = str(auth_path or "/auth/token").strip()
    if not normalized_auth_path.startswith("/"):
        normalized_auth_path = f"/{normalized_auth_path}"

    normalized_runsync_path = str(runsync_path or "/runsync").strip()
    if not normalized_runsync_path.startswith("/"):
        normalized_runsync_path = f"/{normalized_runsync_path}"

    if current_path.endswith(normalized_runsync_path):
        base_path = current_path[: -len(normalized_runsync_path)]
        target_path = f"{base_path}{normalized_auth_path}"
    elif current_path.endswith("/runsync"):
        target_path = f"{current_path[: -len('/runsync')]}{normalized_auth_path}"
    else:
        target_path = normalized_auth_path

    return urlunparse(parsed._replace(path=target_path))


def _gateway_auth_cache_key(
    *,
    auth_url: str,
    validator_hotkey: str,
    miner_hotkey: str,
    miner_uid: int,
) -> str:
    return f"{auth_url}|{validator_hotkey}|{miner_hotkey}|{miner_uid}"


def _get_cached_gateway_auth_token(cache_key: str) -> str:
    now = time.time()
    with _GATEWAY_AUTH_TOKEN_LOCK:
        entry = _GATEWAY_AUTH_TOKEN_CACHE.get(cache_key)
        if not entry:
            return ""
        token, expires_at = entry
        if expires_at <= now:
            _GATEWAY_AUTH_TOKEN_CACHE.pop(cache_key, None)
            return ""
        return token


def _store_gateway_auth_token(cache_key: str, token: str, ttl_seconds: int) -> None:
    ttl = max(30, int(ttl_seconds))
    with _GATEWAY_AUTH_TOKEN_LOCK:
        _GATEWAY_AUTH_TOKEN_CACHE[cache_key] = (token, time.time() + ttl)


def _clear_gateway_auth_token(cache_key: str) -> None:
    with _GATEWAY_AUTH_TOKEN_LOCK:
        _GATEWAY_AUTH_TOKEN_CACHE.pop(cache_key, None)


async def _request_gateway_auth_token(
    *,
    auth_url: str,
    validator_identity: dict[str, Any],
    miner_hotkey: str,
    miner_uid: int,
    timeout: float,
) -> tuple[str, int, str]:
    session = await get_async_client()
    request_specs = [
        {
            "miner_hotkey": miner_hotkey,
            "uid": miner_uid,
        },
        {
            "hotkey": validator_identity["hotkey"],
            "miner_hotkey": miner_hotkey,
            "uid": miner_uid,
        },
        {
            "scope": "gateway",
        },
        {
            "hotkey": validator_identity["hotkey"],
        },
    ]
    payload: dict[str, Any] | None = None
    last_error = "gateway_auth_request_failed"

    for idx in range(0, len(request_specs), 2):
        auth_input_payload, body_fields = request_specs[idx], request_specs[idx + 1]
        timestamp_ms = int(time.time() * 1000)
        nonce = str(time.time_ns())
        gateway_message = f"{timestamp_ms}|{nonce}|{dumps(auth_input_payload, separators=(',', ':'), sort_keys=True)}"
        raw_signature = validator_identity["keypair"].sign(
            gateway_message.encode("utf-8")
        )
        if isinstance(raw_signature, (bytes, bytearray)):
            signature_hex = raw_signature.hex()
        else:
            signature_hex = str(raw_signature)
            if signature_hex.startswith("0x"):
                signature_hex = signature_hex[2:]

        request_body: dict[str, Any] = {
            **body_fields,
            "timestamp_ms": timestamp_ms,
            "nonce": nonce,
            "signature": signature_hex,
        }

        async with session.post(
            auth_url,
            headers={"Content-Type": "application/json"},
            json=request_body,
            timeout=ClientTimeout(total=timeout),
        ) as response:
            text = await response.text()
            if response.status != 200:
                last_error = f"status={response.status} body={text[:300]}"
                if idx == 0:
                    continue
                return "", 0, last_error

            try:
                loaded = loads(text)
            except Exception as exc:
                return "", 0, f"invalid_json:{exc}"

            if not isinstance(loaded, dict):
                return "", 0, "payload_not_object"
            payload = loaded
            break

    if payload is None:
        return "", 0, last_error

    token = payload.get("auth_token") or payload.get("token")
    if not isinstance(token, str) or not token.strip():
        return "", 0, "missing_auth_token"

    expires_in_raw = payload.get("expires_in", _GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S)
    try:
        expires_in = int(expires_in_raw)
    except Exception:
        expires_in = _GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S
    if expires_in <= 0:
        expires_in = _GATEWAY_AUTH_TOKEN_TTL_FALLBACK_S

    return token.strip(), expires_in, ""


def _get_validator_identity():
    global _VALIDATOR_IDENTITY_CACHE
    if _VALIDATOR_IDENTITY_CACHE is None:
        settings = get_settings()
        keypair = load_hotkey_keypair(
            settings.BITTENSOR_WALLET_COLD,
            settings.BITTENSOR_WALLET_HOT,
        )
        _VALIDATOR_IDENTITY_CACHE = {
            "keypair": keypair,
            "hotkey": keypair.ss58_address,
            "external_ip": networking.get_external_ip(),
            "uuid": str(uuid.uuid4()),
        }
        logger.info(
            "Validator identity initialized: hotkey=%s...", keypair.ss58_address[:8]
        )
    return _VALIDATOR_IDENTITY_CACHE


async def call_miner_axon_audio_endpoint(
    axon_ip: str,
    axon_port: int,
    payload: BBAudioMinerInitPayload | BBAudioMinerPredictPayload,
    miner_hotkey: str,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    settings = get_settings()
    if timeout is None:
        timeout = float(getattr(settings, "BB_MINER_TIMEOUT_SEC", 10))

    try:
        validator_identity = _get_validator_identity()
    except Exception as e:
        return {"error": f"validator_identity_error:{e}"}

    try:
        if getattr(settings, "BB_DEV_MODE", False):
            local_ip = getattr(settings, "BB_LOCAL_MINER_IP", "") or None
            if axon_ip in ("127.0.0.1", "localhost", "0.0.0.0") or (
                local_ip and axon_ip == local_ip
            ):
                logger.info(
                    "Dev mode: translating axon IP %s -> host.docker.internal", axon_ip
                )
                axon_ip = "host.docker.internal"
    except Exception:
        pass

    url = f"http://{axon_ip}:{axon_port}/{settings.BB_MINER_PREDICT_ENDPOINT}"
    session = await get_async_client()

    nonce = time.time_ns()
    body_hash = ""
    message = (
        f"{nonce}.{validator_identity['hotkey']}."
        f"{miner_hotkey}.{validator_identity['uuid']}.{body_hash}"
    )
    signature = f"0x{validator_identity['keypair'].sign(message).hex()}"
    headers = {
        "Content-Type": "application/json",
        "bt_header_dendrite_nonce": str(nonce),
        "bt_header_dendrite_hotkey": validator_identity["hotkey"],
        "bt_header_dendrite_signature": signature,
        "bt_header_dendrite_uuid": validator_identity["uuid"],
        "bt_header_dendrite_ip": validator_identity["external_ip"],
        "bt_header_dendrite_version": "7002000",
        "bt_header_axon_hotkey": miner_hotkey,
        "bt_header_axon_ip": axon_ip,
        "bt_header_axon_port": str(axon_port),
        "timeout": str(timeout),
        "name": type(payload).__name__,
        "computed_body_hash": body_hash,
    }

    try:
        async with session.post(
            url,
            headers=headers,
            json=payload.model_dump(mode="json"),
            timeout=ClientTimeout(total=timeout),
        ) as response:
            text = await response.text()
            if response.status != 200:
                logger.debug(
                    "Axon audio non-200: status=%s body='%s' url=%s miner_hk=%s",
                    response.status,
                    text[:200],
                    url,
                    (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                )
                return {"error": f"{response.status}:{text[:300]}"}
            try:
                data = loads(text)
            except Exception as e:
                return {"error": f"parse:{e}"}
            if isinstance(data, dict):
                return data
            return {"error": "invalid_payload_type"}
    except TimeoutError:
        return {"error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"error": f"{type(e).__name__}:{e}"}


async def call_managed_container_audio_endpoint(
    endpoint_url: str,
    payload: BBAudioMinerInitPayload | BBAudioMinerPredictPayload,
    miner_hotkey: str,
    endpoint_id: Optional[str] = None,
    endpoint_type: Optional[str] = None,
    status: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    settings = get_settings()
    if timeout is None:
        timeout = float(
            getattr(
                settings,
                "BB_ARENA_MINER_TIMEOUT_SEC",
                getattr(settings, "BB_MINER_TIMEOUT_SEC", 10),
            )
        )
    timeout = max(
        float(timeout),
        float(getattr(settings, "BB_S2S_INIT_TIMEOUT_SEC", 60.0)),
    )

    url = endpoint_url.strip()
    if not url:
        return {"error": "empty_endpoint_url"}

    try:
        validator_identity = _get_validator_identity()
    except Exception as e:
        return {"error": f"validator_identity_error:{e}"}

    endpoint_type_value = str(endpoint_type or "").strip().upper()
    pod_wake_timeout = max(timeout, _env_float("POD_WAKE_TIMEOUT_SECONDS", 90.0))
    if endpoint_type_value == "POD":
        runpod_api_key = str(getenv("RUNPOD_API_KEY", "") or "").strip()
        if not runpod_api_key:
            return {"error": "missing_runpod_api_key_for_pod_route"}
        if not endpoint_id:
            return {"error": "missing_endpoint_id_for_pod_route"}
        try:
            if not _is_active_pod_status(status):
                await _start_runpod_pod(
                    endpoint_id=str(endpoint_id),
                    runpod_api_key=runpod_api_key,
                    timeout=timeout,
                )
            await _wait_for_managed_pod_health(base_url=url, timeout=pod_wake_timeout)
        except Exception as e:
            return {"error": f"{type(e).__name__}:{e}"}

    url = _normalize_managed_predict_url(
        endpoint_url=url,
        predict_endpoint=str(getattr(settings, "BB_MINER_PREDICT_ENDPOINT", "predict")),
    )
    session = await get_async_client()
    parsed_endpoint = urlparse(endpoint_url if endpoint_url.startswith(("http://", "https://")) else f"http://{endpoint_url}")
    axon_ip = parsed_endpoint.hostname or "0.0.0.0"
    axon_port = str(parsed_endpoint.port or 0)
    headers = _build_bt_predict_headers(
        validator_identity=validator_identity,
        miner_hotkey=miner_hotkey,
        axon_ip=axon_ip,
        axon_port=axon_port,
        timeout=timeout,
        payload=payload,
    )
    try:
        async with session.post(
            url,
            headers=headers,
            json=payload.model_dump(mode="json"),
            timeout=ClientTimeout(total=timeout),
        ) as response:
            text = await response.text()
            if response.status != 200:
                logger.debug(
                    "Managed audio non-200: status=%s body='%s' url=%s miner_hk=%s",
                    response.status,
                    text[:200],
                    url,
                    (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                )
                return {"error": f"{response.status}:{text[:300]}"}
            try:
                data = loads(text)
            except Exception as e:
                return {"error": f"parse:{e}"}
            if isinstance(data, dict):
                return data
            return {"error": "invalid_payload_type"}
    except TimeoutError:
        return {"error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"error": f"{type(e).__name__}:{e}"}


async def call_gateway_runsync_audio_endpoint(
    gateway_url: str,
    payload: BBAudioMinerInitPayload | BBAudioMinerPredictPayload,
    miner_hotkey: str,
    miner_uid: Optional[int] = None,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    settings = get_settings()
    if timeout is None:
        timeout = float(
            getattr(
                settings,
                "BB_ARENA_MINER_TIMEOUT_SEC",
                getattr(settings, "BB_MINER_TIMEOUT_SEC", 10),
            )
        )
    timeout = max(
        float(timeout),
        float(getattr(settings, "BB_S2S_INIT_TIMEOUT_SEC", 60.0)),
        float(getattr(settings, "BB_ARENA_GATEWAY_TIMEOUT_SEC", 300.0)),
    )

    url = _normalize_gateway_runsync_url(gateway_url)
    if not url:
        return {"error": "empty_gateway_url"}

    try:
        validator_identity = _get_validator_identity()
    except Exception as e:
        return {"error": f"validator_identity_error:{e}"}

    if not isinstance(miner_uid, int) or miner_uid < 0:
        return {"error": "missing_miner_uid_for_gateway"}

    predict_payload = payload.model_dump(mode="json")
    body_hash = hashlib.sha256(
        dumps(predict_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    dendrite_nonce = time.time_ns()
    dendrite_message = (
        f"{dendrite_nonce}.{validator_identity['hotkey']}."
        f"{miner_hotkey}.{validator_identity['uuid']}.{body_hash}"
    )
    raw_dendrite_signature = validator_identity["keypair"].sign(dendrite_message)
    if isinstance(raw_dendrite_signature, (bytes, bytearray)):
        dendrite_signature_hex = raw_dendrite_signature.hex()
    else:
        dendrite_signature_hex = str(raw_dendrite_signature)
        if dendrite_signature_hex.startswith("0x"):
            dendrite_signature_hex = dendrite_signature_hex[2:]

    bt_headers: dict[str, str] = {
        "bt_header_dendrite_nonce": str(dendrite_nonce),
        "bt_header_dendrite_hotkey": validator_identity["hotkey"],
        "bt_header_dendrite_signature": f"0x{dendrite_signature_hex}",
        "bt_header_dendrite_uuid": validator_identity["uuid"],
        "bt_header_dendrite_ip": validator_identity["external_ip"],
        "bt_header_dendrite_version": "7002000",
        "bt_header_axon_hotkey": miner_hotkey,
        "bt_header_axon_ip": "0.0.0.0",
        "bt_header_axon_port": "0",
        "timeout": str(timeout),
        "name": type(payload).__name__,
        "computed_body_hash": body_hash,
    }
    gateway_input_payload: dict[str, Any] = {
        "predict_payload": predict_payload,
        "bt_headers": bt_headers,
    }

    gateway_auth_url = _derive_gateway_auth_url(
        gateway_runsync_url=url,
        auth_path=getattr(settings, "BB_ARENA_GATEWAY_AUTH_API_PATH", "/auth/token"),
        runsync_path=getattr(settings, "BB_ARENA_RUNSYNC_API_PATH", "/runsync"),
    )
    if not gateway_auth_url:
        return {"error": "empty_gateway_auth_url"}

    auth_cache_key = _gateway_auth_cache_key(
        auth_url=gateway_auth_url,
        validator_hotkey=str(validator_identity["hotkey"]),
        miner_hotkey=str(miner_hotkey),
        miner_uid=int(miner_uid),
    )

    session = await get_async_client()
    try:
        auth_token = _get_cached_gateway_auth_token(auth_cache_key)

        deadline = monotonic() + max(1.0, float(timeout))
        request_attempt = 0
        auth_attempt = 0
        while True:
            if not auth_token:
                token, expires_in, auth_error = await _request_gateway_auth_token(
                    auth_url=gateway_auth_url,
                    validator_identity=validator_identity,
                    miner_hotkey=str(miner_hotkey),
                    miner_uid=int(miner_uid),
                    timeout=float(timeout),
                )
                if not token:
                    return {"error": f"gateway_auth_failed:{auth_error}"}
                auth_token = token
                _store_gateway_auth_token(auth_cache_key, auth_token, expires_in)

            request_attempt += 1
            request_body: dict[str, Any] = {
                "input": gateway_input_payload,
                "auth_token": auth_token,
                "uid": miner_uid,
                "miner_hotkey": miner_hotkey,
                "request_id": (
                    f"gw-audio:{type(payload).__name__}:{miner_uid}:"
                    f"{request_attempt}:{uuid.uuid4().hex}"
                ),
            }
            post_timeout = max(1.0, min(float(timeout), deadline - monotonic()))
            logger.info(
                "Gateway audio runsync request begin url=%s miner_hk=%s miner_uid=%s timeout=%.2fs attempt=%s",
                url,
                (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                miner_uid,
                post_timeout,
                request_attempt,
            )
            async with session.post(
                url,
                headers={"Content-Type": "application/json"},
                json=request_body,
                timeout=ClientTimeout(total=post_timeout),
            ) as response:
                text = await response.text()
                if response.status == 401 and auth_attempt == 0:
                    _clear_gateway_auth_token(auth_cache_key)
                    auth_token = ""
                    auth_attempt += 1
                    continue
                if response.status != 200:
                    remaining = deadline - monotonic()
                    if _is_retryable_gateway_error(response.status, text) and remaining > 1.0:
                        delay = min(_gateway_retry_delay_seconds(request_attempt - 1), max(0.1, remaining - 0.5))
                        logger.info(
                            "Gateway audio retryable response: status=%s delay=%.2fs remaining=%.2fs body='%s' url=%s miner_hk=%s",
                            response.status,
                            delay,
                            remaining,
                            text[:200],
                            url,
                            (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.debug(
                        "Gateway audio non-200: status=%s body='%s' url=%s miner_hk=%s",
                        response.status,
                        text[:200],
                        url,
                        (miner_hotkey[:16] + "...") if miner_hotkey else "?",
                    )
                    return {"error": f"{response.status}:{text[:300]}"}
                try:
                    data = loads(text)
                except Exception as e:
                    return {"error": f"parse:{e}"}
                if isinstance(data, dict):
                    output = data.get("output")
                    if isinstance(output, dict):
                        return output
                    return data
                return {"error": "invalid_payload_type"}
            if monotonic() >= deadline:
                break
        return {"error": f"gateway_retry_timeout after {timeout}s"}
    except TimeoutError:
        return {"error": f"timeout after {timeout}s"}
    except Exception as e:
        return {"error": f"{type(e).__name__}:{e}"}


async def call_managed_route_audio_endpoint(
    *,
    route: Any,
    payload: BBAudioMinerInitPayload | BBAudioMinerPredictPayload,
    miner_hotkey: str,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    endpoint_url = str(getattr(route, "endpoint_url", "") or "").strip()
    provider = str(getattr(route, "provider", "") or "").strip().lower()
    miner_uid_raw = getattr(route, "miner_uid", None)
    miner_uid: Optional[int] = None
    if miner_uid_raw is not None:
        try:
            miner_uid = int(miner_uid_raw)
        except Exception:
            miner_uid = None

    if provider == "gateway":
        return await call_gateway_runsync_audio_endpoint(
            gateway_url=endpoint_url,
            payload=payload,
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            timeout=timeout,
        )

    return await call_managed_container_audio_endpoint(
        endpoint_url=endpoint_url,
        payload=payload,
        miner_hotkey=miner_hotkey,
        endpoint_id=getattr(route, "endpoint_id", None),
        endpoint_type=getattr(route, "endpoint_type", None),
        status=getattr(route, "status", None),
        timeout=timeout,
    )
