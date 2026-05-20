from dataclasses import dataclass
from json import loads
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from aiohttp import ClientTimeout

from babelbit.utils.async_clients import get_async_client
from babelbit.utils.miner_registry import Miner, get_miners_from_registry
from babelbit.utils.predict_engine import (
    _clear_gateway_auth_token,
    _derive_gateway_auth_url,
    _get_cached_gateway_auth_token,
    _get_validator_identity,
    _request_gateway_auth_token,
    _store_gateway_auth_token,
)
from babelbit.utils.settings import get_settings

logger = getLogger(__name__)


@dataclass
class ManagedRoute:
    miner_hotkey: str
    endpoint_url: str
    miner_uid: Optional[int] = None
    provider: str = "managed_container"
    endpoint_id: Optional[str] = None
    endpoint_type: Optional[str] = None
    status: Optional[str] = None
    container_name: Optional[str] = None
    last_seen_at: Optional[str] = None


def _extract_metadata_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str) and metadata.strip():
        try:
            parsed = loads(metadata)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _normalize_route_item(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(item)
    metadata = _extract_metadata_dict(item)
    if metadata:
        normalized["__metadata"] = metadata
        for key, value in metadata.items():
            if key not in normalized:
                normalized[key] = value
    return normalized


def _extract_hotkey(item: Dict[str, Any]) -> Optional[str]:
    for key in ("miner_hotkey", "hotkey", "axon_hotkey", "wallet_hotkey"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_endpoint_url(item: Dict[str, Any]) -> Optional[str]:
    settings = get_settings()

    for key in ("endpoint_url", "url", "container_url", "endpoint"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            endpoint_url = value.strip()
            if endpoint_url.startswith("http://") or endpoint_url.startswith("https://"):
                return endpoint_url
            return f"http://{endpoint_url}"

    host = item.get("ip") or item.get("host") or item.get("container_ip")
    port = item.get("port") or item.get("container_port")
    if not host or not port:
        return None

    try:
        port_int = int(port)
        if port_int <= 0:
            return None
    except Exception:
        return None

    path = item.get("path") or item.get("endpoint_path") or item.get("predict_path") or settings.BB_MINER_PREDICT_ENDPOINT
    path_str = str(path or "").lstrip("/")
    if path_str:
        return f"http://{host}:{port_int}/{path_str}"
    return f"http://{host}:{port_int}"


def _build_gateway_runsync_url(api_base: str, runsync_path: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    path = (runsync_path or "/runsync").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"


def _extract_status_reason(item: Dict[str, Any]) -> str:
    for key in ("reason", "status_reason", "termination_reason"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    metadata = item.get("__metadata")
    if isinstance(metadata, dict):
        for key in ("reason", "status_reason", "termination_reason"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return ""


def _is_soft_terminated_route(item: Dict[str, Any]) -> bool:
    status = str(item.get("status") or "").strip().lower()
    if status != "terminated":
        return False

    reason = _extract_status_reason(item)
    is_idle_not_listed = ("not_in_list" in reason) or ("idle" in reason)
    if not is_idle_not_listed:
        return False
    endpoint_url = _extract_endpoint_url(item)
    return bool(endpoint_url)


def _extract_provider(
    item: Dict[str, Any],
    *,
    endpoint_url: Optional[str],
) -> str:
    for key in ("provider", "cloud_provider", "platform"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if endpoint_url:
        return "managed_container"
    return "managed_container"


def _detect_endpoint_source_key(item: Dict[str, Any]) -> Optional[str]:
    for key in ("endpoint_url", "url", "container_url", "endpoint"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return key

    for key in ("endpoint_id",):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return key

    host = item.get("ip") or item.get("host") or item.get("container_ip")
    port = item.get("port") or item.get("container_port")
    if host and port:
        return "host_port"
    return None


def _to_managed_route(item: Dict[str, Any]) -> Optional[ManagedRoute]:
    hotkey = _extract_hotkey(item)
    endpoint_url = _extract_endpoint_url(item)
    if not hotkey or not endpoint_url:
        return None

    uid_val = item.get("miner_uid")
    if uid_val is None:
        uid_val = item.get("uid")
    miner_uid: Optional[int] = None
    if uid_val is not None:
        try:
            miner_uid = int(uid_val)
        except Exception:
            miner_uid = None

    provider = _extract_provider(
        item,
        endpoint_url=endpoint_url,
    )

    status_val = item.get("status")
    endpoint_id = item.get("endpoint_id")
    endpoint_type = item.get("endpoint_type")
    container_name = item.get("container_name")
    last_seen_at = item.get("last_seen_at")

    return ManagedRoute(
        miner_hotkey=hotkey,
        endpoint_url=endpoint_url,
        miner_uid=miner_uid,
        provider=provider,
        endpoint_id=str(endpoint_id) if endpoint_id is not None else None,
        endpoint_type=str(endpoint_type) if endpoint_type is not None else None,
        status=str(status_val) if status_val is not None else None,
        container_name=str(container_name) if container_name is not None else None,
        last_seen_at=str(last_seen_at) if last_seen_at is not None else None,
    )


def _canonicalize_discovery_path(path: str) -> str:
    path_str = str(path or "").strip()
    if not path_str:
        return "/list_arena_miners"
    if not path_str.startswith("/"):
        path_str = f"/{path_str}"

    while "//" in path_str:
        path_str = path_str.replace("//", "/")

    if path_str.endswith("/live_containers"):
        return f"{path_str[:-len('/live_containers')]}/list_arena_miners"
    if path_str == "/live_containers":
        return "/list_arena_miners"
    return path_str


def _candidate_discovery_paths(api_base: str, api_path: str) -> List[str]:
    parsed_base = urlparse(api_base)
    base_has_v1_prefix = parsed_base.path.rstrip("/").endswith("/v1")

    normalized_path = _canonicalize_discovery_path(api_path)
    candidates: List[str] = []

    def _add(path_value: str) -> None:
        if path_value and path_value not in candidates:
            candidates.append(path_value)

    if base_has_v1_prefix and normalized_path.startswith("/v1/"):
        _add(normalized_path[3:])
        _add(normalized_path)
        return candidates

    _add(normalized_path)
    if normalized_path.startswith("/v1/"):
        _add(normalized_path[3:])
    else:
        _add(f"/v1{normalized_path}")
    return candidates


def _discovery_status_filters(
    status: Optional[str],
    default_status: str,
) -> List[str]:
    req_status = status if status is not None else default_status
    status_filters = [str(req_status or "").strip()] if req_status is not None else [""]
    if len(status_filters) == 1 and "," in status_filters[0]:
        status_filters = [part.strip() for part in status_filters[0].split(",") if part.strip()]
    status_filters = [value for value in status_filters if value]
    if not status_filters:
        status_filters = ["running"]
    # Default Round2 discovery should keep warmable POD routes visible so the
    # direct predictor can start/wait on them instead of filtering them out.
    if status is None and status_filters == ["running"]:
        return ["running", "warming", "idle", "unhealthy", "unavailable", "stopped"]
    return status_filters


def _looks_like_auth_token_error(status: int, text: str) -> bool:
    if status != 401:
        return False
    normalized = (text or "").lower()
    return "invalid_auth_token" in normalized or "auth token" in normalized


def _gateway_discovery_auth_cache_key(auth_url: str, validator_hotkey: str) -> str:
    return f"discovery|{auth_url}|{validator_hotkey}"


async def _get_gateway_discovery_headers(
    *,
    auth_url: str,
    timeout: float,
    force_refresh: bool = False,
) -> Dict[str, str]:
    validator_identity = _get_validator_identity()
    auth_cache_key = _gateway_discovery_auth_cache_key(
        auth_url=auth_url,
        validator_hotkey=str(validator_identity["hotkey"]),
    )

    auth_token = "" if force_refresh else _get_cached_gateway_auth_token(auth_cache_key)
    if not auth_token:
        token, expires_in, auth_error = await _request_gateway_auth_token(
            auth_url=auth_url,
            validator_identity=validator_identity,
            miner_hotkey="discovery",
            miner_uid=0,
            timeout=timeout,
        )
        if not token:
            raise RuntimeError(f"arena miners discovery auth failed: {auth_error}")
        auth_token = token
        _store_gateway_auth_token(auth_cache_key, auth_token, expires_in)

    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }


async def _request_discovery_payload(
    session,
    *,
    url: str,
    auth_url: str,
    params: Dict[str, Any],
    timeout: float,
) -> Tuple[int, str, Optional[Dict[str, Any]]]:
    auth_headers = await _get_gateway_discovery_headers(auth_url=auth_url, timeout=timeout)
    request_kwargs: Dict[str, Any] = {
        "headers": auth_headers,
        "params": params,
        "timeout": ClientTimeout(total=timeout),
    }
    attempted_reauth_retry = False

    while True:
        async with session.get(url, **request_kwargs) as response:
            text = await response.text()
            payload: Optional[Dict[str, Any]] = None
            if response.status == 200:
                try:
                    loaded = loads(text) if text.strip() else await response.json()
                except Exception as exc:
                    raise RuntimeError(
                        f"arena miners API returned invalid JSON: {exc}"
                    ) from exc
                if not isinstance(loaded, dict):
                    raise RuntimeError("arena miners API payload must be an object")
                payload = loaded

            if _looks_like_auth_token_error(response.status, text) and not attempted_reauth_retry:
                validator_identity = _get_validator_identity()
                _clear_gateway_auth_token(
                    _gateway_discovery_auth_cache_key(
                        auth_url=auth_url,
                        validator_hotkey=str(validator_identity["hotkey"]),
                    )
                )
                request_kwargs["headers"] = await _get_gateway_discovery_headers(
                    auth_url=auth_url,
                    timeout=timeout,
                    force_refresh=True,
                )
                attempted_reauth_retry = True
                logger.info(
                    "Arena discovery auth received 401; refreshed gateway auth and retrying %s once",
                    url,
                )
                continue

            return response.status, text, payload


async def fetch_live_containers(
    *,
    path: Optional[str] = None,
    status: Optional[str] = None,
    window_seconds: Optional[int] = None,
    timeout: Optional[float] = None,
) -> List[Dict[str, Any]]:
    settings = get_settings()

    gateway_base = (getattr(settings, "BB_ARENA_GATEWAY_URL", "") or settings.BB_SUBMIT_API_URL or "").strip().rstrip("/")
    if not gateway_base:
        raise RuntimeError("BB_ARENA_GATEWAY_URL or BB_SUBMIT_API_URL is required for Round2 live container discovery")

    configured_api_path = (path or settings.BB_ARENA_CONTAINERS_API_PATH or "/list_arena_miners").strip()
    api_path = _canonicalize_discovery_path(configured_api_path)
    api_path_candidates = _candidate_discovery_paths(gateway_base, api_path)

    if configured_api_path != api_path:
        logger.info(
            "Normalized Round2 discovery path from %s to %s",
            configured_api_path,
            api_path,
        )

    gateway_runsync_url = _build_gateway_runsync_url(
        api_base=gateway_base,
        runsync_path=getattr(settings, "BB_ARENA_RUNSYNC_API_PATH", "/runsync"),
    )
    gateway_auth_url = _derive_gateway_auth_url(
        gateway_runsync_url=gateway_runsync_url,
        auth_path=getattr(settings, "BB_ARENA_GATEWAY_AUTH_API_PATH", "/auth/token"),
        runsync_path=getattr(settings, "BB_ARENA_RUNSYNC_API_PATH", "/runsync"),
    )
    if not gateway_auth_url:
        raise RuntimeError("BB_ARENA_GATEWAY_AUTH_API_PATH resolved to an empty auth URL")

    req_window_seconds = (
        int(window_seconds)
        if window_seconds is not None
        else int(settings.BB_ARENA_CONTAINERS_WINDOW_SECONDS)
    )
    req_timeout = (
        float(timeout)
        if timeout is not None
        else float(settings.BB_ARENA_CONTAINERS_TIMEOUT_SEC)
    )

    session = await get_async_client()
    status_filters = _discovery_status_filters(
        status,
        str(getattr(settings, "BB_ARENA_CONTAINERS_STATUS", "running") or "running"),
    )

    attempted_urls: List[str] = []
    last_404_error: Optional[str] = None

    for candidate_path in api_path_candidates:
        url = f"{gateway_base}{candidate_path}"
        attempted_urls.append(url)

        merged_containers: List[Dict[str, Any]] = []
        seen_rows: set[tuple[str, str, str, str]] = set()
        used_arena_miners_shape = False
        path_missing = False

        for status_filter in status_filters:
            params: Dict[str, Any] = {"window_seconds": req_window_seconds}
            if status_filter:
                params["status"] = status_filter
            response_status, response_text, payload = await _request_discovery_payload(
                session,
                url=url,
                auth_url=gateway_auth_url,
                params=params,
                timeout=req_timeout,
            )
            if response_status == 404:
                path_missing = True
                last_404_error = (
                    f"arena miners API failed 404 at {url}: {response_text[:300]}"
                )
                logger.warning("Round2 discovery path not found: %s", url)
                break
            if response_status != 200:
                raise RuntimeError(
                    f"arena miners API failed {response_status} at {url}: {response_text[:300]}"
                )
            if payload is None:
                raise RuntimeError("arena miners API payload must be an object")

            rows: list[Dict[str, Any]] = []
            if isinstance(payload.get("containers"), list):
                containers = payload.get("containers")
                if not isinstance(containers, list):
                    raise RuntimeError("arena miners API field 'containers' must be a list")
                for row in containers:
                    if isinstance(row, dict):
                        rows.append(row)
            elif isinstance(payload.get("miners"), list):
                used_arena_miners_shape = True
                miners = payload.get("miners")
                if not isinstance(miners, list):
                    raise RuntimeError("arena miners API field 'miners' must be a list")
                for raw_miner in miners:
                    if not isinstance(raw_miner, dict):
                        continue
                    hotkey = raw_miner.get("hotkey")
                    if not isinstance(hotkey, str) or not hotkey.strip():
                        continue
                    uid_val = raw_miner.get("uid")
                    try:
                        uid = int(uid_val)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "miner_hotkey": hotkey.strip(),
                            "miner_uid": uid,
                            "status": "running",
                            "provider": "gateway",
                            "endpoint_url": gateway_runsync_url,
                            "container_name": f"arena-miner-{uid}",
                        }
                    )
            else:
                raise RuntimeError("arena miners API must include either 'containers' or 'miners' list")

            count = payload.get("count")
            if isinstance(count, int) and count != len(rows):
                logger.warning(
                    "arena miners API count mismatch: count=%s len(rows)=%s",
                    count,
                    len(rows),
                )

            for raw_item in rows:
                dedupe_key = (
                    str(raw_item.get("submission_id") or ""),
                    str(raw_item.get("miner_hotkey") or raw_item.get("hotkey") or ""),
                    str(raw_item.get("container_name") or ""),
                    str(raw_item.get("endpoint_url") or raw_item.get("url") or ""),
                )
                if dedupe_key in seen_rows:
                    continue
                seen_rows.add(dedupe_key)
                merged_containers.append(raw_item)

            if used_arena_miners_shape:
                # list_arena_miners already returns the filtered miner set.
                break

        if path_missing:
            continue

        logger.info(
            "Fetched %d live containers from %s (status filters=%s)",
            len(merged_containers),
            url,
            ",".join(status_filters),
        )
        return merged_containers

    if last_404_error is not None:
        raise RuntimeError(f"{last_404_error} (attempted={','.join(attempted_urls)})")
    raise RuntimeError(f"arena miners API discovery failed (attempted={','.join(attempted_urls)})")


async def resolve_round2_routes(
    *,
    netuid: int,
    subtensor=None,
    containers: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Miner], Dict[str, ManagedRoute]]:
    if containers is None:
        containers = await fetch_live_containers()

    miners_by_uid = await get_miners_from_registry(netuid, subtensor=subtensor)
    miners_by_hotkey: Dict[str, Miner] = {m.hotkey: m for m in miners_by_uid.values()}

    hotkey_to_route: Dict[str, ManagedRoute] = {}
    dropped_malformed = 0

    for item in containers:
        if not isinstance(item, dict):
            dropped_malformed += 1
            continue

        normalized_item = _normalize_route_item(item)
        status_value = str(normalized_item.get("status") or "").strip().lower()
        if status_value == "terminated" and not _is_soft_terminated_route(normalized_item):
            logger.debug(
                "Round2 route ignored due to terminated status: hotkey=%s endpoint=%s",
                normalized_item.get("miner_hotkey") or normalized_item.get("hotkey") or "?",
                normalized_item.get("endpoint_url") or normalized_item.get("url") or "?",
            )
            continue

        route = _to_managed_route(normalized_item)
        if route is None:
            dropped_malformed += 1
            continue

        matched_miner = miners_by_hotkey.get(route.miner_hotkey)
        if matched_miner is None and route.miner_uid is not None:
            matched_miner = miners_by_uid.get(route.miner_uid)

        if matched_miner is None:
            logger.debug(
                "Round2 container route ignored (hotkey not in metagraph): hotkey=%s endpoint=%s",
                route.miner_hotkey,
                route.endpoint_url,
            )
            continue

        if route.miner_hotkey != matched_miner.hotkey:
            logger.warning(
                "Round2 route hotkey mismatch for uid=%s; canonicalizing %s -> %s",
                route.miner_uid,
                route.miner_hotkey,
                matched_miner.hotkey,
            )
            route.miner_hotkey = matched_miner.hotkey

        if route.miner_uid is None:
            route.miner_uid = matched_miner.uid

        if route.miner_hotkey in hotkey_to_route:
            logger.warning(
                "Duplicate Round2 route for hotkey=%s; keeping first endpoint=%s",
                route.miner_hotkey,
                hotkey_to_route[route.miner_hotkey].endpoint_url,
            )
            continue

        endpoint_source_key = _detect_endpoint_source_key(normalized_item) or "unknown"
        logger.debug(
            "Round2 route accepted: hotkey=%s endpoint=%s source_key=%s",
            route.miner_hotkey,
            route.endpoint_url,
            endpoint_source_key,
        )
        hotkey_to_route[route.miner_hotkey] = route

    if dropped_malformed:
        logger.warning("Dropped %d malformed container rows while resolving Round2 routes", dropped_malformed)

    round2_miners = [miner for miner in miners_by_uid.values() if miner.hotkey in hotkey_to_route]

    logger.info(
        "Resolved Round2 routes for %d/%d on-chain miners",
        len(round2_miners),
        len(miners_by_uid),
    )
    return round2_miners, hotkey_to_route
