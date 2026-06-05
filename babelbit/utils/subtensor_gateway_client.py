from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import aiohttp

from babelbit.utils.settings import get_settings

_SESSIONS: dict[int, aiohttp.ClientSession] = {}


def _loop_key() -> int:
    return id(asyncio.get_running_loop())


async def _get_session() -> aiohttp.ClientSession:
    settings = get_settings()
    key = _loop_key()
    sess = _SESSIONS.get(key)
    if sess is None or sess.closed:
        sess = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.SUBTENSOR_GATEWAY_TIMEOUT_S)
        )
        _SESSIONS[key] = sess
    return sess


def _weights_timeout_s(retries: int, delay_s: float) -> float:
    settings = get_settings()
    retry_budget = max(1, int(retries)) * (max(0.0, float(delay_s)) + 20.0)
    return max(float(settings.SUBTENSOR_GATEWAY_TIMEOUT_S), retry_budget)


async def close_gateway_clients() -> None:
    for sess in list(_SESSIONS.values()):
        try:
            if sess and not sess.closed:
                await sess.close()
        except Exception:
            pass
    _SESSIONS.clear()


class SubtensorGatewayClient:
    def __init__(self, base_url: str | None = None):
        settings = get_settings()
        self.base_url = (base_url or settings.SUBTENSOR_GATEWAY_URL).rstrip("/")

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        allow_error: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        sess = await _get_session()
        timeout = None if timeout_s is None else aiohttp.ClientTimeout(total=timeout_s)
        async with sess.post(
            f"{self.base_url}{path}", json=payload, timeout=timeout
        ) as resp:
            text = await resp.text()
            try:
                body = json.loads(text)
            except Exception:
                body = None
            if resp.status >= 400 and not allow_error:
                raise RuntimeError(
                    f"gateway {path} failed status={resp.status} body={text[:300]}"
                )
            if not isinstance(body, dict):
                raise RuntimeError(
                    f"gateway {path} returned invalid json body={text[:300]}"
                )
            if allow_error:
                body.setdefault("status", resp.status)
            return body

    async def get_current_block(self) -> int:
        body = await self._post_json("/v1/block/current", {})
        return int(body["block"])

    async def wait_for_block(self, timeout_s: int | None = None) -> int:
        body = await self._post_json("/v1/block/wait", {"timeout_s": timeout_s})
        return int(body["block"])

    async def metagraph_snapshot(
        self, netuid: int, lite: bool = False
    ) -> dict[str, Any]:
        return await self._post_json(
            "/v1/metagraph/snapshot", {"netuid": int(netuid), "lite": bool(lite)}
        )

    async def metagraph_object(self, netuid: int, lite: bool = False) -> Any:
        snap = await self.metagraph_snapshot(netuid=netuid, lite=lite)
        return SimpleNamespace(
            hotkeys=snap.get("hotkeys", []),
            last_update=snap.get("last_update", []),
            axons=snap.get("axons", []),
            block=snap.get("block"),
        )

    async def registry_context(self, netuid: int, lite: bool = False) -> dict[str, Any]:
        return await self._post_json(
            "/v1/registry/context", {"netuid": int(netuid), "lite": bool(lite)}
        )

    async def set_weights_and_confirm(
        self,
        *,
        netuid: int,
        uids: list[int],
        weights: list[float],
        wait_for_inclusion: bool = False,
        retries: int = 20,
        delay_s: float = 2.0,
        wallet_hotkey: str | None = None,
    ) -> bool:
        payload = {
            "netuid": int(netuid),
            "uids": [int(u) for u in uids],
            "weights": [float(w) for w in weights],
            "wait_for_inclusion": bool(wait_for_inclusion),
            "retries": int(retries),
            "delay_s": float(delay_s),
            "wallet_hotkey": wallet_hotkey,
        }
        body = await self._post_json(
            "/v1/weights/set_and_confirm",
            payload,
            timeout_s=_weights_timeout_s(retries, delay_s),
        )
        return bool(body.get("success", False))

    async def set_weights_and_confirm_response(
        self,
        *,
        netuid: int,
        uids: list[int],
        weights: list[float],
        wait_for_inclusion: bool = False,
        retries: int = 20,
        delay_s: float = 2.0,
        wallet_hotkey: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "netuid": int(netuid),
            "uids": [int(u) for u in uids],
            "weights": [float(w) for w in weights],
            "wait_for_inclusion": bool(wait_for_inclusion),
            "retries": int(retries),
            "delay_s": float(delay_s),
            "wallet_hotkey": wallet_hotkey,
        }
        return await self._post_json(
            "/v1/weights/set_and_confirm",
            payload,
            allow_error=True,
            timeout_s=_weights_timeout_s(retries, delay_s),
        )
