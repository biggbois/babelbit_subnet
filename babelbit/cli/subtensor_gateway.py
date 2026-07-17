from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import time
from typing import Any

import bittensor as bt
from aiohttp import web

from babelbit.utils.settings import get_settings

logger = logging.getLogger("sv-subtensor-gateway")

_BLOCK_WAIT_POLL_INTERVAL_S = 0.5


class _GatewayState:
    def __init__(self):
        self.subtensor = None
        self.subtensor_lock = asyncio.Lock()
        self.meta_cache: dict[tuple[int, bool], Any] = {}
        self.meta_locks: dict[tuple[int, bool], asyncio.Lock] = {}
        self.created_at: float | None = None

    def _meta_lock(self, key: tuple[int, bool]) -> asyncio.Lock:
        lock = self.meta_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.meta_locks[key] = lock
        return lock

    async def get_subtensor(self):
        async with self.subtensor_lock:
            if self.subtensor is None:
                settings = get_settings()
                primary = settings.BITTENSOR_SUBTENSOR_ENDPOINT
                fallback = settings.BITTENSOR_SUBTENSOR_FALLBACK
                logger.info("[gateway] connecting subtensor primary=%s", primary)
                self.subtensor = bt.AsyncSubtensor(network=primary)
                try:
                    await self.subtensor.initialize()
                    self.created_at = time.monotonic()
                    logger.info("[gateway] subtensor connected primary=%s", primary)
                except Exception as e:
                    logger.warning("[gateway] primary failed: %s", e)
                    self.subtensor = bt.AsyncSubtensor(network=fallback)
                    await self.subtensor.initialize()
                    self.created_at = time.monotonic()
                    logger.info("[gateway] subtensor connected fallback=%s", fallback)
            return self.subtensor

    async def reset_subtensor(self):
        async with self.subtensor_lock:
            if self.subtensor is not None:
                try:
                    await self.subtensor.close()
                except Exception:
                    logger.debug("[gateway] subtensor close failed", exc_info=True)
            self.subtensor = None
            self.created_at = None
            self.meta_cache.clear()

    async def metagraph(self, netuid: int, lite: bool):
        key = (int(netuid), bool(lite))
        lock = self._meta_lock(key)
        async with lock:
            st = await self.get_subtensor()
            meta = self.meta_cache.get(key)
            if meta is None:
                meta = await st.metagraph(netuid, lite=lite)
                self.meta_cache[key] = meta
            else:
                await meta.sync(subtensor=st, lite=lite)
            return meta


STATE = _GatewayState()


async def _get_current_block_resilient(*, retries: int = 2) -> int:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            st = await STATE.get_subtensor()
            block = await st.get_current_block()
            return int(block)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[gateway] current block failed attempt %d/%d: %s: %s",
                attempt + 1,
                retries,
                type(exc).__name__,
                exc,
            )
            await STATE.reset_subtensor()
    assert last_error is not None
    raise last_error


async def _wait_for_next_block_polling(timeout_s: float | None) -> int:
    start_block = await _get_current_block_resilient()
    deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)

    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError("timed out waiting for next block")
            await asyncio.sleep(min(_BLOCK_WAIT_POLL_INTERVAL_S, remaining))
        else:
            await asyncio.sleep(_BLOCK_WAIT_POLL_INTERVAL_S)

        block = await _get_current_block_resilient()
        if block > start_block:
            return block


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    try:
        return [int(v) for v in values]
    except Exception:
        return []


def _serialize_axons(meta: Any) -> list[dict[str, Any]]:
    out = []
    axons = getattr(meta, "axons", []) or []
    for axon in axons:
        ip = getattr(axon, "ip", None)
        port = getattr(axon, "port", None)
        out.append({"ip": ip, "port": int(port) if port is not None else None})
    return out


def _meta_snapshot(meta: Any) -> dict[str, Any]:
    return {
        "block": int(getattr(meta, "block", 0) or 0),
        "hotkeys": list(getattr(meta, "hotkeys", []) or []),
        "last_update": _to_int_list(getattr(meta, "last_update", [])),
        "axons": _serialize_axons(meta),
    }


def _serialize_commitments(commits: dict[str, Any]) -> dict[str, list[list[Any]]]:
    out: dict[str, list[list[Any]]] = {}
    for hotkey, arr in (commits or {}).items():
        rows: list[list[Any]] = []
        for item in arr or []:
            try:
                block, data = item
            except Exception:
                continue
            if isinstance(data, bytes):
                try:
                    data = data.decode("utf-8")
                except Exception:
                    data = data.hex()
            rows.append([int(block), str(data)])
        out[str(hotkey)] = rows
    return out


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _set_weights_result(result: Any) -> tuple[bool, str | None]:
    if isinstance(result, tuple):
        success = bool(result[0]) if result else False
        message = str(result[1]) if len(result) > 1 else None
        return success, message
    return bool(result), None


def _top_nonzero_weights(
    uids: list[int], weights: list[float], *, limit: int = 10
) -> list[dict[str, Any]]:
    pairs = [
        {"uid": int(uid), "weight": float(weight)}
        for uid, weight in zip(uids, weights)
        if float(weight) > 0.0
    ]
    return sorted(pairs, key=lambda item: item["weight"], reverse=True)[:limit]


async def run_subtensor_gateway() -> None:
    settings = get_settings()
    host = settings.SUBTENSOR_GATEWAY_HOST
    port = settings.SUBTENSOR_GATEWAY_PORT
    wallet = bt.Wallet(
        name=settings.BITTENSOR_WALLET_COLD,
        hotkey=settings.BITTENSOR_WALLET_HOT,
    )

    @web.middleware
    async def access_log(request: web.Request, handler):
        t0 = time.monotonic()
        try:
            resp = await handler(request)
            return resp
        finally:
            dt = (time.monotonic() - t0) * 1000
            logger.info(
                "[gateway] %s %s -> %s %.1fms",
                request.method,
                request.path,
                getattr(getattr(request, "response", None), "status", "?"),
                dt,
            )

    async def health(_req: web.Request):
        return web.json_response({"ok": True})

    async def block_current_handler(_req: web.Request):
        try:
            block = await _get_current_block_resilient()
            return web.json_response({"block": int(block)})
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[gateway] block/current failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"}, status=503
            )

    async def block_wait_handler(req: web.Request):
        payload = await req.json()
        timeout_s = payload.get("timeout_s")
        try:
            block = await _wait_for_next_block_polling(
                None if timeout_s is None else float(timeout_s)
            )
            return web.json_response({"block": int(block)})
        except asyncio.TimeoutError as exc:
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"}, status=504
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[gateway] block/wait failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"}, status=503
            )

    async def metagraph_snapshot_handler(req: web.Request):
        payload = await req.json()
        netuid = int(payload.get("netuid", os.getenv("BABELBIT_NETUID", "59")))
        lite = bool(payload.get("lite", False))
        meta = await STATE.metagraph(netuid=netuid, lite=lite)
        return web.json_response(_meta_snapshot(meta))

    async def registry_context_handler(req: web.Request):
        payload = await req.json()
        netuid = int(payload.get("netuid", os.getenv("BABELBIT_NETUID", "59")))
        lite = bool(payload.get("lite", False))
        st = await STATE.get_subtensor()
        meta = await STATE.metagraph(netuid=netuid, lite=lite)
        commits = await st.get_all_revealed_commitments(netuid)
        return web.json_response(
            {
                "metagraph": _meta_snapshot(meta),
                "commitments": _serialize_commitments(commits),
            }
        )

    async def set_weights_and_confirm_handler(req: web.Request):
        payload = await req.json()
        netuid = int(payload.get("netuid", os.getenv("BABELBIT_NETUID", "59")))
        uids = [int(u) for u in (payload.get("uids") or [])]
        weights = [float(w) for w in (payload.get("weights") or [])]
        wfi = bool(payload.get("wait_for_inclusion", False))
        retries = int(payload.get("retries", 20))
        delay_s = float(payload.get("delay_s", 2.0))
        wallet_hotkey = payload.get("wallet_hotkey") or wallet.hotkey.ss58_address

        if not uids or len(uids) != len(weights):
            return web.json_response(
                {"success": False, "error": "uids/weights mismatch or empty"},
                status=400,
            )

        last_details = {
            "wallet_hotkey": wallet_hotkey,
            "uid_count": len(uids),
            "nonzero_weights": sum(1 for w in weights if float(w) > 0.0),
            "top_weights": _top_nonzero_weights(uids, weights),
        }
        for attempt in range(retries):
            try:
                st = await STATE.get_subtensor()
                ref = await st.get_current_block()
                last_details.update({"ref_block": int(ref), "attempt": attempt + 1})
                commit_reveal_enabled = bool(
                    await _maybe_await(st.commit_reveal_enabled(netuid=netuid))
                )
                last_details["commit_reveal_enabled"] = commit_reveal_enabled
                result = await st.set_weights(
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids,
                    weights=weights,
                    wait_for_inclusion=wfi,
                )
                success, message = _set_weights_result(result)
                last_details.update(
                    {"set_weights_success": success, "set_weights_message": message}
                )
                if not success:
                    await asyncio.sleep(delay_s)
                    continue
                await st.wait_for_block()
                if commit_reveal_enabled:
                    commits = await _maybe_await(
                        st.get_timelocked_weight_commits(netuid=netuid)
                    )
                    hotkey_commits = [
                        commit for commit in commits or [] if commit[0] == wallet_hotkey
                    ]
                    newest_commit = max(
                        hotkey_commits,
                        key=lambda commit: int(commit[1]),
                        default=None,
                    )
                    last_details.update(
                        {
                            "timelocked_commit_count": len(commits or []),
                            "wallet_timelocked_commit_count": len(hotkey_commits),
                        }
                    )
                    if newest_commit is not None:
                        last_details.update(
                            {
                                "commit_block": int(newest_commit[1]),
                                "reveal_round": int(newest_commit[3]),
                            }
                        )
                        if int(newest_commit[1]) >= int(ref):
                            return web.json_response(
                                {
                                    "success": True,
                                    "details": {
                                        **last_details,
                                        "confirmation": "timelocked_commit",
                                    },
                                }
                            )
                    await asyncio.sleep(delay_s)
                    continue
                meta = await STATE.metagraph(netuid=netuid, lite=True)
                try:
                    idx = meta.hotkeys.index(wallet_hotkey)
                except ValueError:
                    last_details.update(
                        {
                            "wallet_hotkey_found": False,
                            "metagraph_block": int(getattr(meta, "block", 0) or 0),
                        }
                    )
                    await asyncio.sleep(delay_s)
                    continue
                lu = int(meta.last_update[idx])
                last_details.update(
                    {
                        "wallet_hotkey_found": True,
                        "last_update": lu,
                        "metagraph_block": int(getattr(meta, "block", 0) or 0),
                    }
                )
                if lu >= int(ref):
                    return web.json_response(
                        {
                            "success": True,
                            "details": {
                                "last_update": lu,
                                "ref_block": int(ref),
                                "attempt": attempt + 1,
                            },
                        }
                    )
            except Exception as e:
                logger.warning(
                    "[gateway] set_weights attempt %d/%d failed: %s: %s",
                    attempt + 1,
                    retries,
                    type(e).__name__,
                    e,
                )
                last_details.update(
                    {"exception_type": type(e).__name__, "exception": str(e)}
                )
                await STATE.reset_subtensor()
            await asyncio.sleep(delay_s)

        return web.json_response(
            {
                "success": False,
                "error": "confirmation failed",
                "details": last_details,
            },
            status=500,
        )

    app = web.Application(middlewares=[access_log])
    app.add_routes(
        [
            web.get("/healthz", health),
            web.post("/v1/block/current", block_current_handler),
            web.post("/v1/block/wait", block_wait_handler),
            web.post("/v1/metagraph/snapshot", metagraph_snapshot_handler),
            web.post("/v1/registry/context", registry_context_handler),
            web.post("/v1/weights/set_and_confirm", set_weights_and_confirm_handler),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    try:
        hn = socket.gethostname()
        ip = socket.gethostbyname(hn)
    except Exception:
        hn, ip = ("?", "?")
    logger.info(
        "Subtensor gateway listening on http://%s:%s hostname=%s ip=%s",
        host,
        port,
        hn,
        ip,
    )

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Gateway received shutdown signal")
    finally:
        await STATE.reset_subtensor()
        await runner.cleanup()
