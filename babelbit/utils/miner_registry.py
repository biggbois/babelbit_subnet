from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

from babelbit.utils.bittensor_helpers import get_subtensor
from babelbit.utils.subtensor_gateway_client import SubtensorGatewayClient


@dataclass
class Miner:
    uid: int
    hotkey: str
    block: int
    model: Optional[str] = None
    revision: Optional[str] = None
    slug: Optional[str] = None
    chute_id: Optional[str] = None
    axon_ip: Optional[str] = None
    axon_port: Optional[int] = None


def _is_valid_axon(ip: Optional[str], port: Optional[int]) -> bool:
    if not ip or not port:
        return False
    ip_str = str(ip).strip()
    if not ip_str or ip_str in {"0.0.0.0", "0", "None"}:
        return False
    try:
        return int(port) > 0
    except Exception:
        return False


def _extract_axon(meta: Any, uid: int) -> tuple[Optional[str], Optional[int]]:
    axons = getattr(meta, "axons", []) or []
    if uid >= len(axons):
        return None, None

    axon = axons[uid]
    if isinstance(axon, dict):
        ip = axon.get("ip")
        port = axon.get("port")
    else:
        ip = getattr(axon, "ip", None)
        port = getattr(axon, "port", None)

    ip_str = str(ip).strip() if ip is not None else None
    port_int: Optional[int]
    try:
        port_int = int(port) if port is not None else None
    except Exception:
        port_int = None
    return ip_str, port_int


def _snapshot_to_meta(snapshot: dict[str, Any]) -> Any:
    return SimpleNamespace(
        hotkeys=snapshot.get("hotkeys", []),
        last_update=snapshot.get("last_update", []),
        axons=snapshot.get("axons", []),
        block=snapshot.get("block"),
    )


async def _load_registry_context(netuid: int, subtensor=None, logger: Optional[logging.Logger] = None) -> dict[str, Any]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if subtensor is None:
        try:
            gateway = SubtensorGatewayClient()
            return {
                "metagraph": await gateway.metagraph_snapshot(
                    netuid=netuid,
                    lite=False,
                )
            }
        except Exception as exc:
            logger.warning(
                "Gateway metagraph snapshot failed; falling back to subtensor: %s: %s",
                type(exc).__name__,
                exc,
            )
            st = await get_subtensor()
            meta = await st.metagraph(netuid)
            return {
                "metagraph": {
                    "hotkeys": getattr(meta, "hotkeys", []),
                    "axons": getattr(meta, "axons", []),
                    "block": getattr(meta, "block", 0),
                }
            }

    if hasattr(subtensor, "registry_context"):
        return await subtensor.registry_context(netuid=netuid, lite=False)

    if hasattr(subtensor, "metagraph_object"):
        meta = await subtensor.metagraph_object(netuid=netuid, lite=False)
        return {
            "metagraph": {
                "hotkeys": getattr(meta, "hotkeys", []),
                "last_update": getattr(meta, "last_update", []),
                "axons": getattr(meta, "axons", []),
                "block": getattr(meta, "block", 0),
            }
        }

    if hasattr(subtensor, "metagraph"):
        meta = await subtensor.metagraph(netuid)
        return {
            "metagraph": {
                "hotkeys": getattr(meta, "hotkeys", []),
                "last_update": getattr(meta, "last_update", []),
                "axons": getattr(meta, "axons", []),
                "block": getattr(meta, "block", 0),
            }
        }

    raise RuntimeError(f"Unsupported subtensor client type: {type(subtensor).__name__}")


async def get_miners_from_registry(netuid: int, subtensor=None) -> Dict[int, Miner]:
    """
    Resolve main-mode miner candidates from self-hosted axons.

    On-chain commitments are no longer part of the scoring protocol. They may
    still be present in registry snapshots for older miners, but they must not
    gate S2S eligibility or trigger stale Hugging Face/chute filtering.
    Main-mode runner uses this registry source.
    """
    logger = logging.getLogger(__name__)
    ctx = await _load_registry_context(netuid=netuid, subtensor=subtensor, logger=logger)
    snapshot = ctx.get("metagraph", {}) if isinstance(ctx, dict) else {}
    meta = _snapshot_to_meta(snapshot if isinstance(snapshot, dict) else {})

    hotkeys = list(getattr(meta, "hotkeys", []) or [])
    logger.info("Checking %d hotkeys for valid axons", len(hotkeys))

    candidates: Dict[int, Miner] = {}

    for uid, hotkey in enumerate(hotkeys):
        axon_ip, axon_port = _extract_axon(meta, uid)

        if _is_valid_axon(axon_ip, axon_port):
            candidates[uid] = Miner(
                uid=uid,
                hotkey=hotkey,
                block=0,
                model=None,
                revision=None,
                slug=None,
                chute_id=None,
                axon_ip=axon_ip,
                axon_port=axon_port,
            )
            continue

    if not candidates:
        logger.info("Found 0 eligible miners from axons")
        return {}

    logger.info("Found %d eligible miners from registry", len(candidates))
    return candidates
