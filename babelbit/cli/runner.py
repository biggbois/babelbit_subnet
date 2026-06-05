import gc
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Dict, Tuple
from logging import INFO, getLogger
import os
import sys
import subprocess
from pathlib import Path
import json
import random
import asyncio
import time
import traceback

from babelbit.utils.s3_manager import S3Manager
from babelbit.utils.settings import get_settings

from babelbit.schemas.audio_prediction import (
    BBAudioChallengeResult,
    BBAudioMinerInitPayload,
    BBAudioMinerPredictPayload,
)
from babelbit.utils.predict_utterances import (
    get_current_challenge_uid,
)
from babelbit.utils.predict_audio import predict_source_audio_multi_miner
from babelbit.utils.utterance_auth import (
    init_utterance_auth,
    authenticate_utterance_engine,
    is_non_retryable_auth_error,
)
from babelbit.utils.async_clients import close_http_clients

from babelbit.utils.miner_registry import get_miners_from_registry, Miner
from babelbit.utils.managed_container_registry import (
    ManagedRoute,
    resolve_round2_routes,
)
from babelbit.utils.subtensor_gateway_client import (
    SubtensorGatewayClient,
    close_gateway_clients,
)
from babelbit.utils.audio_artifacts import (
    create_audio_challenge_run_data,
    create_audio_challenge_score_data,
    save_audio_artifact_bundle,
    save_audio_run_log,
)
from babelbit.utils.file_handling import (
    get_processed_miners_for_challenge,
    save_challenge_run_file,
    save_challenge_score_file,
)
from babelbit.utils.challenge_status import mark_challenge_processed
from babelbit.utils.validation_submission import ValidationSubmissionClient

logger = getLogger(__name__)


s3_manager: Optional[S3Manager] = None
settings = get_settings()
_DEFER_HTTP_CLIENT_CLOSE = False

_ARENA_READY_ROUTE_STATUSES = {"running", "idle"}


def _is_arena_route_ready(route: Optional[ManagedRoute]) -> bool:
    if route is None:
        return False
    status = str(route.status or "").strip().lower()
    return status in _ARENA_READY_ROUTE_STATUSES


async def _resolve_round2_routes_when_ready(
    *,
    netuid: int,
    subtensor,
    ready_timeout_sec: float,
    poll_sec: float,
) -> Tuple[List[Miner], Dict[str, ManagedRoute]]:
    deadline = time.monotonic() + max(0.0, float(ready_timeout_sec))
    poll_interval = max(1.0, float(poll_sec))
    while True:
        arena_miners, routes_by_hotkey = await resolve_round2_routes(
            netuid=netuid,
            subtensor=subtensor,
        )
        ready_miners = [
            miner
            for miner in arena_miners
            if _is_arena_route_ready(routes_by_hotkey.get(miner.hotkey))
        ]

        if arena_miners and len(ready_miners) == len(arena_miners):
            if len(arena_miners) > 0:
                logger.info(
                    "All discovered Round2 routes are live: ready=%d total=%d",
                    len(ready_miners),
                    len(arena_miners),
                )
            return arena_miners, routes_by_hotkey

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if arena_miners:
                logger.warning(
                    "Round2 route readiness timeout expired: ready=%d total=%d; "
                    "starting with live routes only",
                    len(ready_miners),
                    len(arena_miners),
                )
            ready_hotkeys = {miner.hotkey for miner in ready_miners}
            return (
                ready_miners,
                {
                    hotkey: route
                    for hotkey, route in routes_by_hotkey.items()
                    if hotkey in ready_hotkeys
                },
            )

        logger.info(
            "Waiting for Round2 routes to become live: ready=%d total=%d timeout_remaining=%.1fs",
            len(ready_miners),
            len(arena_miners),
            remaining,
        )
        await asyncio.sleep(min(poll_interval, remaining))


async def get_subtensor():
    return SubtensorGatewayClient()


async def reset_subtensor():
    await close_gateway_clients()


def _close_http_clients_if_allowed(*, caller: str) -> None:
    """Avoid tearing down shared clients between main/arena phases in runner_loop."""
    if _DEFER_HTTP_CLIENT_CLOSE:
        _stderr_boot(f"{caller} exit: close_http_clients deferred")
        return
    _stderr_boot(f"{caller} exit: close_http_clients")
    close_http_clients()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@lru_cache(maxsize=1)
def _get_runner_build_info() -> str:
    """Return git branch/commit information for boot logs when available."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        branch = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        state = "dirty" if dirty else "clean"
        return (
            f"branch={branch or 'unknown'} commit={commit or 'unknown'} state={state}"
        )
    except Exception as e:
        return f"branch=unknown commit=unknown state=unavailable({type(e).__name__})"


def _format_runner_startup_context(*, version: Optional[str] = None) -> str:
    parts = [_get_runner_build_info()]
    if version:
        parts.append(f"version={version}")
    return " ".join(parts)


def _stderr_boot(message: str) -> None:
    """Emit startup/debug messages even if logging is misconfigured."""
    try:
        print(f"[runner-boot] {message}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _enforce_runner_logging_level() -> None:
    """Keep runner loggers at INFO+ even if external libs mutate log levels."""
    try:
        for name in (
            "babelbit",
            "babelbit.cli.runner",
            "babelbit.utils.predict_utterances",
            "babelbit.utils.predict_engine",
            "babelbit.utils.managed_container_registry",
        ):
            getLogger(name).setLevel(INFO)
    except Exception:
        pass


def _coerce_timeout_seconds(value: object, default: float = 10.0) -> float:
    """Convert timeout config to float safely (handles mocked settings in tests)."""
    try:
        if isinstance(value, bool):
            return default
        parsed = float(value)  # type: ignore[arg-type]
        if parsed > 0:
            return parsed
    except Exception:
        pass
    return default


def _resolve_s2s_audio_timeouts(
    *, settings: object, challenge_type: str
) -> tuple[float, float]:
    base_timeout = _coerce_timeout_seconds(
        getattr(settings, "BB_MINER_TIMEOUT_SEC", None),
        default=10.0,
    )
    if challenge_type == "arena":
        miner_timeout = _coerce_timeout_seconds(
            getattr(settings, "BB_ARENA_MINER_TIMEOUT_SEC", None),
            default=base_timeout,
        )
    else:
        miner_timeout = base_timeout

    init_timeout = _coerce_timeout_seconds(
        getattr(settings, "BB_S2S_INIT_TIMEOUT_SEC", None),
        default=max(60.0, miner_timeout),
    )
    return miner_timeout, init_timeout


def _is_first_audio_utterance(payload: object) -> bool:
    utterance_id = str(getattr(payload, "utterance_id", "") or "").strip()
    if not utterance_id:
        return True
    return utterance_id == "0" or utterance_id.endswith(":0")


def _format_profile_seconds(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def _log_runner_profile(
    *,
    run_kind: str,
    challenge_uid: Optional[str],
    miner_count: int,
    miners_with_utterances: int,
    dialogues_scored: Optional[int],
    miner_serving_seconds: Optional[float],
    scoring_seconds: Optional[float],
    persistence_seconds: Optional[float],
) -> None:
    profile_message = (
        f"[RunnerProfile][{run_kind}] challenge_uid={challenge_uid or 'N/A'} "
        f"miners={miner_count} miners_with_utterances={miners_with_utterances} "
        f"dialogues_scored={dialogues_scored if dialogues_scored is not None else 'N/A'} "
        f"miner_serving_sec={_format_profile_seconds(miner_serving_seconds)} "
        f"scoring_sec={_format_profile_seconds(scoring_seconds)} "
        f"persistence_sec={_format_profile_seconds(persistence_seconds)}"
    )
    logger.info(profile_message)
    prefix = "runner" if run_kind == "main" else f"runner {run_kind}"
    _stderr_boot(f"{prefix} profile {profile_message}")


def _should_use_scoring_process_pool() -> bool:
    """Legacy test hook retained after dialogue scorer removal."""
    return False


async def _score_audio_miners_for_challenge(
    *,
    challenge_uid: Optional[str],
    challenge_type: str,
    miner_list: List[Miner],
    miner_results: Dict[str, BBAudioChallengeResult],
    logs_dir: Path,
    scores_dir: Path,
    submission_client: ValidationSubmissionClient,
    active_s3_manager: Optional[S3Manager],
    main_challenge_uid: Optional[str] = None,
) -> tuple[int, int, List[float]]:
    total_miners_processed = 0
    total_dialogues_processed = 0
    main_challenge_uid = main_challenge_uid or challenge_uid or ""
    artifact_challenge_type = challenge_type
    all_challenge_scores: List[float] = []

    for miner in miner_list:
        miner_result = miner_results.get(miner.hotkey)
        if miner_result is None or not miner_result.utterances:
            logger.warning(
                "Miner uid=%s hotkey=%s has no S2S utterance results to persist",
                getattr(miner, "uid", "?"),
                getattr(miner, "hotkey", "?"),
            )
            continue

        try:
            artifact_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path, _ = save_audio_run_log(miner_result, output_dir=str(logs_dir))
            tar_path = save_audio_artifact_bundle(
                miner_result, output_dir=str(logs_dir)
            )

            s3_log_path = None
            if active_s3_manager:
                s3_log_path = f"{settings.S3_LOG_DIR}/logs/{log_path.name}"
                active_s3_manager.upload_file(str(log_path), s3_log_path)

            if submission_client.is_ready:
                try:
                    await submission_client.submit_validation_file(
                        file_path=log_path,
                        file_type="dialogue_log",
                        kind="dialogue_logs",
                        challenge_id=challenge_uid or "",
                        main_challenge_uid=main_challenge_uid,
                        miner_uid=getattr(miner, "uid", None),
                        miner_hotkey=getattr(miner, "hotkey", None),
                        dialogue_uid=None,
                        s3_path=s3_log_path,
                        extra_data={
                            "challenge_type": artifact_challenge_type,
                            "protocol": "s2s_audio_v1",
                            "score_is_fallback": miner_result.score_is_fallback,
                            "score_method": miner_result.score_method,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Validation submission error for %s: %s", log_path, e
                    )

            s3_tar_path = None
            if active_s3_manager:
                s3_tar_path = f"submissions/{tar_path.name}"
                active_s3_manager.upload_file(str(tar_path), s3_tar_path)

            if submission_client.is_ready:
                try:
                    await submission_client.submit_validation_artifact(
                        file_path=tar_path,
                        kind="audio_bundle",
                        challenge_id=challenge_uid or "",
                        main_challenge_uid=main_challenge_uid,
                        miner_uid=getattr(miner, "uid", None),
                        miner_hotkey=getattr(miner, "hotkey", None),
                        extra_data={
                            "challenge_type": artifact_challenge_type,
                            "protocol": "s2s_audio_v1",
                            "score_is_fallback": miner_result.score_is_fallback,
                            "score_method": miner_result.score_method,
                            "s3_path": s3_tar_path if active_s3_manager else None,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Validation artifact submission error for %s: %s", tar_path, e
                    )

            challenge_run = create_audio_challenge_run_data(
                miner=miner,
                challenge_uid=challenge_uid or "",
                challenge_type=artifact_challenge_type,
                challenge_result=miner_result,
                log_file_path=str(log_path),
            )
            challenge_run_path = Path(
                save_challenge_run_file(
                    challenge_run,
                    output_dir=str(scores_dir),
                    timestamp=artifact_timestamp,
                )
            )
            s3_challenge_run_path = None
            if active_s3_manager:
                s3_challenge_run_path = f"submissions/{challenge_run_path.name}"
                active_s3_manager.upload_file(
                    str(challenge_run_path), s3_challenge_run_path
                )

            if submission_client.is_ready:
                try:
                    await submission_client.submit_validation_file(
                        file_path=challenge_run_path,
                        file_type="challenge_run",
                        kind="dialogue_scores",
                        challenge_id=challenge_uid or "",
                        main_challenge_uid=main_challenge_uid,
                        miner_uid=getattr(miner, "uid", None),
                        miner_hotkey=getattr(miner, "hotkey", None),
                        dialogue_uid=None,
                        s3_path=s3_challenge_run_path,
                        extra_data={
                            "challenge_type": artifact_challenge_type,
                            "protocol": "s2s_audio_v1",
                            "score_is_fallback": miner_result.score_is_fallback,
                            "score_method": miner_result.score_method,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Validation submission error for %s: %s",
                        challenge_run_path,
                        e,
                    )

            summary = create_audio_challenge_score_data(
                miner=miner,
                challenge_uid=challenge_uid or "",
                challenge_type=artifact_challenge_type,
                challenge_result=miner_result,
                log_file_path=str(log_path),
            )
            summary_path = Path(
                save_challenge_score_file(
                    summary,
                    output_dir=str(scores_dir),
                    timestamp=artifact_timestamp,
                )
            )
            s3_summary_path = None
            if active_s3_manager:
                s3_summary_path = f"submissions/{summary_path.name}"
                active_s3_manager.upload_file(str(summary_path), s3_summary_path)

            if submission_client.is_ready:
                try:
                    await submission_client.submit_validation_file(
                        file_path=summary_path,
                        file_type="challenge_scores",
                        kind="challenge_scores",
                        challenge_id=challenge_uid or "",
                        main_challenge_uid=main_challenge_uid,
                        miner_uid=getattr(miner, "uid", None),
                        miner_hotkey=getattr(miner, "hotkey", None),
                        dialogue_uid=None,
                        s3_path=s3_summary_path,
                        extra_data={
                            "challenge_type": artifact_challenge_type,
                            "protocol": "s2s_audio_v1",
                            "score_is_fallback": miner_result.score_is_fallback,
                            "score_method": miner_result.score_method,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Validation submission error for %s: %s", summary_path, e
                    )

            total_miners_processed += 1
            total_dialogues_processed += len(miner_result.utterances)
            all_challenge_scores.append(miner_result.score)
        except Exception as e:
            logger.warning(
                "Failed to persist S2S outputs for miner uid=%s hotkey=%s: %s",
                getattr(miner, "uid", "?"),
                getattr(miner, "hotkey", "?"),
                e,
            )
            continue

    return total_miners_processed, total_dialogues_processed, all_challenge_scores


def _build_axon_audio_callbacks(
    *, miner_timeout: float, init_timeout: float | None = None
):
    from babelbit.utils.predict_engine import call_miner_axon_audio_endpoint

    init_request_timeout = init_timeout if init_timeout is not None else miner_timeout

    async def init_callback(miner: Miner, payload: BBAudioMinerInitPayload) -> dict:
        request_timeout = (
            init_request_timeout
            if _is_first_audio_utterance(payload)
            else miner_timeout
        )
        response = await call_miner_axon_audio_endpoint(
            axon_ip=miner.axon_ip or "",
            axon_port=int(miner.axon_port or 0),
            payload=payload,
            miner_hotkey=miner.hotkey,
            timeout=request_timeout,
        )
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        return response

    async def predict_callback(
        miner: Miner, payload: BBAudioMinerPredictPayload
    ) -> dict:
        response = await call_miner_axon_audio_endpoint(
            axon_ip=miner.axon_ip or "",
            axon_port=int(miner.axon_port or 0),
            payload=payload,
            miner_hotkey=miner.hotkey,
            timeout=miner_timeout,
        )
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        return response

    return init_callback, predict_callback


def _build_round2_audio_callbacks(
    *,
    routes_by_hotkey: Dict[str, ManagedRoute],
    miner_timeout: float,
    init_timeout: float | None = None,
):
    from babelbit.utils.predict_engine import call_managed_route_audio_endpoint

    init_request_timeout = init_timeout if init_timeout is not None else miner_timeout

    async def init_callback(miner: Miner, payload: BBAudioMinerInitPayload) -> dict:
        route = routes_by_hotkey.get(getattr(miner, "hotkey", ""))
        if route is None:
            raise RuntimeError(
                f"Miner {getattr(miner, 'uid', '?')} has no managed route"
            )
        if getattr(route, "miner_uid", None) is None:
            route.miner_uid = getattr(miner, "uid", None)
        logger.info(
            "Round2 managed init dispatch uid=%s hotkey=%s provider=%s status=%s endpoint_url=%s",
            getattr(miner, "uid", "?"),
            (str(getattr(miner, "hotkey", ""))[:16] + "..."),
            getattr(route, "provider", ""),
            getattr(route, "status", ""),
            getattr(route, "endpoint_url", ""),
        )
        request_timeout = (
            init_request_timeout
            if _is_first_audio_utterance(payload)
            else miner_timeout
        )
        response = await call_managed_route_audio_endpoint(
            route=route,
            payload=payload,
            miner_hotkey=miner.hotkey,
            timeout=request_timeout,
        )
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        return response

    async def predict_callback(
        miner: Miner, payload: BBAudioMinerPredictPayload
    ) -> dict:
        route = routes_by_hotkey.get(getattr(miner, "hotkey", ""))
        if route is None:
            raise RuntimeError(
                f"Miner {getattr(miner, 'uid', '?')} has no managed route"
            )
        if getattr(route, "miner_uid", None) is None:
            route.miner_uid = getattr(miner, "uid", None)
        logger.info(
            "Round2 managed predict dispatch uid=%s hotkey=%s provider=%s status=%s endpoint_url=%s",
            getattr(miner, "uid", "?"),
            (str(getattr(miner, "hotkey", ""))[:16] + "..."),
            getattr(route, "provider", ""),
            getattr(route, "status", ""),
            getattr(route, "endpoint_url", ""),
        )
        response = await call_managed_route_audio_endpoint(
            route=route,
            payload=payload,
            miner_hotkey=miner.hotkey,
            timeout=miner_timeout,
        )
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        return response

    return init_callback, predict_callback


async def runner(
    utterance_engine_url: str | None = None,
    output_dir: Optional[str] = None,
    subtensor=None,
) -> None:
    _enforce_runner_logging_level()
    settings = get_settings()
    NETUID = settings.BABELBIT_NETUID
    MAX_MINERS = int(os.getenv("BB_MAX_MINERS_PER_RUN", "256"))
    utterance_engine_url = utterance_engine_url or os.getenv(
        "BB_UTTERANCE_ENGINE_URL", "http://localhost:8000"
    )
    enable_solo_challenge = os.getenv("BB_ENABLE_SOLO_CHALLENGE", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    startup_context = _format_runner_startup_context(
        version=getattr(settings, "BABELBIT_VERSION", None)
    )
    _stderr_boot(
        "runner entry "
        f"utterance_engine_url={utterance_engine_url} "
        f"netuid={NETUID} max_miners={MAX_MINERS} "
        f"solo_enabled={enable_solo_challenge} "
        f"{startup_context}",
    )
    logger.info("[RunnerBoot] %s", startup_context)

    # Determine directories:
    #   Raw logs:   ./logs (override with BB_OUTPUT_LOGS_DIR)
    #   Scores:     ./scores (override with BB_OUTPUT_SCORES_DIR or output_dir argument) produced after scoring
    #   output_dir argument retained for backward compatibility
    logs_dir = Path(os.getenv("BB_OUTPUT_LOGS_DIR", "logs"))
    scores_dir = Path(output_dir or os.getenv("BB_OUTPUT_SCORES_DIR", "scores"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "[runner] output dirs ready: logs_dir=%s scores_dir=%s (output_dir_arg=%s)",
        str(logs_dir),
        str(scores_dir),
        str(output_dir),
    )

    s3_enabled = os.getenv("BB_ENABLE_S3_UPLOADS", "0").lower() in {"1", "true", "yes"}
    global s3_manager
    if s3_enabled and s3_manager is None:
        try:
            s3_manager = S3Manager(
                bucket_name=settings.S3_BUCKET_NAME,
                access_key=settings.S3_ACCESS_KEY_ID,
                secret_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
                endpoint_url=settings.S3_ENDPOINT_URL or None,
                region=settings.S3_REGION,
                addressing_style=settings.S3_ADDRESSING_STYLE or "auto",
                signature_version=settings.S3_SIGNATURE_VERSION or "s3v4",
                use_ssl=settings.S3_USE_SSL,
                prefix="",  # Empty prefix so logs go directly to bucket/logs/
            )
            logger.info("S3 Manager initialized (uploads enabled)")
        except Exception as e:
            logger.warning(
                "S3 Manager initialization failed; disabling S3 uploads: %s", e
            )
            s3_manager = None
    logger.debug(
        "[runner] S3 uploads enabled=%s active=%s", s3_enabled, bool(s3_manager)
    )

    submission_client = ValidationSubmissionClient()
    logger.debug(
        "[runner] validation submissions ready=%s endpoint=%s",
        submission_client.is_ready,
        submission_client.submit_url if submission_client else "N/A",
    )
    selected_miner_count = 0
    miners_with_utterances = 0
    total_dialogues_processed: Optional[int] = None
    miner_serving_seconds: Optional[float] = None
    scoring_seconds: Optional[float] = None
    persistence_seconds: Optional[float] = None
    challenge_error: Optional[str] = None

    try:
        challenge_uid = await get_current_challenge_uid(utterance_engine_url)
        _stderr_boot(f"runner challenge_uid={challenge_uid}")
    except Exception as e:
        _stderr_boot(f"runner challenge_uid fetch failed: {type(e).__name__}: {e}")
        logger.warning(f"Could not get current challenge ID: {e}")
        return
    logger.debug(
        "[runner] fetched challenge_uid=%s from %s", challenge_uid, utterance_engine_url
    )

    # Prevents runner loop from running multiple times a challenge
    if challenge_uid:
        already_processed = get_processed_miners_for_challenge(
            str(scores_dir),
            challenge_uid,
            challenge_type="main",
        )
        if already_processed:
            _stderr_boot(
                f"runner skip already_processed challenge_uid={challenge_uid} "
                f"miners={len(already_processed)}",
            )
            logger.info(
                f"Challenge {challenge_uid} already has {len(already_processed)} scored miners. "
                f"Skipping entire run to avoid duplicate work."
            )
            return
        else:
            logger.info(
                f"Challenge {challenge_uid}: No existing scores found, proceeding with miner evaluation"
            )
            logger.debug(
                "[runner] already_processed=%s",
                list(already_processed) if already_processed else [],
            )

    try:
        _stderr_boot("runner fetching miners from registry")
        miners = await get_miners_from_registry(NETUID, subtensor=subtensor)
        if not isinstance(miners, dict):
            _stderr_boot(f"runner miner registry invalid type={type(miners).__name__}")
            logger.warning(
                "Miner registry returned unexpected type %s; skipping run",
                type(miners).__name__,
            )
            return
        _stderr_boot(f"runner miners fetched count={len(miners)}")
        logger.info(
            f"Found {len(miners)} eligible miners from registry: {list(miners.keys())}"
        )
        if not miners:
            _stderr_boot("runner no eligible miners")
            logger.warning("No eligible miners found on-chain.")
            return

        miner_list = list(miners.values())
        random.shuffle(miner_list)
        miner_list = miner_list[: min(MAX_MINERS, len(miner_list))]
        selected_miner_count = len(miner_list)
        logger.debug(
            "[runner] miners selected=%d (max=%d)",
            selected_miner_count,
            MAX_MINERS,
        )

        if not miner_list:
            logger.info("No miners to process after filtering")
            return

        miner_timeout, s2s_init_timeout = _resolve_s2s_audio_timeouts(
            settings=settings,
            challenge_type="main",
        )
        init_audio_callback, predict_audio_callback = _build_axon_audio_callbacks(
            miner_timeout=miner_timeout,
            init_timeout=s2s_init_timeout,
        )

        logger.info(
            "Starting shared source-audio session for %d miners", len(miner_list)
        )
        _stderr_boot(
            "runner predict begin "
            f"miners={len(miner_list)} timeout={miner_timeout:.2f} "
            f"init_timeout={s2s_init_timeout:.2f}",
        )

        audio_profile: Dict[str, float] = {}
        (
            resolved_challenge_uid,
            miner_audio_results,
        ) = await predict_source_audio_multi_miner(
            utterance_engine_url=utterance_engine_url,
            miners=miner_list,
            init_callback=init_audio_callback,
            predict_callback=predict_audio_callback,
            challenge_type="main",
            profile=audio_profile,
        )
        challenge_error = audio_profile.get("challenge_error")
        miner_serving_seconds = audio_profile.get("miner_serving_seconds")
        scoring_seconds = audio_profile.get("scoring_seconds")
        if resolved_challenge_uid:
            challenge_uid = resolved_challenge_uid
        miners_with_utterances = sum(
            1 for result in (miner_audio_results or {}).values() if result.utterances
        )
        _stderr_boot(
            f"runner predict end miners_with_utterances={miners_with_utterances}",
        )
        try:
            total_utterances = sum(
                len(result.utterances)
                for result in (miner_audio_results or {}).values()
            )
            logger.debug(
                "[runner] source-audio collected: miners_with_utterances=%d total_utterances=%d",
                miners_with_utterances,
                total_utterances,
            )
        except Exception:
            pass

        (
            total_miners_processed,
            total_dialogues_processed,
            all_challenge_scores,
        ) = (0, 0, [])
        persistence_started_at = time.perf_counter()
        try:
            (
                total_miners_processed,
                total_dialogues_processed,
                all_challenge_scores,
            ) = await _score_audio_miners_for_challenge(
                challenge_uid=challenge_uid,
                challenge_type="main",
                miner_list=miner_list,
                miner_results=miner_audio_results or {},
                logs_dir=logs_dir,
                scores_dir=scores_dir,
                submission_client=submission_client,
                active_s3_manager=s3_manager,
            )
        finally:
            persistence_seconds = time.perf_counter() - persistence_started_at

        if challenge_uid and total_miners_processed > 0 and not challenge_error:
            overall_mean = (
                sum(all_challenge_scores) / len(all_challenge_scores)
                if all_challenge_scores
                else None
            )
            mark_challenge_processed(
                challenge_uid=challenge_uid,
                miner_count=total_miners_processed,
                total_dialogues=total_dialogues_processed,
                mean_score=overall_mean,
                challenge_type="main",
                metadata={
                    "scores_dir": str(scores_dir),
                    "logs_dir": str(logs_dir),
                },
            )
            logger.debug(
                "[runner] challenge processed: uid=%s miners=%d dialogues=%d mean=%s",
                challenge_uid,
                total_miners_processed,
                total_dialogues_processed,
                (f"{overall_mean:.4f}" if overall_mean is not None else "N/A"),
            )
            mean_score_str = (
                f"{overall_mean:.4f}" if overall_mean is not None else "N/A"
            )
            logger.info(
                f"Challenge {challenge_uid} completed: {total_miners_processed} miners, "
                f"{total_dialogues_processed} dialogues, mean_score={mean_score_str}"
            )
            _stderr_boot(
                "runner challenge complete "
                f"challenge_uid={challenge_uid} miners={total_miners_processed} "
                f"dialogues={total_dialogues_processed} mean={mean_score_str}",
            )
        elif challenge_uid and total_miners_processed > 0 and challenge_error:
            logger.warning(
                "Challenge %s persisted partial S2S results but was not marked processed due to challenge error: %s",
                challenge_uid,
                challenge_error,
            )
            _stderr_boot(
                "runner challenge partial "
                f"challenge_uid={challenge_uid} miners={total_miners_processed} "
                f"dialogues={total_dialogues_processed} error={challenge_error}",
            )

        if enable_solo_challenge:
            solo_timeout, solo_init_timeout = _resolve_s2s_audio_timeouts(
                settings=settings,
                challenge_type="solo",
            )
            solo_init_callback, solo_predict_callback = _build_axon_audio_callbacks(
                miner_timeout=solo_timeout,
                init_timeout=solo_init_timeout,
            )
            solo_profile: Dict[str, float] = {}
            _stderr_boot(
                "runner solo predict begin "
                f"miners={len(miner_list)} timeout={solo_timeout:.2f} "
                f"init_timeout={solo_init_timeout:.2f}",
            )
            try:
                (
                    solo_challenge_uid,
                    solo_audio_results,
                ) = await predict_source_audio_multi_miner(
                    utterance_engine_url=utterance_engine_url,
                    miners=miner_list,
                    init_callback=solo_init_callback,
                    predict_callback=solo_predict_callback,
                    challenge_type="solo",
                    profile=solo_profile,
                )
                solo_error = solo_profile.get("challenge_error")
                solo_persistence_started_at = time.perf_counter()
                try:
                    (
                        solo_total_miners,
                        solo_total_dialogues,
                        solo_scores,
                    ) = await _score_audio_miners_for_challenge(
                        challenge_uid=solo_challenge_uid,
                        challenge_type="solo",
                        miner_list=miner_list,
                        miner_results=solo_audio_results or {},
                        logs_dir=logs_dir,
                        scores_dir=scores_dir,
                        submission_client=submission_client,
                        active_s3_manager=s3_manager,
                        main_challenge_uid=challenge_uid,
                    )
                finally:
                    solo_persistence_seconds = (
                        time.perf_counter() - solo_persistence_started_at
                    )

                if solo_challenge_uid and solo_total_miners > 0 and not solo_error:
                    solo_mean = (
                        sum(solo_scores) / len(solo_scores) if solo_scores else None
                    )
                    mark_challenge_processed(
                        challenge_uid=solo_challenge_uid,
                        miner_count=solo_total_miners,
                        total_dialogues=solo_total_dialogues,
                        mean_score=solo_mean,
                        challenge_type="solo",
                        metadata={
                            "scores_dir": str(scores_dir),
                            "logs_dir": str(logs_dir),
                            "paired_challenge_uid": challenge_uid,
                        },
                    )
                    solo_mean_str = (
                        f"{solo_mean:.4f}" if solo_mean is not None else "N/A"
                    )
                    logger.info(
                        "Solo challenge %s completed: %d miners, %d dialogues, mean_score=%s",
                        solo_challenge_uid,
                        solo_total_miners,
                        solo_total_dialogues,
                        solo_mean_str,
                    )
                    _stderr_boot(
                        "runner solo challenge complete "
                        f"challenge_uid={solo_challenge_uid} miners={solo_total_miners} "
                        f"dialogues={solo_total_dialogues} mean={solo_mean_str}",
                    )
                elif solo_challenge_uid and solo_total_miners > 0 and solo_error:
                    logger.warning(
                        "Solo challenge %s persisted partial S2S results but was not marked processed due to challenge error: %s",
                        solo_challenge_uid,
                        solo_error,
                    )
                _log_runner_profile(
                    run_kind="solo",
                    challenge_uid=solo_challenge_uid,
                    miner_count=len(miner_list),
                    miners_with_utterances=sum(
                        1
                        for result in (solo_audio_results or {}).values()
                        if result.utterances
                    ),
                    dialogues_scored=solo_total_dialogues,
                    miner_serving_seconds=solo_profile.get("miner_serving_seconds"),
                    scoring_seconds=solo_profile.get("scoring_seconds"),
                    persistence_seconds=solo_persistence_seconds,
                )
            except Exception as e:
                logger.warning("[Solo Challenge] Failed to run solo challenge: %s", e)
                _stderr_boot(f"runner solo failed: {type(e).__name__}: {e}")
        else:
            logger.debug(
                "[runner] Solo challenge phase disabled via BB_ENABLE_SOLO_CHALLENGE"
            )
            _stderr_boot("runner solo challenge disabled")

    except Exception as e:
        _stderr_boot(f"runner failed: {type(e).__name__}: {e}")
        try:
            traceback.print_exc()
        except Exception:
            pass
        logger.error(f"Runner failed: {type(e).__name__}: {e}", exc_info=True)
    finally:
        _log_runner_profile(
            run_kind="main",
            challenge_uid=challenge_uid if "challenge_uid" in locals() else None,
            miner_count=selected_miner_count,
            miners_with_utterances=miners_with_utterances,
            dialogues_scored=total_dialogues_processed,
            miner_serving_seconds=miner_serving_seconds,
            scoring_seconds=scoring_seconds,
            persistence_seconds=persistence_seconds,
        )
        _close_http_clients_if_allowed(caller="runner")


async def runner_round2(
    utterance_engine_url: str | None = None,
    output_dir: Optional[str] = None,
    subtensor=None,
) -> None:
    _enforce_runner_logging_level()
    settings = get_settings()
    NETUID = settings.BABELBIT_NETUID
    MAX_MINERS = int(os.getenv("BB_MAX_MINERS_PER_RUN", "256"))
    utterance_engine_url = utterance_engine_url or os.getenv(
        "BB_UTTERANCE_ENGINE_URL", "http://localhost:8000"
    )
    _stderr_boot(
        "runner arena entry "
        f"utterance_engine_url={utterance_engine_url} "
        f"netuid={NETUID} max_miners={MAX_MINERS}",
    )

    logs_dir = Path(os.getenv("BB_OUTPUT_LOGS_DIR", "logs"))
    scores_dir = Path(output_dir or os.getenv("BB_OUTPUT_SCORES_DIR", "scores"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    s3_enabled = os.getenv("BB_ENABLE_S3_UPLOADS", "0").lower() in {"1", "true", "yes"}
    global s3_manager
    if s3_enabled and s3_manager is None:
        try:
            s3_manager = S3Manager(
                bucket_name=settings.S3_BUCKET_NAME,
                access_key=settings.S3_ACCESS_KEY_ID,
                secret_key=settings.S3_SECRET_ACCESS_KEY.get_secret_value(),
                endpoint_url=settings.S3_ENDPOINT_URL or None,
                region=settings.S3_REGION,
                addressing_style=settings.S3_ADDRESSING_STYLE or "auto",
                signature_version=settings.S3_SIGNATURE_VERSION or "s3v4",
                use_ssl=settings.S3_USE_SSL,
                prefix="",
            )
            logger.info("S3 Manager initialized (uploads enabled)")
        except Exception as e:
            logger.warning(
                "S3 Manager initialization failed; disabling S3 uploads: %s", e
            )
            s3_manager = None

    submission_client = ValidationSubmissionClient()
    selected_miner_count = 0
    miners_with_utterances = 0
    total_dialogues_processed: Optional[int] = None
    miner_serving_seconds: Optional[float] = None
    scoring_seconds: Optional[float] = None
    persistence_seconds: Optional[float] = None
    challenge_error: Optional[str] = None

    try:
        challenge_uid = await get_current_challenge_uid(utterance_engine_url)
        _stderr_boot(f"runner arena challenge_uid={challenge_uid}")
    except Exception as e:
        _stderr_boot(
            f"runner arena challenge_uid fetch failed: {type(e).__name__}: {e}"
        )
        logger.warning("Could not get arena challenge ID: %s", e)
        return

    if challenge_uid:
        already_processed = get_processed_miners_for_challenge(
            str(scores_dir),
            challenge_uid,
            challenge_type="arena",
        )
        if already_processed:
            _stderr_boot(
                f"runner arena skip already_processed challenge_uid={challenge_uid} "
                f"miners={len(already_processed)}",
            )
            logger.info(
                "Arena challenge %s already has %d scored miners; skipping.",
                challenge_uid,
                len(already_processed),
            )
            return

    try:
        _stderr_boot("runner arena resolving managed routes")
        arena_miners, routes_by_hotkey = await _resolve_round2_routes_when_ready(
            netuid=NETUID,
            subtensor=subtensor,
            ready_timeout_sec=float(
                getattr(settings, "BB_ARENA_ROUTE_READY_TIMEOUT_SEC", 300)
            ),
            poll_sec=float(getattr(settings, "BB_ARENA_ROUTE_READY_POLL_SEC", 10)),
        )
        if not arena_miners:
            _stderr_boot("runner arena no live managed miners")
            logger.warning("No live managed miners found for arena.")
            return

        random.shuffle(arena_miners)
        arena_miners = arena_miners[: min(MAX_MINERS, len(arena_miners))]
        selected_miner_count = len(arena_miners)
        routes_by_hotkey = {
            m.hotkey: routes_by_hotkey[m.hotkey]
            for m in arena_miners
            if m.hotkey in routes_by_hotkey
        }

        arena_timeout, s2s_init_timeout = _resolve_s2s_audio_timeouts(
            settings=settings,
            challenge_type="arena",
        )
        init_audio_callback, predict_audio_callback = _build_round2_audio_callbacks(
            routes_by_hotkey=routes_by_hotkey,
            miner_timeout=arena_timeout,
            init_timeout=s2s_init_timeout,
        )
        _stderr_boot(
            "runner arena predict begin "
            f"miners={len(arena_miners)} timeout={arena_timeout:.2f} "
            f"init_timeout={s2s_init_timeout:.2f}",
        )

        audio_profile: Dict[str, float] = {}
        (
            resolved_challenge_uid,
            miner_audio_results,
        ) = await predict_source_audio_multi_miner(
            utterance_engine_url=utterance_engine_url,
            miners=arena_miners,
            init_callback=init_audio_callback,
            predict_callback=predict_audio_callback,
            challenge_type="arena",
            profile=audio_profile,
        )
        challenge_error = audio_profile.get("challenge_error")
        miner_serving_seconds = audio_profile.get("miner_serving_seconds")
        scoring_seconds = audio_profile.get("scoring_seconds")
        if resolved_challenge_uid:
            challenge_uid = resolved_challenge_uid
        miners_with_utterances = sum(
            1 for result in (miner_audio_results or {}).values() if result.utterances
        )
        _stderr_boot(
            f"runner arena predict end miners_with_utterances={miners_with_utterances}",
        )
        try:
            total_utterances = sum(
                len(result.utterances)
                for result in (miner_audio_results or {}).values()
            )
            logger.debug(
                "[runner arena] source-audio collected: miners_with_utterances=%d total_utterances=%d",
                miners_with_utterances,
                total_utterances,
            )
        except Exception:
            pass

        (
            total_miners_processed,
            total_dialogues_processed,
            all_challenge_scores,
        ) = (0, 0, [])
        persistence_started_at = time.perf_counter()
        try:
            (
                total_miners_processed,
                total_dialogues_processed,
                all_challenge_scores,
            ) = await _score_audio_miners_for_challenge(
                challenge_uid=challenge_uid,
                challenge_type="arena",
                miner_list=arena_miners,
                miner_results=miner_audio_results or {},
                logs_dir=logs_dir,
                scores_dir=scores_dir,
                submission_client=submission_client,
                active_s3_manager=s3_manager,
            )
        finally:
            persistence_seconds = time.perf_counter() - persistence_started_at

        if challenge_uid and total_miners_processed > 0 and not challenge_error:
            overall_mean = (
                sum(all_challenge_scores) / len(all_challenge_scores)
                if all_challenge_scores
                else None
            )
            mark_challenge_processed(
                challenge_uid=challenge_uid,
                challenge_type="arena",
                miner_count=total_miners_processed,
                total_dialogues=total_dialogues_processed,
                mean_score=overall_mean,
                metadata={
                    "scores_dir": str(scores_dir),
                    "logs_dir": str(logs_dir),
                    "route_source": "list_arena_miners",
                },
            )
            mean_score_str = (
                f"{overall_mean:.4f}" if overall_mean is not None else "N/A"
            )
            _stderr_boot(
                "runner arena challenge complete "
                f"challenge_uid={challenge_uid} miners={total_miners_processed} "
                f"dialogues={total_dialogues_processed} mean={mean_score_str}",
            )
            logger.info(
                "Arena challenge %s completed: %d miners, %d dialogues, mean_score=%s",
                challenge_uid,
                total_miners_processed,
                total_dialogues_processed,
                mean_score_str,
            )
        elif challenge_uid and total_miners_processed > 0 and challenge_error:
            logger.warning(
                "Arena challenge %s persisted partial S2S results but was not marked processed due to challenge error: %s",
                challenge_uid,
                challenge_error,
            )
            _stderr_boot(
                "runner arena challenge partial "
                f"challenge_uid={challenge_uid} miners={total_miners_processed} "
                f"dialogues={total_dialogues_processed} error={challenge_error}",
            )
    except Exception as e:
        _stderr_boot(f"runner arena failed: {type(e).__name__}: {e}")
        logger.error("Arena runner failed: %s: %s", type(e).__name__, e, exc_info=True)
    finally:
        _log_runner_profile(
            run_kind="arena",
            challenge_uid=challenge_uid if "challenge_uid" in locals() else None,
            miner_count=selected_miner_count,
            miners_with_utterances=miners_with_utterances,
            dialogues_scored=total_dialogues_processed,
            miner_serving_seconds=miner_serving_seconds,
            scoring_seconds=scoring_seconds,
            persistence_seconds=persistence_seconds,
        )
        _close_http_clients_if_allowed(caller="runner arena")


async def runner_loop():
    """Runs `runner()` every N blocks (default: 2160)."""
    global _DEFER_HTTP_CLIENT_CLOSE
    _enforce_runner_logging_level()
    settings = get_settings()
    TEMPO = int(os.getenv("BABELBIT_RUNNER_TEMPO", "2160"))
    MAX_SUBTENSOR_RETRIES = int(os.getenv("BABELBIT_MAX_SUBTENSOR_RETRIES", "5"))
    run_on_startup = _env_flag(
        "BB_RUNNER_ON_STARTUP",
        default=getattr(settings, "BB_RUNNER_ON_STARTUP", False),
    )
    arena_enabled = _env_flag(
        "BB_ENABLE_ARENA_CHALLENGE",
        default=getattr(settings, "BB_ENABLE_ARENA_CHALLENGE", False),
    )
    arena_run_on_startup = _env_flag(
        "BB_ARENA_RUN_ON_STARTUP",
        default=getattr(settings, "BB_ARENA_RUN_ON_STARTUP", False),
    )
    try:
        arena_cadence_blocks = int(
            os.getenv(
                "BB_ARENA_CADENCE_BLOCKS",
                str(getattr(settings, "BB_ARENA_CADENCE_BLOCKS", 300)),
            ),
        )
    except Exception:
        arena_cadence_blocks = 300
    if arena_cadence_blocks <= 0:
        arena_cadence_blocks = 1

    st = None
    last_block = -1
    last_arena_block = -1
    last_successful_run = 0
    consecutive_failures = 0
    run_count = 0
    last_wait_for_block_error: tuple[str, str] | None = None
    last_wait_for_block_error_count = 0
    suppress_wait_for_block_reconnect_logs = False

    def _reset_wait_for_block_error_state() -> None:
        nonlocal last_wait_for_block_error
        nonlocal last_wait_for_block_error_count
        nonlocal suppress_wait_for_block_reconnect_logs
        last_wait_for_block_error = None
        last_wait_for_block_error_count = 0
        suppress_wait_for_block_reconnect_logs = False

    # Initialize utterance engine authentication on startup
    utterance_engine_url = os.getenv(
        "BB_UTTERANCE_ENGINE_URL", "https://api.babelbit.ai"
    )
    wallet_name = os.getenv("BITTENSOR_WALLET_COLD", "default")
    hotkey_name = os.getenv("BITTENSOR_WALLET_HOT", "default")

    init_utterance_auth(utterance_engine_url, wallet_name, hotkey_name)
    startup_context = _format_runner_startup_context(
        version=getattr(settings, "BABELBIT_VERSION", None)
    )
    _stderr_boot(
        "runner_loop start "
        f"BB_RUNNER_ON_STARTUP={os.getenv('BB_RUNNER_ON_STARTUP')} "
        f"resolved={run_on_startup} "
        f"BB_ENABLE_ARENA_CHALLENGE={os.getenv('BB_ENABLE_ARENA_CHALLENGE')} "
        f"arena_enabled={arena_enabled} "
        f"BB_ARENA_CADENCE_BLOCKS={arena_cadence_blocks} "
        f"BB_ARENA_RUN_ON_STARTUP={arena_run_on_startup} "
        f"BB_UTTERANCE_ENGINE_URL={utterance_engine_url} "
        f"BABELBIT_RUNNER_TEMPO={TEMPO} "
        f"{startup_context}",
    )
    logger.info("[RunnerLoop] %s", startup_context)
    logger.info(
        "[RunnerLoop] Startup run enabled=%s (BB_RUNNER_ON_STARTUP=%s); arena enabled=%s cadence_blocks=%s startup=%s",
        run_on_startup,
        os.getenv("BB_RUNNER_ON_STARTUP"),
        arena_enabled,
        arena_cadence_blocks,
        arena_run_on_startup,
    )

    startup_auth_attempt = 0
    while True:
        startup_auth_attempt += 1
        try:
            logger.info(
                "[RunnerLoop] Authenticating with utterance engine on startup..."
            )
            _stderr_boot("auth startup begin")
            await authenticate_utterance_engine()
            logger.info("[RunnerLoop] Successfully authenticated with utterance engine")
            _stderr_boot("auth startup success")
            break
        except Exception as e:
            permanent_failure = is_non_retryable_auth_error(e)
            retry_delay = 300 if permanent_failure else 60
            _stderr_boot(f"auth startup failed: {type(e).__name__}: {e}")
            if permanent_failure:
                logger.error(
                    "[RunnerLoop] Startup authentication rejected by utterance engine: %s. Retrying in %ss without exiting.",
                    e,
                    retry_delay,
                )
            else:
                logger.warning(
                    "[RunnerLoop] Failed to authenticate with utterance engine on startup (attempt %s): %s. Retrying in %ss.",
                    startup_auth_attempt,
                    e,
                    retry_delay,
                )
            await asyncio.sleep(retry_delay)

    try:
        _DEFER_HTTP_CLIENT_CLOSE = True
        while True:
            try:
                if st is None:
                    connect_log = (
                        logger.debug
                        if suppress_wait_for_block_reconnect_logs
                        else logger.info
                    )
                    connect_log(
                        "[RunnerLoop] Attempting to connect to subtensor gateway "
                        "(attempt %s/%s)...",
                        consecutive_failures + 1,
                        MAX_SUBTENSOR_RETRIES,
                    )
                    if not suppress_wait_for_block_reconnect_logs:
                        _stderr_boot(
                            f"subtensor connect attempt={consecutive_failures + 1}/{MAX_SUBTENSOR_RETRIES}",
                        )
                    try:
                        await reset_subtensor()  # Clear any stale cached connection
                        st = await asyncio.wait_for(get_subtensor(), timeout=60)
                        connect_log(
                            "[RunnerLoop] Successfully created subtensor connection"
                        )
                        if not suppress_wait_for_block_reconnect_logs:
                            _stderr_boot("subtensor connected")

                        # Test the connection by fetching a block
                        test_block = await asyncio.wait_for(
                            st.get_current_block(), timeout=30
                        )
                        connect_log(
                            f"[RunnerLoop] Connection verified at block {test_block}"
                        )

                    except asyncio.TimeoutError as te:
                        st = None  # Clear invalid connection
                        await reset_subtensor()  # Also clear the global cache
                        raise TimeoutError(f"Subtensor initialization timed out: {te}")
                    except Exception as e:
                        st = None  # Clear invalid connection
                        await reset_subtensor()  # Also clear the global cache
                        logger.error(
                            f"[RunnerLoop] Subtensor connection failed: {type(e).__name__}: {e}",
                            exc_info=True,
                        )
                        raise

                # Try to get current block for tempo-based scheduling
                should_run_main = False
                should_run_arena = False
                block = None
                use_time_fallback = False

                try:
                    block = await asyncio.wait_for(st.get_current_block(), timeout=30)
                    logger.debug(f"[RunnerLoop] Current block: {block}")

                    # Refresh authentication 100 blocks before each run (or less if TEMPO < 100)
                    auth_refresh_offset = TEMPO - min(100, max(1, TEMPO - 1))
                    if block % TEMPO == auth_refresh_offset:
                        try:
                            logger.info(
                                f"[RunnerLoop] Refreshing authentication at block {block} ({TEMPO - auth_refresh_offset} blocks before next run)"
                            )
                            await authenticate_utterance_engine()
                            logger.info(
                                "[RunnerLoop] Authentication refresh successful"
                            )
                        except Exception as auth_e:
                            logger.error(
                                f"[RunnerLoop] Authentication refresh failed: {auth_e}"
                            )
                            # Don't stop the loop, but this will cause issues for the next runner() call

                    # Main challenge trigger
                    if (run_on_startup and last_successful_run == 0) or (
                        block > last_block and block % TEMPO == 0
                    ):
                        should_run_main = True
                        logger.info(f"[RunnerLoop] Triggering runner at block {block}")

                    # Arena challenge trigger on separate cadence
                    if arena_enabled:
                        if (arena_run_on_startup and last_arena_block < 0) or (
                            block > last_arena_block
                            and block % arena_cadence_blocks == 0
                        ):
                            should_run_arena = True
                            logger.info(
                                "[RunnerLoop] Triggering arena runner at block %s (cadence=%s)",
                                block,
                                arena_cadence_blocks,
                            )

                    if not should_run_main and not should_run_arena:
                        # Wait for next block with timeout
                        try:
                            await asyncio.wait_for(st.wait_for_block(), timeout=60)
                            if last_wait_for_block_error_count:
                                _reset_wait_for_block_error_state()
                        except asyncio.TimeoutError:
                            # Don't reset on timeout - just log and retry
                            if last_wait_for_block_error_count:
                                _reset_wait_for_block_error_state()
                            logger.debug(
                                "[RunnerLoop] wait_for_block timeout (60s) — retrying"
                            )
                            await asyncio.sleep(5)
                        except Exception as e:
                            error_signature = (type(e).__name__, str(e))
                            if error_signature == last_wait_for_block_error:
                                last_wait_for_block_error_count += 1
                            else:
                                last_wait_for_block_error = error_signature
                                last_wait_for_block_error_count = 1

                            if last_wait_for_block_error_count == 1:
                                logger.warning(
                                    "[RunnerLoop] wait_for_block error: %s",
                                    e,
                                )
                            elif last_wait_for_block_error_count == 2:
                                logger.warning(
                                    "[RunnerLoop] wait_for_block error repeating: %s (suppressing duplicate warnings and reconnect info)",
                                    e,
                                )
                            else:
                                logger.debug(
                                    "[RunnerLoop] wait_for_block error repeated %s times: %s",
                                    last_wait_for_block_error_count,
                                    e,
                                )

                            suppress_wait_for_block_reconnect_logs = (
                                last_wait_for_block_error_count >= 2
                            )
                            st = None
                            await reset_subtensor()
                        continue

                except Exception as e:
                    if last_wait_for_block_error_count:
                        _reset_wait_for_block_error_state()
                    # Block fetch failed - fall back to time-based scheduling
                    logger.warning(
                        f"[RunnerLoop] Block fetch failed: {type(e).__name__}: {e}"
                    )
                    st = None  # Force reconnection on next iteration
                    await reset_subtensor()  # Clear the global cached connection

                    if run_on_startup and last_successful_run == 0:
                        should_run_main = True
                        use_time_fallback = True
                        logger.warning(
                            "[RunnerLoop] Startup run is enabled and block fetch failed; "
                            "running validation via fallback path."
                        )
                        block = None
                    else:
                        time_elapsed = time.time() - last_successful_run
                        expected_interval = (
                            TEMPO * 12
                        )  # TEMPO blocks * ~12 seconds per block

                        if (
                            last_successful_run > 0
                            and time_elapsed >= expected_interval
                        ):
                            should_run_main = True
                            use_time_fallback = True
                            logger.warning(
                                f"[RunnerLoop] Blockchain unreachable. Using time-based fallback: "
                                f"elapsed={time_elapsed:.0f}s, expected={expected_interval:.0f}s"
                            )
                        else:
                            # Not enough time has passed, or first run - skip and let retry logic handle it
                            if last_successful_run == 0:
                                logger.info(
                                    "[RunnerLoop] First run - will retry connection"
                                )
                            else:
                                logger.info(
                                    f"[RunnerLoop] Only {time_elapsed:.0f}s elapsed (need {expected_interval:.0f}s), will retry"
                                )
                            raise  # Re-raise to trigger retry logic

                if should_run_main:
                    if last_wait_for_block_error_count:
                        _reset_wait_for_block_error_state()
                    if use_time_fallback:
                        logger.info(
                            "[RunnerLoop] Running validation via time-based fallback (blockchain unreachable)"
                        )
                    _stderr_boot(
                        "trigger runner call "
                        f"use_time_fallback={use_time_fallback} "
                        f"block={block}",
                    )

                    await runner(subtensor=st if st is not None else None)
                    _stderr_boot("runner call completed")

                    if block is not None:
                        last_block = block
                    last_successful_run = time.time()
                    consecutive_failures = 0  # Reset after successful validation cycle
                    run_count += 1
                    logger.info(f"[RunnerLoop] Completed runner cycle #{run_count}")

                    if run_count >= 10:
                        logger.info(
                            "[RunnerLoop] Reached 10 successful runs, resetting subtensor connection to free resources."
                        )
                        st = None
                        await reset_subtensor()
                        run_count = 0
                        gc.collect()

                if should_run_arena and block is not None:
                    if last_wait_for_block_error_count:
                        _reset_wait_for_block_error_state()
                    _stderr_boot(
                        "trigger arena runner call "
                        f"block={block} cadence_blocks={arena_cadence_blocks}",
                    )
                    await runner_round2(subtensor=st if st is not None else None)
                    _stderr_boot("arena runner call completed")
                    last_arena_block = block

            except asyncio.CancelledError:
                _stderr_boot(
                    "runner_loop received CancelledError; exiting loop",
                )
                break
            except Exception as e:
                _stderr_boot(
                    f"runner_loop iteration error: {type(e).__name__}: {e}",
                )
                consecutive_failures += 1
                logger.warning(
                    f"[RunnerLoop] Error (attempt {consecutive_failures}/{MAX_SUBTENSOR_RETRIES}): {type(e).__name__}: {e}"
                )

                if consecutive_failures >= MAX_SUBTENSOR_RETRIES:
                    logger.error(
                        f"[RunnerLoop] Max retries ({MAX_SUBTENSOR_RETRIES}) exceeded. "
                        f"gateway={settings.SUBTENSOR_GATEWAY_URL}"
                    )
                    logger.error(
                        "[RunnerLoop] Unable to connect to subtensor gateway. "
                        "Sleeping for 5 minutes before retry cycle..."
                    )
                    consecutive_failures = 0  # Reset counter
                    st = None
                    await asyncio.sleep(300)  # Sleep 5 minutes before trying again
                else:
                    logger.info(f"[RunnerLoop] Retrying in 120 seconds...")
                    st = None
                    await asyncio.sleep(120)
    finally:
        # Ensure cleanup on exit
        _DEFER_HTTP_CLIENT_CLOSE = False
        _stderr_boot("runner_loop shutdown: closing HTTP clients")
        logger.info("[RunnerLoop] Shutting down, cleaning up resources...")
        await reset_subtensor()
        close_http_clients()
