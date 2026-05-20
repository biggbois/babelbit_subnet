from logging import getLogger
from typing import Any, Dict, Optional

from babelbit.utils.async_clients import get_async_client
from babelbit.utils.utterance_auth import (
    authenticate_utterance_engine,
    get_auth_headers,
)

logger = getLogger(__name__)


class UtteranceEngineError(Exception):
    pass


async def retry_with_exponential_backoff(
    func,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs,
):
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    getattr(func, "__name__", "request"),
                    type(exc).__name__,
                    exc,
                    delay,
                )
                import asyncio

                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(
                    "All %d attempts failed for %s: %s: %s",
                    max_retries,
                    getattr(func, "__name__", "request"),
                    type(exc).__name__,
                    exc,
                )

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retry failed without exception")


async def _request_with_reauth(
    session,
    method: str,
    url: str,
    *,
    json_payload: Optional[dict] = None,
    allow_retry: bool = True,
) -> tuple[int, Dict[str, Any] | str]:
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
                "Utterance engine returned 401 while fetching challenge UID; refreshing auth and retrying once."
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


async def get_current_challenge_uid(utterance_engine_url: str) -> Optional[str]:
    async def _get_challenge() -> Optional[str]:
        session = await get_async_client()
        status, start_data = await _request_with_reauth(
            session,
            "POST",
            f"{utterance_engine_url}/source-audio/start",
        )
        if status != 200:
            raise UtteranceEngineError(f"Failed to get challenge ID: HTTP {status}")

        challenge_uid = (
            start_data.get("challenge_uid") if isinstance(start_data, dict) else None
        )
        logger.info("Current challenge ID: %s", challenge_uid)
        return challenge_uid

    try:
        return await retry_with_exponential_backoff(
            _get_challenge, max_retries=3, initial_delay=1.0
        )
    except Exception as exc:
        logger.error("Error getting challenge ID after retries: %s", exc)
        raise UtteranceEngineError(f"Failed to get challenge ID: {exc}")


__all__ = [
    "UtteranceEngineError",
    "get_current_challenge_uid",
    "retry_with_exponential_backoff",
]
