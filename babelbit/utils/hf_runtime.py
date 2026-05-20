from __future__ import annotations

import importlib.util
import os
from logging import getLogger


logger = getLogger(__name__)


def ensure_hf_transfer_available() -> bool:
    enabled = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return False

    if importlib.util.find_spec("hf_transfer") is not None:
        return True

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    logger.warning(
        "HF_HUB_ENABLE_HF_TRANSFER is enabled but hf_transfer is not installed; "
        "disabling fast download for this process"
    )
    return False


__all__ = ["ensure_hf_transfer_available"]
