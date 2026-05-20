from __future__ import annotations

import os

from babelbit.utils.hf_runtime import ensure_hf_transfer_available


def test_disables_hf_transfer_when_package_missing(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    assert ensure_hf_transfer_available() is False
    assert ensure_hf_transfer_available() is False
    assert os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "0"


def test_keeps_hf_transfer_enabled_when_package_present(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    assert ensure_hf_transfer_available() is True
    assert os.getenv("HF_HUB_ENABLE_HF_TRANSFER") == "1"
