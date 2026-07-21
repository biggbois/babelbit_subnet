from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "benchmarks" / "benchmark_miner_sample.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_miner_sample", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

workspace_root_from = _MOD.workspace_root_from
subnet_root = _MOD.subnet_root
locale_sample_path = _MOD.locale_sample_path


def test_workspace_root_points_at_repo_root() -> None:
    root = workspace_root_from(subnet_root())
    assert (root / "babelbit_miner").is_dir()
    assert (root / "babelbit_subnet").is_dir()
    assert (root / "miner-test-data").is_dir()


def test_validator_challenge_uids_simulates_overlapping_sessions() -> None:
    from tests.benchmarks import benchmark_miner_sample as sample_mod

    uids = sample_mod._validator_challenge_uids("challenge-abc", 3)
    assert uids[0] == "challenge-abc"
    assert uids[1].startswith("solo-challenge-")
    assert uids[2].startswith("qualifying-challenge-")

    sample = locale_sample_path(workspace_root_from(subnet_root()), locale="fr")
    assert sample.name == "en-npr-001481.json"
