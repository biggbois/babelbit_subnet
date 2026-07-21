#!/usr/bin/env python3
"""Move legacy API challenge fixtures from benchmark_outputs to miner-test-data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from babelbit.benchmarks.api_challenge import migrate_legacy_api_challenge_fixtures
from babelbit.benchmarks.miner_test_data import workspace_root_from


def subnet_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate API challenge fixture cache into miner-test-data/api_challenges."
    )
    parser.add_argument("--challenge-uid", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace_root = workspace_root_from(subnet_root())
    reports = migrate_legacy_api_challenge_fixtures(
        workspace_root,
        challenge_uid=args.challenge_uid,
        dry_run=args.dry_run,
    )
    if not reports:
        print("No legacy fixture cache found to migrate.", file=sys.stderr)
        return 0

    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
