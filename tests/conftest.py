import os

import pytest
from babelbit.cli import runner as runner_mod

# Disable solo challenge in tests by default to avoid exercising unmocked solo paths.
os.environ.setdefault("BB_ENABLE_SOLO_CHALLENGE", "0")
# Disable round2 challenge by default; tests enable it explicitly when needed.
os.environ.setdefault("BB_ENABLE_ARENA_CHALLENGE", "0")
# Prevent test runs from spinning up multiple scorer workers.
os.environ.setdefault("BB_SCORE_PARALLELISM", "1")
os.environ.setdefault("BB_SCORE_IO_PARALLELISM", "1")
os.environ.setdefault("BB_SCORE_TORCH_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EMBED_BATCH_SIZE", "4")


@pytest.fixture(autouse=True)
def _limit_runner_scoring_memory(monkeypatch):
    # Keep scorer work in-process by default so tests do not fork extra model workers.
    monkeypatch.setattr(runner_mod, "_should_use_scoring_process_pool", lambda: False)
