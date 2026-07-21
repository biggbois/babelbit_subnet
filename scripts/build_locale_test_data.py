#!/usr/bin/env python3
"""Build locale-specific miner-test-data JSON files from original EN challenges."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

from babelbit.benchmarks.miner_test_data import (
    build_locale_challenge_doc,
    build_utterance_entry,
    iter_en_challenge_utterances,
    miner_test_data_root,
    workspace_root_from,
)

LOCALE_LABELS = {
    "fr": "French",
    "de": "German",
}


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _deepinfra_headers() -> dict[str, str]:
    token = os.environ.get("DEEPINFRA_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DEEPINFRA_TOKEN is not set")
    return {"Authorization": f"Bearer {token}"}


def _translate(reference: str, *, locale: str) -> str:
    language = LOCALE_LABELS[locale]
    response = requests.post(
        "https://api.deepinfra.com/v1/openai/chat/completions",
        headers=_deepinfra_headers(),
        json={
            "model": os.environ.get("DEEPINFRA_LLM_MODEL", "Qwen/Qwen3-14B"),
            "temperature": 0,
            "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "/no_think\n"
                        f"Translate the English sentence into natural spoken {language}. "
                        "Return only the translated sentence."
                    ),
                },
                {"role": "user", "content": reference},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    text = str(response.json()["choices"][0]["message"]["content"]).strip()
    return text.strip().strip('"').strip()


def build_locale_file(
    *,
    en_path: Path,
    locale: str,
    data_root: Path,
    overwrite: bool,
    max_utterances: int | None,
    min_words: int,
) -> Path:
    rel = str(en_path.relative_to(data_root / "en"))
    out_path = data_root / locale / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    en_doc = json.loads(en_path.read_text(encoding="utf-8"))
    refs = iter_en_challenge_utterances(en_doc, min_words=min_words)
    if max_utterances is not None:
        refs = refs[: max(0, max_utterances)]

    utterance_entries = []
    for ref in refs:
        if locale == "en":
            source_text = ref.reference_text
        else:
            source_text = _translate(ref.reference_text, locale=locale)
        utterance_entries.append(
            build_utterance_entry(ref=ref, source_text=source_text)
        )

    locale_doc = build_locale_challenge_doc(
        en_doc=en_doc,
        locale=locale,
        utterance_entries=utterance_entries,
        derived_from=f"en/{rel}",
    )
    out_path.write_text(json.dumps(locale_doc, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--en-sample", action="append", dest="en_samples", default=[])
    parser.add_argument("--locale", action="append", choices=["fr", "de"], default=["fr", "de"])
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--miner-env", type=Path, default=None)
    args = parser.parse_args()

    subnet_root = Path(__file__).resolve().parents[1]
    workspace_root = workspace_root_from(subnet_root)
    data_root = miner_test_data_root(workspace_root)
    miner_env = args.miner_env or (workspace_root / "babelbit_miner/.env")
    _load_env(miner_env)

    en_samples = args.en_samples or [
        "npr/01/en-npr-001481.json",
        "npr/01/en-npr-001002.json",
        "npr/01/en-npr-001099.json",
    ]

    written: list[str] = []
    for rel in en_samples:
        en_path = data_root / "en" / rel
        if not en_path.is_file():
            raise FileNotFoundError(en_path)
        for locale in args.locale:
            out_path = build_locale_file(
                en_path=en_path,
                locale=locale,
                data_root=data_root,
                overwrite=args.overwrite,
                max_utterances=args.max_utterances,
                min_words=args.min_words,
            )
            written.append(str(out_path))

    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
