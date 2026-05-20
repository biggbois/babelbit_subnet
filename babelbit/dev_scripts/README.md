# Development Scripts Notes

These scripts are experimental utilities, not the canonical validator or scoring pipeline.

## Current Status

- `score_dialogue.py` in `babelbit/scoring/` is the maintained dialogue scorer.
- There is no `score_challenge.py` in this repository anymore.
- Older phrase-completion examples in this directory describe research workflows, not the current validator runtime.

## What To Use Instead

- Use the top-level `README.md` for validator and participation guidance.
- Use `babelbit/scoring/score_dialogue_README.md` for the current scorer behavior.
- Use the miner repository for the live miner request and response contract.

## Gotcha

Some older notes in this directory refer to token-Jaccard semantics, perplexity penalties, or end-to-end scripts that no longer exist. Those descriptions are historical only.
