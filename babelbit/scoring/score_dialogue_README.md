# `score_dialogue.py`

`score_dialogue.py` scores per-step predictions from a dialogue JSONL log and writes both human-readable and JSON summaries.

## Current Scoring Model

For each predicted step the scorer computes:

- `lexical_similarity`: normalized character-level edit similarity
- `semantic_similarity`: calibrated cosine similarity from sentence-transformer embeddings
- `earliness`: `1 / (step + 1)`
- `U_step`: `((lex * lex_weight) + (sem * (1 - lex_weight))) * earliness`

Important current behavior:

- there is no perplexity penalty in `U_step`
- semantic scoring is embedding-based, not token Jaccard
- the default `lex_weight` is `0.0`, so the scorer is semantic-only unless you pass `--lex-weight`

## Semantic Calibration

Raw cosine similarity is calibrated against a baseline estimated from random pairs of ground-truth utterances in the same file:

```text
semantic = clamp01((cos_raw - baseline_b) / (1 - baseline_b))
```

This keeps unrelated-but-similar-sounding sentences from scoring too generously.

## Environment Controls

- `EMBEDDER_NAME` default: `mixedbread-ai/mxbai-embed-large-v1`
- `EMBED_DIM` default: `64`
- `EMBED_BATCH_SIZE` default: `32`
- `BB_SCORER_DEVICE` default: `cpu`
- `BASELINE_PAIRS` default: `100`
- `BASELINE_SEED` default: `0`

## Run

```bash
python score_dialogue.py --jsonl path/to/dialogue.jsonl
python score_dialogue.py --jsonl path/to/dialogue.jsonl --lex-weight 0.3
```

Outputs:

- `scores/<stem>-score.txt`
- `scores/<stem>-score.json`

## Gotchas

- If docs elsewhere mention `BASELINE_PAIRS=20000`, token-Jaccard semantics, or a perplexity term, those notes are stale.
- The scorer code is the source of truth if a markdown example disagrees with it.
