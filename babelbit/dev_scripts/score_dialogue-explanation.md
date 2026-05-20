### Setup

- A dialogue has utterances `u = 1..U`.
- Each utterance has a final ground-truth string `G_u`.
- At each prediction step `s = 0..S_u-1`, the model emits a full prediction `P_{u,s}`.

### Similarities At Each Step

1. Lexical similarity uses normalized character-level edit similarity.

2. Semantic similarity uses sentence-transformer embeddings.

For each `(P_{u,s}, G_u)` pair:

- compute normalized embeddings
- compute raw cosine similarity `cos_raw`
- estimate a dialogue baseline `b` from random pairs of ground-truth utterances
- calibrate with:

```text
sem(P_{u,s}, G_u) = clamp01((cos_raw - b) / (1 - b))
```

### Earliness Weight

Earlier correct predictions are better:

```text
earliness(s) = 1 / (s + 1)
```

### Per-Step Utility

With lexical weight `w`:

```text
U_{u,s} = (w * lex(P_{u,s}, G_u) + (1 - w) * sem(P_{u,s}, G_u)) * (1 / (s + 1))
```

Current code defaults `w` to `0.0`, so scoring is semantic-only unless `--lex-weight` is supplied.

### Per-Utterance Score

The utterance score is the best step utility:

```text
U_u = max_s U_{u,s}
```

### Dialogue Score

The dialogue score is the average utterance score:

```text
U_dialogue = (1 / U) * sum_u U_u
```

### Notes

- There is no perplexity penalty in the current scorer.
- Older Jaccard-based explanations are obsolete.
- If this note disagrees with `score_dialogue.py`, trust the code.
