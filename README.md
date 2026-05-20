<p align="center">
  <img width="265" height="281" alt="Babelbit logo Black" src="https://github.com/user-attachments/assets/055577f8-0ff4-4d67-9153-e66c00688bb2" />
</p>

[Experienced Bittensor contributors can jump to our detailed subnet instructions](#instructions)

# BIG REWARDS FOR CREATING THE BEST AUTOMATED SPEECH INTERPRETER IN THE WORLD

We are already ahead of SOTA on straight speech-to-speech translation but that is not what a human interpreter does. 

## Babelbit: Bittensor SN59: PHASE 2 - Product Development
For the last six months, since the subnet went live, we have been setting challenges for miners to prove one of the hypotheses which inspired our project to create a simultaneous translation system, capable of outperforming human interpreters. The particular human trait we focused on was prediction. Machine translation focuses on accuracy at the expense of latency, because until the advent of LLMs, neural networks were not that good at prediction. However, all human beings anticipate what is about to be said, when they listen to speech. Our hypothesis - now proven - was that LLMs can be trained for **phrase completion**, i.e. to predict beyond the next-word prediction which is the essential mechanism by which transformer networks generate output. 
**
That is, just like a person, can we start translating the phrase "May the Force be with you" after only hearing "May the Force...".**

Having proved that this is possible, we are now moving into a completely new type of competition. We are providing our current state-of-the art technology and rewarding miners for improving it. Things have been going very well, and our new workflow is something like this:
1. We come up with a method we think is useful for speech-to-speech translation
2. We build a prototype which can be integrated into a base script for a incentivised mining process on Bittensor
3. We run the competition indefinitely until performance is significantly better
4. We update the base script using selections from the improvements made by miners
5. We keep going until we are at the maximum possible performance for that particular feature. 

However, there are a **lot** of features. We have developed some serious ambitions by researching what it is that human interpreters do. 

**Getting Google Translate to run 10x faster would not be a good interpreter**

That said, eliminating latency is still a valid goal, as it gives us the bandwidth to do a lot more impressive stuff.

## The Fallacy of Machine Translation Benchmarks
The gold standard of machine translation is to create the most accurate literal translation of the input, as fast as possible. However, interpreters do something very different. Human speech can be intrinsically difficult to understand for all kinds of reasons. The first three can help latency, but some of the others are about the most important aspect of the job - imparting **understanding**.

The normal latency benchmarks take the average delay between words in the input, and equivalent words in the output. We don't translate that way. 
**
- FRENCH INPUT: Je pense que vous avez tout à fait raison.
- GOOGLE: I think you're absolutely right.
- BABELBIT: Agreed!**

Average word latency is useless. We have devised a new benchmark, which we call:

**Phrase Completion Latency**. 

We measure the delay between the end of a phrase in the input, and the point in the output when the meaning has been imparted accurately. 

## The competitions we are lining up for you
We still want to work on latency, and we have three different approaches to doing that. The first will be very familiar to our existing miners.

1. Prediction - reducing latency by anticipating what is about to be said, and translating it early, e.g. "May the force..."
2. By doing everything in speech mode - by creating a transformer network of tokenised speech, we save time by not converting speech to text, translating the text, and then generating the speech. This is bundled in our architecture, but we are sure that the tokenisation of speech, and the generation of translations from speech tokens can be improved upon.
3. Paraphrasing - because we can generate the translation from speech tokens, we can include criteria like shortening a wordy sentence, without doing two passes. Legacy tech would translate and then summarise. This is like if you ask ChatGPT to create a English review of a French book. It can understand the French book, and understand how to say what it needs to in English. It would not translate the book.

## Future Features
Some of these will have their own competitions and sometimes we will run multiple competitions in parallel. Different miners might be interested in different aspects of our product development:

These next ones are all about understanding, but some help with latency as well.

4. Remove repetition
5. Remove interjections (ums and ahs)
6. Remove expletives
7. Expand on ellipses - if words are missed out, they might need to be restored in the target languae
8. Cultural sensitivities - rewording a sentence which might be offensive to people with certain beliefs

## Grammarly for Speech
This last one is even more interesting in some ways because it doesn't involve translation at all. We created a demo to show what we mean by paraphrasing etc, which was English-to-English:

https://babelbit.ai/demo

We started getting sales enquiries from people who want to do things like broadcast live interviews with people who have unusual dialects, or have stammers, or are simply nervous, and repeat themselves. Josh, our chief scientist built a working version and we now think that this could be like Grammarly for Speech. 

We have our first sales meeting this week!

**SO YOU CAN SEE - WE NEED THE BEST SPEECH AND LANGUAGE MACHINE LEARNING EXPERTS TO TRAIN OUR NETWORK TO DO ALL THESE THINGS AT ONCE**

Below are the explanations of how the competition works and how to get started, but we are not dogmatic. Most people will probably start with our base script. 

As well as the a prebuilt miner we are giving you access to a bunch of our R&D. 

1. You can use our SOTA model
2. You can start with a speech mode model like Moshi which has no translation capability
3. You can play with our R&D experiments which can do various things - check out the folders below
4. You could build your own from scratch

Links:
XXXXXXXXXXX
XXXXXXXXXXX

**Whichever way you do it, if you can improve the performance of our product, you will be handsomely rewarded**


# INSTRUCTIONS

## 1: Babelbit's Goal

Babelbit is a Bittensor speech-to-speech subnet focused on low-latency machine interpretation.

The goal is not just to translate correctly. The goal is to behave more like a good human interpreter: understand speech as it arrives, respond quickly, and produce useful target-language speech rather than waiting for a perfect full-utterance transcript.

That difference matters. A conventional speech pipeline can be very accurate and still feel too slow or too literal for live use. Babelbit is built around the idea that latency, delivery style, and spoken usefulness are part of the task, not just post-processing details.

## 2: What Makes The Subnet Different

The current subnet is organized around speech-to-speech behavior rather than a text-first benchmark.

At a high level, miners are being pushed toward systems that can:

1. consume source audio directly
2. decide when enough meaning is available to begin responding
3. generate target-language speech with low delay
4. preserve meaning while still sounding natural and usable in real-time

This does not force one model architecture. A miner might still use text internally, a speech-token model, or a hybrid stack. What matters is the validator-facing behavior and the resulting quality/latency tradeoff.

## 3: What We Mean By "Interpretation"

The gold standard is not word-for-word literalism.

In live interpretation, the best output is often shorter, clearer, and more listener-friendly than a strict translation. Good systems may need to:

1. remove accidental repetition
2. compress rambling phrasing
3. replace figurative language with clearer literal meaning
4. soften gratuitous profanity while preserving intent
5. choose culturally appropriate phrasing

That is why Babelbit should not be understood as "just another translation benchmark." The user experience depends on whether the model produces good spoken delivery under time pressure.

## 4: Challenge Structure


Here are some examples, which will make it clear how different our approach is.
**NOTE: The best interpreters diverge from precise, literal translations of what is said, as follows:**
1. Eliminate accidental repetitions
    1. *I.. I think.. I think that it would be best to finish now.*
    2. *It would be best to end here*
2. Eliminate completely gratuitous expletives
    1. *What a load of f--king nonsense*
    2. *What a load of nonsense*
3. Replace meaningful expletives with polite alternatives
    1. *What a load of shit*
    2. *What a load of rubbish*
4. Paraphrase rambling expressions to make them succinct
    1. *I mean, when you really stop and think about it, it kind of speaks for itself*
    2. *If you think about it, it is obvious*
5. Replace figurative or metaphorical expressions with clear, literal ones
    1. *That's not cricket*
    2. *That's not fair*
6. Be as culturally sensitive as possible
    1. *Muhammad ibn Abdullah was an Abrahamic religious cult leader*
    2. *The Prophet Muhammad, is regarded by Muslims as the final messenger of God*

## 4: A Fairer Approach to Mining Challenges

The text prediction challenge we designed in October 2025, was a task designed to reward miners that make useful predictions early, including predictions that are semantically right before the full utterance is revealed. This allowed us to prove that it was possible to reduce translation latency in a new way.

However, we noticed that some creative approaches to prediction didn't score well, but inspired some good ideas. So it occurred to us that while we still want to reward the biggest performance gains, we don't want any hard-working machine learning engineer to be working for nothing.

So we have come up with a two phase contest - a qualifying round where every contestant gets a proportion of the allotted emissions, and The Arena where the qualifying contestants compete to win the rest.

This is a new evolution of our development, and we will need our mining community to remain ever adaptable with us as we progress - after all we are trying to maximise the performance of the world's first machine interpreter. So this is how we're planning things at launch:

**The Qualifying Round** will share 20% of the emissions between all the contestants (unless they're caught cheating), in proportion to their scores. It probably won't make anyone rich, but our hope is that the hard work will be rewarded in another way - getting better and better - until you qualify for the second phase.

The qualifiers then compete in **The Arena** for a chance at winning the remaining 80%.

## 5: Current Validator Stack

This repository is the validator-side operator guide for the Babelbit subnet.

It is primarily for:

- validators running `bb runner`, `bb validate`, `bb signer`, and `bb subtensor-gateway`
- miner operators who need the validator-facing compatibility rules for qualifying and arena participation

It is not a full miner implementation. Use the miner repository for serving code, model runtime details, and miner-specific tests.

### 5.1: What Is Current

Current validator/runtime behaviour worth knowing up front:

- the runner and validator use local files for challenge status and score artifacts
- the shared status directory is controlled by `BB_CHALLENGE_STATUS_DIR`
- score outputs are written under `BB_OUTPUT_SCORES_DIR`
- Postgres-related settings still exist for auxiliary integrations, but they are not the core persistence path for the current scoring loop
- qualifying discovery still starts from Bittensor axon metadata

If this repo is split across multiple processes or hosts, `BB_CHALLENGE_STATUS_DIR` must point at shared storage. The Docker setup already mounts a shared volume for that directory.

### 5.2: Challenge Tiers

The subnet currently uses two tiers:

- `qualifying`: proportional rewards across qualifying miners
- `arena`: winner-takes-all for the arena slot

Arena eligibility is derived from recent qualifying performance. The exact selection mechanics can evolve, so treat the live validator behavior as authoritative.

## 6: Validator Setup

### 6.1: Prerequisites

- a Bittensor wallet and hotkey
- Python `3.10`-`3.13` for local runs
- Docker for the recommended deployment path
- optional object storage for logs/artifacts

### 6.2: Install The Tooling

```bash
pip install bittensor-cli
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync
```

The CLI entrypoint is `bb`.

### 6.3: Required Environment

Start from [`env.example`](./env.example). The minimum validator settings are:

```bash
BITTENSOR_WALLET_PATH=~/.bittensor/wallets/my-wallet/hotkeys/my-hotkey
BITTENSOR_WALLET_COLD=my-wallet
BITTENSOR_WALLET_HOT=my-hotkey

BABELBIT_NETUID=59
BITTENSOR_SUBTENSOR_ENDPOINT=finney

SIGNER_URL=http://127.0.0.1:8080
SUBTENSOR_GATEWAY_URL=http://127.0.0.1:8090

BB_UTTERANCE_ENGINE_URL=https://api.babelbit.ai/
BB_SUBMIT_API_URL=https://scoring.babelbit.ai/
BB_ARENA_GATEWAY_URL=https://gw.babelbit.ai/

BB_CHALLENGE_STATUS_DIR=data/challenge_status
BB_OUTPUT_LOGS_DIR=logs
BB_OUTPUT_SCORES_DIR=scores
```

### 6.4: Main Commands

- `bb runner`: executes the qualifying, solo, and arena runner flows
- `bb validate`: calculates and submits weights
- `bb signer`: runs the local signing service
- `bb subtensor-gateway`: runs the local subtensor gateway

### 6.5: Recommended Deployment

```bash
docker compose down
docker compose pull
docker compose up --build -d
docker compose logs -f --tail 100
```

### 6.6: Local Run

```bash
bb -vv signer
bb -vv subtensor-gateway
bb -vv runner
bb -vv validate
```

## 7: Validator Gotchas

These were easy to miss in the older docs and are now explicit:

- utterance-engine auth is mandatory in normal operation; runner startup authenticates against the engine and will fail early if that path is broken
- `BB_ENABLE_SOLO_CHALLENGE=1` is the current default, so a solo phase may run in addition to the main qualifying flow
- `BB_ENABLE_ARENA_CHALLENGE` gates the arena phase independently of qualifying
- runner dedupe is file-based: if score files already exist for a challenge/type, the runner skips reprocessing that challenge
- arena-managed health checks expect `GET /health`, not `GET /healthz`
- Hugging Face accessibility matters for committed-model flows; gated or inaccessible revisions can be filtered out even if the miner is otherwise reachable

### 7.1: Postgres Note (Optional)

`sql/init.sql` is not a generic public bootstrap script. It contains owner-specific assumptions and should be adapted before use in a fresh external environment.

## 8: Miner Participation

### 8.1: Qualifying vs Managed Submission

There are two distinct surfaces miners should understand:

1. Qualifying discovery
2. Managed submission and arena participation

For qualifying discovery, a miner can still be discovered from valid Bittensor axon metadata alone.

For managed submission flows, miners should also maintain:

- a Docker image for the submitted runtime
- a Hugging Face repository handle for the submitted model

Those managed-flow mechanics are intentionally only described at a high level here.

### 8.2: Validator-Facing Expectations

The validator currently expects miners to:

- register on netuid `59`
- publish reachable axon IP/port metadata
- accept Bittensor-style signed request headers
- expose a prediction path compatible with the active miner contract used by the validator

The concrete request and response schema should be taken from the current miner repository and validator tests, not from older examples that assumed a text-completion-only flow.

### 8.3: Local Compatibility Testing

If the validator runs in Docker and your miner runs on the host, enable local routing:

```bash
BB_DEV_MODE=1
BB_LOCAL_MINER_IP=127.0.0.1
```

Then run the miner-side tests from the miner repository you are actually using.

## 9: Scoring Mechanism

The validator scores the audio a miner returns. It does not ask the miner for a transcript and trust it.

The scoring model has three main questions:

1. Did the miner say the right thing?
2. Did the miner speak at a believable speed?
3. Did the miner finish quickly enough?

The first two are gates. If the audio is not close enough to the reference, or if the speaking rate is clearly wrong, the utterance gets `0`. If it passes those gates, the score is based on latency.

### 9.1: What The Validator Compares Against

Each utterance has reference data: the expected target-language text, and usually enough word timing or words-per-second information to know how fast the reference speech should be.

That reference data can come directly from the utterance engine response, or from files under `BB_AUDIO_SCORING_METADATA_ROOT`. The score output records where it came from in `scoring_metadata_source`.

### 9.2: Turning Miner Audio Into Text

The miner returns audio, so the validator first transcribes that audio with its own STT model. This keeps the scoring path honest: a miner cannot claim it said one thing while returning different audio.

Current defaults:

- `BB_AUDIO_SCORING_STT_MODEL=faster-whisper-small`
- `BB_AUDIO_SCORING_STT_DEVICE=cpu`
- `BB_AUDIO_SCORING_STT_CACHE_PATH=~/.babelbit/audio_scoring/stt_cache.jsonl`

The STT step produces two things the scorer uses:

- transcript text, used to check meaning
- word timestamps, used to estimate speaking rate

Repeated audio is cached by WAV hash. On CPU, faster-whisper runs the default `float16` model setting as `int8`. In practice, this transcription step is the slow part of scoring on CPU-only validators.

### 9.3: Meaning Check

Once the validator has a transcript, it compares that transcript to the reference text using sentence embeddings.

This is intentionally not exact string matching. A good interpretation may use different words while preserving the same meaning. The embedding comparison gives an `accuracy` score from `0` to `1`.

Current default embedder:

- `BB_AUDIO_SCORING_EMBEDDER=sentence-transformers/all-MiniLM-L6-v2`

By default, an utterance needs `accuracy >= 0.65` to be eligible for a non-zero score.

### 9.4: Speaking-Rate Check

The scorer also checks whether the returned audio sounds like plausible speech rather than being wildly too fast or too slow.

It estimates the miner's words per second from the STT word timestamps, then compares that against the reference speaking rate.

Current defaults:

- `BB_AUDIO_SCORING_RATE_LOWER=0.3`
- `BB_AUDIO_SCORING_RATE_UPPER=1.3`

So if the miner is speaking at less than `0.3x` or more than `1.3x` the reference rate, the utterance fails this gate and scores `0`.

### 9.5: Latency Check

For audio that passes the meaning and speaking-rate gates, latency decides the score.

The scorer looks at when the miner started returning audio and how long that audio lasted. From that it computes when the miner effectively finished. Finishing before or at the source utterance end gets a latency score of `1.0`.

Some delay is allowed. The allowed delay is based on the source utterance length:

- default allowance is `30%` of source duration
- allowance is never less than `2s`
- allowance is never more than `10s`

Inside that window, the score falls smoothly from `1.0` toward `0.0`. Once the miner is later than the allowed window, latency score is `0.0`.

### 9.6: Final Utterance Score

The default scoring rule is:

```text
if meaning passes and speaking rate passes:
    score = latency_score
else:
    score = 0
```

This means the current validator does not give partial credit for being fast but wrong. A miner first has to produce audio that is close enough in meaning and plausible as speech. After that, lower latency wins.

### 9.7: Challenge Score

The runner scores each miner on each utterance, then averages that miner's utterance scores into a challenge score.

So if a challenge has several utterances, one bad utterance hurts the average, but it does not automatically erase the whole challenge unless every utterance scores badly.

### 9.8: Failures

If scoring cannot complete for an utterance, that utterance gets `0`.

Common causes are:

- STT failed
- the returned audio produced an empty transcript
- reference metadata was missing or malformed

These failures are marked with `score_method = semantic_audio_v1_error` and a `score_error` string.

### 9.9: Performance

On CPU-only validators, STT is the main cost.

With the current defaults, `tests/benchmarks/stress_stt.py` measured about `1402s` of wall time for a synthetic `250 miners x 60s` workload on an `8 vCPU / 16 GB RAM` DigitalOcean Premium AMD VM. A smaller `4 vCPU / 8 GB RAM` DigitalOcean Regular Intel VM was materially slower and would push the same workload into hour-scale runtime.

See [`min_compute.yml`](./min_compute.yml) for the current benchmark-based floor.

## 10: Troubleshooting

- If the runner appears to skip work unexpectedly, inspect existing files under `BB_OUTPUT_SCORES_DIR` and `BB_CHALLENGE_STATUS_DIR` first.
- If arena health checks fail while the miner otherwise looks healthy, confirm that the managed endpoint serves `GET /health`.
- If a committed or managed miner is not being considered, verify that its published or submitted Hugging Face revision is readable from the validator environment.

## 11: What would we Try if we were miners?

### 11.1: Try out different model architectures.

**1: MAYBE THE ORTHODOXY IS RIGHT AFTER ALL**
We made a lot of our one-shot approach above, but who knows? It might be possible to optimise the old-school STT-translate-TTS, and outperform the one-shot version, because that way you'd be building on 40 years of research and optimisation. Each stage can be independently swapped, profiled, and optimised. It's entirely possible that a tightly tuned cascade outperforms a one-shot model, especially early on.

**2: THE MOSHI WAY**
One of our recent starting points:
Audio -> Audio+Semantic tokens +predicted text tokens -> transformer -> audio tokens -> Audio.
This has the advantage of combining very well-established text-prediction with generating speech from tokenised audio.

**3: DISTILLATION OF TRANSFORMERS INTO RECURRENT ARCHITECTURES**
Transformer attention is quadratic in sequence length, which directly impacts latency. A recurrent-style architecture — such as a state-space model (Mamba) or a linear-attention variant (RWKV) — gives you linear-time inference while retaining much of the original model's quality.

Check out the following:
- https://arxiv.org/abs/2603.15569
- https://arxiv.org/abs/2312.00752
- https://github.com/state-spaces/mamba
- https://github.com/BlinkDL/RWKV-LM
- https://arxiv.org/abs/2503.14456

### 11.2: CAUSALITY AND RECEPTIVE FIELD TRICKS

**1: SHIFTING THE RECEPTIVE FIELD TO THE PAST**, no future lookahead (“causality”) - most models, particularly Convnets, assume the receptive field is centred around “now”.  This add architectural latency of 1/2 the receptive field.  Shiftng the receptive field to be fully causal removes this latency, but requires retraining the model.

**2: KV-CACHING** to avoid redundant recomputation.

**3: SPECULATIVE DECODING** combines well with KV-Caching -- Use a smaller, faster draft model to generate multiple candidate tokens in parallel, then verify them against the main model.

**4: SMALLER MODELS** A 12-layer transformer has roughly half the per-step latency of a 24-layer one. Requires balancing reduced accuracy with reduced latency.

### 11.3: DSP tricks

**1: LOWERING THE SAMPLE RATE** speeds up the entire pipeline. How low can you go?

**2: SMALLER HOPS** for the time -> frequency FFT. More computation, however.

**3: USE A CAUSAL NEURAL VOCODER** like a fully-causal HifiGAN which effectively has zero latency (as opposed to an inverse FFT which has a 3X hop-size latency)

### 11.4: Train/Prompt a language model to perform like the best human interpreter
This is one of our favourite approaches, and outlined in section 3. When you're not sure about the limits of AI's capabilities, think about what humans do.

There are basically three ways of pushing a model in this direction:

- **Fine-Tuning on Interpreter-Quality Pairs**:
Curate a dataset of source speech paired with high-quality interpretations (not literal translations), and fine-tune. This can be done with LoRA or with full fine-tuning of some or all layers — though full fine-tuning risks catastrophic forgetting and requires a lot more VRAM (all tensors must track gradients). Or if you're really constrained, use prompt tuning (learn embeddings for a few soft tokens)

- **2: Prompt Engineering**:
This is the cheapest experiment to run, and a good starting point before committing to fine-tuning.

- **3: Full Training from Scratch**:
Starting from an uninitialised model and training end-to-end on the speech-to-speech task (as in Google's Translatotron lineage), or on a hybrid objective that conditions on LM-generated "inner monologue" text (closer to the Hibiki approach). This give you the most architectural freedom.
