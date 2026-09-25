# mdb-steer

**A learned LLM router backed by MongoDB Atlas Vector Search.** mdb-steer decides, for each query,
whether a small model is good enough or the large model is worth paying for. It makes that call
from evidence: how both models actually performed on similar past queries.

It runs entirely locally with Docker: MongoDB Atlas Local (with Vector Search) plus Ollama.
No API keys, no cloud accounts.

---

## Why

Sending every request to your largest model wastes compute when a small model would answer just
as well. But how much routing can save depends on your traffic: if the queries a small model can
handle are also the cheap ones, even a perfect router saves little. mdb-steer measures both
things, the **ceiling** (an oracle router) and how much of it a real router **captures**.

## How it works

```
                        query text
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   embed → Atlas      difficulty text     weak model's
   $vectorSearch      features (multi-     self-rated
   k-NN vote          step, length, nums)  confidence
          └─────────────────┼──────────────────┘
                            ▼
            logistic model + cross-validated threshold
                            ▼
                    P(strong needed)
             ≥ threshold    │    < threshold
              ┌─────────────┴─────────────┐
              ▼                           ▼
        strong model                 weak model
              └─────────────┬─────────────┘
                            ▼
                   telemetry → MongoDB
```

1. **Calibrate.** Run *both* models on a calibration set. An LLM judge gives each answer a
   verdict against a reference (CORRECT 1.0 / PARTIAL 0.5 / INCORRECT 0.0). Store the query
   embedding and both scores in MongoDB, build an Atlas Vector Search index, and fit the router.
2. **Route.** Combine these signals into P(strong needed) with a class-balanced logistic model:
   - the similarity-weighted share of the k nearest calibration queries where the **weak model
     failed** (the `ROUTER_LABEL`; predicting weak failure is far more learnable than predicting
     a strong win);
   - text difficulty features: multi-step wording, length, numbers;
   - optionally, the weak model's self-rated confidence (off by default: no signal on our data).

   The router sees **only the query text**. The threshold is chosen by 5-fold cross-validation
   on the calibration set: the largest saving whose quality drop stays within the guardrail.
3. **Benchmark.** On a **held-out** set, route each query *and* run both models, so the router can
   be compared against every alternative:

| strategy     | meaning                                                              |
| ------------ | -------------------------------------------------------------------- |
| `all_strong` | quality ceiling / cost ceiling                                       |
| `all_weak`   | cost floor                                                           |
| `router`     | mdb-steer                                                            |
| `random`     | expected result of offloading the *same share* of traffic at random  |
| `oracle`     | cheapest model achieving the best score per query (savings ceiling)  |

   Check the `oracle` first: it's the most routing can save on this traffic. Then **lift over
   random** shows whether the router is learning anything. The run exits non-zero if the quality
   drop vs. `all_strong` exceeds `QUALITY_GUARDRAIL_PCT`, so it can gate CI or a deploy.

**Measured, not assumed.** Cost is Ollama's server-reported compute time (prompt eval +
generation, *excluding* model load) × `GPU_COST_PER_HOUR`. Token counts come from Ollama. Compose
keeps all models resident (`OLLAMA_MAX_LOADED_MODELS=3`) so swapping doesn't distort latency.

## Results

On 45 held-out queries (CPU-only, `llama3.1:8b` vs `llama3.2:1b`, 150 calibration queries):

- **Ceiling:** the oracle sends 64% of traffic to the 1B model with no quality loss but saves
  only **17.8%**. Sending everything to the 1B saves 36.9%. On this hardware the 1B model is only
  about 1.35–1.6× cheaper per query than the 8B.
- **Router:** beats random routing at every threshold up to 0.65. At 0.40: 24% offload,
  **8.6%** savings, 3.8% quality drop (passes a 5% guardrail). The threshold chosen by
  cross-validation (0.45) delivered a 7.6% drop on held-out data and fails the guardrail.

See [review.md](review.md) for the full results and caveats, and [blog.md](blog.md) for what
sets the savings ceiling.

## Quickstart

```bash
docker compose up -d mongodb ollama
docker compose run --rm app calibrate      # pulls models on first run (~6 GB), then fits the router
docker compose run --rm app benchmark
docker compose run --rm app sweep --rescore  # cost/quality curve, no model calls
docker compose run --rm app route "Implement a lock-free queue in C++" --answer
```

Or run the Python locally against the containers:

```bash
pip install -e .
docker compose up -d mongodb ollama && docker compose run --rm ollama-pull
python -m mdb_steer calibrate
python -m mdb_steer benchmark
```

Results depend on your hardware and models. Every run is stored, so you can compare runs over time.

## Data model (database `mdb_steer`)

| collection       | contents                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `calibration`    | one doc per calibration query: embedding, per-model scores, full answers, judge verdicts. Vector-indexed. |
| `telemetry`      | one doc per routed query: decision, p_strong, features, neighbours used, both models' attempts  |
| `router_models`  | every fit: feature weights, standardisation, training rows, train accuracy                      |
| `benchmark_runs` | run summary: every strategy's quality / cost / offload, guardrail verdict, settings snapshot     |

The telemetry is also the retraining signal. Any benchmark query can be promoted into
`calibration` to sharpen the router where it is weakest.

## Configuration

All settings are environment variables (see `mdb_steer/config.py`):

| variable                | default            | purpose                                         |
| ----------------------- | ------------------ | ----------------------------------------------- |
| `STRONG_MODEL`          | `llama3.1:8b`      | expensive model                                 |
| `WEAK_MODEL`            | `llama3.2:1b`      | cheap model                                     |
| `JUDGE_MODEL`           | = `STRONG_MODEL`   | grader; use a third model to avoid self-bias    |
| `EMBED_MODEL`           | `nomic-embed-text` | query embeddings                                |
| `ROUTER_THRESHOLD`      | cross-validated    | route strong when P(strong needed) ≥ this; unset uses the fitted threshold |
| `ROUTER_LABEL`          | `weak_fails`       | training label: `weak_fails` or `strong_wins`   |
| `ROUTER_CV_FOLDS`       | `5`                | folds for threshold selection                   |
| `ROUTER_K`              | `5`                | neighbours per decision                         |
| `ROUTER_WIN_MARGIN`     | `0.1`              | margin for the `strong_wins` label              |
| `ROUTER_SELF_CONFIDENCE`| `false`            | use the weak model's self-confidence as a feature |
| `ROUTER_L2`             | `0.01`             | regularisation for the logistic combiner        |
| `GPU_COST_PER_HOUR`     | `1.00`             | prices measured model compute time              |
| `QUALITY_GUARDRAIL_PCT` | `5.0`              | max allowed quality drop vs. all-strong         |

Other commands: `fit` refits the router and re-selects the threshold, and `regrade` re-judges stored
calibration answers after a judge change without regenerating them. `calibrate` resumes where it
left off; `--force` re-runs everything.

The threshold is the cost/quality dial. `sweep` replays the latest benchmark run from stored
telemetry across thresholds, making no model calls, to trace the whole trade-off curve:

```bash
docker compose run --rm app sweep --thresholds 0.2,0.4,0.5,0.6,0.8
docker compose run --rm app sweep --rescore   # re-evaluate with the latest fitted router
```

## Known limitations

- **Judge bias.** By default the strong model grades itself. Set `JUDGE_MODEL` to an independent model for fairer scores.
- **Sample size.** 45 benchmark queries give wide error bars; a couple of queries can flip a guardrail result ([#2](https://github.com/ranfysvalle02/mdb-steer/issues/2)).
- **CPU hardware.** Results were measured on CPU, where the 1B model is only modestly cheaper than the 8B. A GPU changes the savings ceiling.
- **Benchmark cost.** The benchmark runs both models on every query so all strategies can be compared. Production routing runs only the chosen model.

## Project layout

```
mdb_steer/
  config.py    settings from env
  llm.py       Ollama native-API client (chat, embeddings, server-side timings)
  judge.py     LLM judge: categorical verdict against a reference
  store.py     MongoDB collections + Atlas Vector Search index
  features.py  difficulty features, self-confidence, logistic fit
  router.py    labels, kNN vote, routing decision, fitting + CV threshold
  evaluation.py  strategy comparison, replay, threshold selection
  pipeline.py  calibrate / regrade / benchmark / sweep workflows
  cli.py       command-line interface
tests/         fast unit tests, no models or database needed
data/
  calibration.jsonl   150 router training queries
  benchmark.jsonl     45 held-out evaluation queries
```

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/pytest -q
```

CI runs both on every push and pull request.

## License

MIT
