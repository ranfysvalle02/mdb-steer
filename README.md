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
            logistic model (fitted on calibration)
                            ▼
                     P(strong wins)
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
2. **Route.** Combine these signals into P(strong wins) with a class-balanced logistic model:
   - the similarity-weighted share of the k nearest calibration queries where the strong model
     won by more than `ROUTER_WIN_MARGIN`;
   - text difficulty features: multi-step wording, length, numbers;
   - the weak model's self-rated confidence (optional; its compute is charged to the router).

   The router sees **only the query text**.
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

On 15 held-out queries (CPU-only, `llama3.1:8b` vs `llama3.2:1b`), the router beats random
routing at every threshold from 0.1 to 0.6. At threshold 0.3 it passes a 5% quality guardrail
while sending 33% of traffic to the 1B model. The saving is small (4.9%), because on this
workload **even an oracle router saves only 14.7%**: the easy queries are also the cheap ones.
See [review.md](review.md) for the full results and caveats, and [blog.md](blog.md) for why the oracle matters.

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
| `ROUTER_THRESHOLD`      | `0.5`              | route strong when P(strong wins) ≥ this         |
| `ROUTER_K`              | `5`                | neighbours per decision                         |
| `ROUTER_WIN_MARGIN`     | `0.1`              | how much strong must beat weak to count as a win |
| `ROUTER_SELF_CONFIDENCE`| `true`             | use the weak model's self-confidence as a feature |
| `ROUTER_L2`             | `0.01`             | regularisation for the logistic combiner        |
| `GPU_COST_PER_HOUR`     | `1.00`             | prices measured model compute time              |
| `QUALITY_GUARDRAIL_PCT` | `5.0`              | max allowed quality drop vs. all-strong         |

Other commands: `fit` refits the router, and `regrade` re-judges stored calibration answers after a
judge change without regenerating them.

The threshold is the cost/quality dial. `sweep` replays the latest benchmark run from stored
telemetry across thresholds, making no model calls, to trace the whole trade-off curve:

```bash
docker compose run --rm app sweep --thresholds 0.2,0.4,0.5,0.6,0.8
docker compose run --rm app sweep --rescore   # re-evaluate with the latest fitted router
```

## Known limitations

- **Judge bias.** By default the strong model grades itself. Set `JUDGE_MODEL` to an independent model for fairer scores.
- **Small datasets.** The bundled sets (30 calibration / 15 benchmark) are for demonstration. Router quality scales with calibration coverage ([#1](https://github.com/ranfysvalle02/mdb-steer/issues/1)).
- **Benchmark cost.** The benchmark runs both models on every query so all strategies can be compared. Production routing runs only the chosen model.

## Project layout

```
mdb_steer/
  config.py    settings from env
  llm.py       Ollama native-API client (chat, embeddings, server-side timings)
  judge.py     LLM judge: categorical verdict against a reference
  store.py     MongoDB collections + Atlas Vector Search index
  features.py  difficulty features, self-confidence, logistic fit
  router.py    feature extraction, routing decision, fitting
  pipeline.py  calibrate / benchmark workflows and strategy comparison
  cli.py       command-line interface
data/
  calibration.jsonl   router training set
  benchmark.jsonl     held-out evaluation set
```

## License

MIT
