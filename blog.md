# Before You Build an LLM Router, Measure the Oracle

*What mdb-steer, a learned router on Ollama and MongoDB Atlas Vector Search, taught us about where routing savings come from.*

---

LLM routing sounds like an easy win. Most queries don't need your biggest model, so send "What's the capital of Japan?" to a 1B model and save the 8B for Dijkstra and Raft. Cost goes down and quality stays the same.

We built a router to test this, measured it on 150 calibration and 45 held-out queries, and found something more useful than a good-looking benchmark number:

> **How much routing can save is decided before you train a router.** It depends on how much of
> your cost sits in queries the small model can handle, and on how much cheaper the small model
> really is on your hardware. Measure the oracle first.

## The oracle

The **oracle** is a router with perfect hindsight. For every query it picks the cheapest model that still gets the best score. No real router can beat it, so the oracle's savings are the most routing can ever achieve on a workload.

It's cheap to compute. Run both models on a sample of your traffic, grade the answers, and for each query take the cheaper model whenever it did as well as the expensive one.

On 45 held-out queries spanning facts, rewrites, code, math, reasoning and long "easy but wordy" tasks, with `llama3.1:8b` as the strong model and `llama3.2:1b` as the weak one:

| strategy | quality | cost | sent to 1B |
|---|---|---|---|
| all-strong | 0.878 | $0.346 | 0% |
| all-weak | 0.611 | $0.218 | 100% |
| **oracle** | **0.889** | **$0.284** | **64%** |

The oracle sends almost two-thirds of the traffic to the 1B model and quality goes *up* slightly, because the small model is sometimes the one that gets it right.

**It saves only 17.8%.**

## Two factors set the ceiling

Break the oracle down by query type and both factors show up:

| category | 1B was good enough | oracle saves |
|---|---|---|
| factual | 5 / 7 | 50.8% |
| long_easy (summaries, rewrites, extraction) | **6 / 6** | 26.1% |
| reasoning | 3 / 8 | 19.2% |
| rewrite | 4 / 6 | 16.5% |
| coding | 8 / 10 | 15.1% |
| math | 3 / 8 | 3.6% |

**Factor 1: where the cost is.** Math and reasoning produce long, expensive answers, and the 1B mostly gets them wrong, so that cost can't move. The queries the 1B handles well tend to be the cheap ones.

**Factor 2: how much cheaper the small model really is.** Look at long_easy. The 1B handled *every* summary, rewrite and extraction task, yet the oracle saves only 26%. On this hardware (CPU, in Docker) the 1B is only about 1.35× cheaper per query on those tasks, and sending all traffic to it saves just 36.9%. A model 8× smaller isn't 8× cheaper when your hardware can't take advantage of the difference.

So routing savings ≈ (share of cost in queries the small model can handle) × (the small model's real per-query cost gap). The router can't change either factor. Measure both before you build one.

## What the router captures

mdb-steer's router sees only the query text. It combines a **neighbour vote** (`$vectorSearch` over graded calibration queries) with a few text features in a class-balanced logistic model, and picks its threshold by cross-validation on the calibration set, never on the benchmark.

On the held-out set:

| threshold | sent to 1B | savings | quality drop | vs. random routing |
|---|---|---|---|---|
| 0.35 | 18% | 5.4% | 3.8% | +0.014 |
| **0.40** | **24%** | **8.6%** | **3.8%** | **+0.032** |
| 0.45 *(chosen by cross-validation)* | 38% | 13.3% | 7.6% | +0.034 |
| 0.50 | 56% | 18.2% | 7.6% | +0.081 |

It beats random routing at every threshold up to 0.65, so it's picking up real signal. At 0.40 it captures about half of the oracle's savings within a 5% quality budget. The threshold cross-validation picked (0.45) predicted a 4.5% quality drop and delivered 7.6% on held-out data, which shows how noisy 45 queries still are. Treat these numbers as directional.

## The label matters more than the features

The biggest improvement didn't come from a new feature. It came from changing what the router is asked to predict.

We first trained it on "did the strong model beat the weak one?". With 150 calibration queries, every signal we had predicted that label at about chance (AUC ≈ 0.50). The label is the difference of two noisy judge verdicts, and it treats "both models failed" the same as "easy query".

Switching the label to **"did the weak model fail?"**, a single verdict, made the neighbour vote a real signal (AUC ≈ 0.65). The same embeddings and the same data, just a better question. With the old label, the router routed worse than random. With the new one it beats random at every threshold up to 0.65.

A related warning: on our first 30 calibration queries, text features looked predictive. At 150 they weren't. A small calibration set can make a feature look useful when it's noise.

## Details that make the numbers trustworthy

- **Use categorical verdicts, not scales.** An 8B judge scoring 0–10 gave 8/10 to an answer of 30 for the arrangements of "BANANA" (the correct answer is 60). CORRECT, PARTIAL or INCORRECT on the final answer fixed that.
- **Count compute, not wall-clock time.** When several models share one Ollama instance they push each other out of memory, and latency ends up measuring loading. Keep them all loaded and price Ollama's reported generation time.
- **Balance the classes.** Otherwise the fit learns to always answer the majority class.
- **Choose thresholds without looking at the test set,** and report when the held-out result misses the prediction.
- **Compare against random routing at the same offload rate.** Offloading always saves money. The question is whether it loses less quality than choosing the same share at random.

## Why everything lives in MongoDB

Every answer, verdict, embedding, routing feature, neighbour list and fitted model is stored in MongoDB: `calibration` (vector-indexed), `telemetry`, `router_models` and `benchmark_runs`. Running the models is the expensive part, so storing everything makes it a one-time cost:

- **Change the judge?** `regrade` re-grades stored answers.
- **Change the label or features?** `fit`, then `sweep --rescore` recomputes the neighbour vote from the stored neighbour lists and replays the held-out set.
- **Change the threshold?** `sweep` traces the full cost/quality curve.

The label switch above was tested this way, in seconds, on hours' worth of stored inference.

## Try it on your traffic

Everything runs locally in Docker (MongoDB Atlas Local with Vector Search, plus Ollama). No API keys.

```bash
git clone https://github.com/ranfysvalle02/mdb-steer && cd mdb-steer
docker compose up -d mongodb ollama
docker compose run --rm app calibrate
docker compose run --rm app benchmark
docker compose run --rm app sweep --rescore
```

Replace `data/benchmark.jsonl` with a sample of your own queries and look at the `oracle` and `all_weak` rows first. If the oracle doesn't save much, stop there. If it does, the router shows how much of that saving you can actually capture.
