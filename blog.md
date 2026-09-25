# Before You Build an LLM Router, Measure the Ceiling

*What mdb-steer, a learned router on Ollama and MongoDB Atlas Vector Search, taught us about where routing savings come from.*

---

LLM routing sounds like an easy win. Most queries don't need your biggest model, so send "What's the capital of Japan?" to a 1B model and save the 8B for Dijkstra and Raft. Cost goes down and quality stays the same.

We built a router to test this, measured it on 150 calibration and 45 held-out queries, and found something more useful than a good-looking benchmark number:

> **How much routing can save is decided before you train a router.** It depends on how much of
> your cost sits in queries the small model can handle, and on how much cheaper the small model
> really is on your hardware or price sheet. Measure the ceiling, a perfect-hindsight router, first.

## The perfect-hindsight router

The **perfect-hindsight router** ("hindsight" for short) is exactly what it sounds like. For every query it picks the cheapest model that still gets the best score. No real router can beat it, so the hindsight router's savings are the most routing can ever achieve on a workload.

It's cheap to compute. Run both models on a sample of your traffic, grade the answers, and for each query take the cheaper model whenever it did as well as the expensive one.

On 45 held-out queries spanning facts, rewrites, code, math, reasoning and long "easy but wordy" tasks, with `llama3.1:8b` as the strong model and `llama3.2:1b` as the weak one:

| strategy | quality | cost | sent to 1B |
|---|---|---|---|
| all-strong | 0.878 | $0.346 | 0% |
| all-weak | 0.611 | $0.218 | 100% |
| **hindsight** | **0.889** | **$0.284** | **64%** |

The hindsight router sends almost two-thirds of the traffic to the 1B model and quality goes *up* slightly, because the small model is sometimes the one that gets it right.

**It saves only 17.8%.**

## Two factors set the ceiling

Break the hindsight router down by query type and both factors show up:

| category | 1B was good enough | hindsight saves |
|---|---|---|
| factual | 5 / 7 | 50.8% |
| long_easy (summaries, rewrites, extraction) | **6 / 6** | 26.1% |
| reasoning | 3 / 8 | 19.2% |
| rewrite | 4 / 6 | 16.5% |
| coding | 8 / 10 | 15.1% |
| math | 3 / 8 | 3.6% |

**Factor 1: where the cost is.** Math and reasoning produce long, expensive answers, and the 1B mostly gets them wrong, so that cost can't move. The queries the 1B handles well tend to be the cheap ones.

**Factor 2: how much cheaper the small model really is.** Look at long_easy. The 1B handled *every* summary, rewrite and extraction task, yet the hindsight router saves only 26%. On this hardware (CPU, in Docker) the 1B is only about 1.35× cheaper per query on those tasks, and sending all traffic to it saves just 36.9%. A model 8× smaller isn't 8× cheaper when your hardware can't take advantage of the difference.

So routing savings ≈ (share of cost in queries the small model can handle) × (the small model's real per-query cost gap). The router can't change either factor. Measure both before you build one.

### What about cloud APIs?

CPU inference is the worst case for factor 2. Hosted APIs charge per token, and small models are often priced far below large ones. Re-pricing the same 45 queries by their measured token counts (output priced at 3× input) shows how the ceiling moves with the price gap:

| strong : weak price per token | hindsight saves (95% CI) | all-weak saves |
|---|---|---|
| 1.35× (≈ our CPU) | 20% (13–29%) | 40% |
| 2× | 33% (23–44%) | 59% |
| 5× | 49% (33–64%) | 84% |
| 10× | 55% (38–70%) | 92% |
| 20× | 57% (40–75%) | 96% |

A wider price gap raises the ceiling quickly, then it levels off around 55–60%. However cheap the small model gets, the roughly one-third of queries that need the strong model set a floor on cost. On cloud pricing routing can be worth much more, but factor 1 still caps it.

## What the router captures

mdb-steer's router sees only the query text. It combines a **neighbour vote** (`$vectorSearch` over graded calibration queries) with a few text features in a class-balanced logistic model, and picks its threshold by cross-validation on the calibration set, never on the benchmark.

On the held-out set:

| threshold | sent to 1B | savings | quality drop | vs. random routing |
|---|---|---|---|---|
| 0.35 | 18% | 5.4% | 3.8% | +0.014 |
| **0.40** | **24%** | **8.6%** | **3.8%** | **+0.032** |
| 0.45 *(chosen by cross-validation)* | 38% | 13.3% | 7.6% | +0.034 |
| 0.50 | 56% | 18.2% | 7.6% | +0.081 |

It beats random routing at every threshold up to 0.65, so it's picking up real signal. At 0.40 it captures about half of the hindsight router's savings within a 5% quality budget. The threshold cross-validation picked (0.45) predicted a 4.5% quality drop and delivered 7.6% on held-out data, which shows how noisy 45 queries still are. Treat these numbers as directional.

**How sure are we?** Resampling the 45 queries 2,000 times (bootstrap) gives 95% intervals:

| at threshold 0.40 | measured | 95% CI |
|---|---|---|
| hindsight saves | 17.8% | 10.0 – 27.5% |
| router saves | 8.6% | 2.9 – 15.5% |
| router quality drop | 3.8% | 0.0 – 10.0% |
| lift over random | +0.032 | −0.01 – +0.08 (positive in 91% of resamples) |

The saving is solid. The quality drop is inside the 5% budget in about 70% of resamples. At the cross-validated 0.45 it's inside in only 29%, so that miss is real and not bad luck. The router's saving under cloud pricing grows more slowly than the ceiling (about 15–17% at a 5–20× price gap), because at 0.40 it is conservative and offloads only 24% of traffic.

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

## What about an off-the-shelf router?

[RouteLLM](https://github.com/lm-sys/RouteLLM) (LMSYS, ICLR 2025) ships routers pre-trained on Chatbot Arena preferences. Its BERT router runs locally, so we scored it on the same 45 stored answers with no new model calls and no API key (`scripts/compare_routellm.py`):

| | mdb-steer | RouteLLM `bert` |
|---|---|---|
| AUC for "weak model fails" | **0.79** | 0.62 |
| best saving within 5% quality budget (CPU) | **8.6%** | 5.6% |
| same, per-token pricing, weak 10× cheaper | **16.1%** | 12.9% |

Two findings. First, a generic router works on day one: RouteLLM beats random routing and passes the guardrail without seeing any of our data. Second, 150 of our own graded queries buy a better ranking and about 3 more points of saving. But both are far below the ceiling (17.8% on CPU, 54.5% at 10×), so the traffic, not the router, still decides the result.

RouteLLM's own paper points the same way: on MMLU and GSM8K, where answers can be checked, its routers gained far less than on open-ended chat (up to 1.4–1.5× vs random, against 3.66× on MT Bench). Our workload is mostly checkable answers too. Correctness is harder to route on than preference.

The full comparison is in [mdb-steer-vs-routellm.md](mdb-steer-vs-routellm.md), and a summary of the paper is in [routellm.md](routellm.md).

## Why everything lives in MongoDB

Every answer, verdict, embedding, routing feature, neighbour list and fitted model is stored in MongoDB: `calibration` (vector-indexed), `telemetry`, `router_models` and `benchmark_runs`. Running the models is the expensive part, so storing everything makes it a one-time cost:

- **Change the judge?** `regrade` re-grades stored answers.
- **Change the label or features?** `fit`, then `sweep --rescore` recomputes the neighbour vote from the stored neighbour lists and replays the held-out set.
- **Change the threshold?** `sweep` traces the full cost/quality curve.
- **Try someone else's router?** `scripts/compare_routellm.py` scored RouteLLM on the stored answers.

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

Replace `data/benchmark.jsonl` with a sample of your own queries and look at the `hindsight` and `all_weak` rows first. If hindsight doesn't save much, stop there. If it does, the router shows how much of that saving you can actually capture.

For the lessons on one page, plus a decision table for "router or not?" and the cheaper levers to try first, see [key-insights.md](key-insights.md). For the one-page version for leadership, see [exec_summary.md](exec_summary.md).
