# We Built an LLM Router, Measured It Honestly, and Learned Where the Savings Actually Are

*mdb-steer: learned model routing with Ollama and MongoDB Atlas Vector Search, running entirely on a laptop.*

---

The pitch for LLM routing is simple. Most queries don't need your biggest model. Send "What's the capital of France?" to a 1B model and save the 8B (or 70B) for concurrency bugs and multi-step proofs. Cost goes down and quality stays the same.

Our first version of mdb-steer made exactly that claim: **36.7% cost reduction with 99% quality retention.** It was also not real. The router was given each query's difficulty label. The quality score was a constant. When a model call failed, the script returned the reference answer and graded it as correct. The numbers only reflected what the script assumed.

So we rebuilt it with one rule: **every number has to be measured.** This post covers what we built, what broke, and what the real numbers say.

## The design

Everything runs locally with `docker compose`. MongoDB Atlas Local provides the database and Vector Search, and Ollama serves three models: `llama3.1:8b` (strong), `llama3.2:1b` (weak) and `nomic-embed-text` (embeddings). No API keys.

**Calibrate.** Run *both* models on a set of reference queries. An LLM judge grades each answer against a reference. Store the query's embedding and both scores in MongoDB, with a vector index on the embeddings.

**Route.** For a new query, compute a few signals:

- **Neighbour vote:** retrieve the k most similar calibration queries with `$vectorSearch` and take the similarity-weighted share where the strong model clearly won.
- **Difficulty wording:** "explain", "implement", "probability", "how many", "at least"...
- **Length** and **whether the query contains numbers**.
- **Weak-model self-confidence:** ask the 1B model how sure it is that it can answer.

A small logistic regression, fitted on the calibration set, turns these into P(strong model wins). If that's above a threshold, the query goes to the strong model.

**Benchmark.** On held-out queries, run *both* models on every query, so the router can be compared against every alternative:

- **all-strong:** the quality ceiling
- **all-weak:** the cost floor
- **random routing at the same offload rate:** the bar to beat. If you can't beat random, you haven't learned anything.
- **oracle:** always the cheapest model that gets the best score. The most routing could ever save.

Every decision, feature value, answer and grade lands in MongoDB. That turned out to matter more than we expected.

## What broke, and what it taught us

### 1. The judge was too lenient

The first calibration run looked strange: almost every answer scored 0.8 or 0.9, whichever model wrote it. Reading the stored answers showed why. The 8B judge had given **0.8 out of 1.0** to this answer for the number of arrangements of "BANANA":

> The number of arrangements = 5! / (2!·2!) = 120/4 = **30**

The correct answer is 60. The judge also gave 0.8 to reasoning that concluded Carol was the *tallest* in a question asking who was second shortest.

Small models are poor at 0–10 scales. We switched to a categorical verdict (CORRECT, PARTIAL or INCORRECT) with explicit rules: any wrong number or wrong conclusion is INCORRECT. Because the answers were stored, a new `regrade` command re-judged them in minutes without regenerating anything. The number of calibration queries where the strong model clearly won went **from 2 to 10**. Before that change, the router had almost nothing to learn from.

### 2. We were measuring model loading, not inference

In the first benchmark, the 1B model averaged 29 seconds per answer and the 8B 39 seconds. That ratio makes no sense for models that differ 8× in size.

The Ollama logs showed **28 model loads** in one run. The strong, weak and judge models kept pushing each other out of memory, so wall-clock latency mostly measured loading time. Two fixes:

- `OLLAMA_MAX_LOADED_MODELS=3` and `OLLAMA_KEEP_ALIVE=-1` keep all three models resident.
- We switched to Ollama's native API, which reports `load_duration` separately from `prompt_eval_duration` and `eval_duration`. Cost is now priced only on real compute time.

The measured cost gap between all-weak and all-strong went **from 26% to 45%**.

### 3. The model learned to always say "weak"

Our first logistic fit reported 67% training accuracy, which sounds fine until you notice that 20 of the 30 labels were "weak is good enough". The model had learned to predict the majority class for every query, and routed 100% of benchmark traffic to the 1B model.

Class balancing and standardised features fixed it: training accuracy rose to 80% against the 67% baseline. Because every benchmark row's features were already stored, `sweep --rescore` could evaluate the new fit on the held-out set **without any new model calls**.

## The real numbers

15 held-out queries, CPU-only, scores out of 1.0:

| strategy | quality | cost | offload to 1B |
|---|---|---|---|
| all-strong | 0.900 | $0.169 | 0% |
| all-weak | 0.567 | $0.092 | 100% |
| **router @ 0.3** | **0.867** | **$0.161** | **33%** |
| random @ 33% | 0.789 | | 33% |
| oracle | 0.900 | $0.144 | 47% |

At a threshold of 0.3, the router sends a third of traffic to the 1B model, stays inside the 5% quality guardrail (a 3.7% drop), and beats random routing by 0.078. It beats random at every threshold from 0.1 to 0.6.

The saving at that setting is **4.9%**.

## The insight: measure the oracle first

The router isn't the main limit. Look at the oracle: **perfect routing would save only 14.7%** on this workload.

That's because the queries a 1B model *can* handle, like "capital of Japan?" or "translate 'thank you'", produce short answers that are cheap on any model. The expensive queries are the ones that produce 500-token implementations of Dijkstra or explanations of Raft, and those are exactly the ones that need the strong model. On this workload, cost and difficulty go together.

**Before building a router, run both models on a sample of your real traffic and compute the oracle.** If the oracle saves 15%, no router will save you 40%. If your traffic has expensive easy queries (summaries, drafts, long rewrites), the ceiling rises and routing becomes worth it.

Other things the data showed:

- **Difficulty wording beat semantic similarity at this data size.** Multi-step, length and numeric features got positive weights. The kNN vote came out *negative*, because with 30 calibration points a query's nearest neighbours are often off-topic. It should improve with more data ([#1](https://github.com/ranfysvalle02/mdb-steer/issues/1)).
- **A 1B model can't judge its own competence.** Its self-rated confidence carried no useful signal and added about half a second per query.
- **Some failures can't be predicted from the text.** The 1B model got a grammar fix and a basic factual question wrong. No text-only router will catch those.

## Why MongoDB made this workable

We changed the judge, the latency measurement and the fitting procedure *after* data had been collected. We never had to throw away hours of CPU inference, because everything was stored:

- `calibration`: embeddings (vector-indexed), both models' answers, verdicts and judge reasons. We changed the judge by re-grading stored answers.
- `telemetry`: every routing decision with its features, neighbours and both attempts. We changed the model by replaying stored features.
- `router_models`: every fit, with weights, scaling and training rows.
- `benchmark_runs`: every run's summary with a settings snapshot.

Vector search, operational data and the ML artefacts all live in one database. That made fast iteration cheap enough to find three measurement bugs in one afternoon.

## Try it

```bash
git clone https://github.com/ranfysvalle02/mdb-steer && cd mdb-steer
docker compose up -d mongodb ollama
docker compose run --rm app calibrate
docker compose run --rm app benchmark
docker compose run --rm app sweep --rescore
```

Then point it at your own traffic and check the oracle first.
