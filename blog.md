# Before You Build an LLM Router, Measure the Oracle

*What mdb-steer, a learned router on Ollama and MongoDB Atlas Vector Search, taught us about where routing savings come from.*

---

LLM routing sounds like an easy win. Most queries don't need your biggest model, so send "What's the capital of Japan?" to a 1B model and save the 8B for Dijkstra and Raft. Cost goes down and quality stays the same.

We built a router to test this, measured it properly, and found something more useful than a good-looking benchmark number:

> **How much routing can save is decided by your traffic, not your router.**
> Measure the oracle before you build anything.

## The oracle

The **oracle** is a router with perfect hindsight. For every query it picks the cheapest model that still gets the best score. No real router can beat it, so the oracle's savings are the most routing can ever achieve on a workload.

You can compute it cheaply. Run both models on a sample of your traffic, grade the answers, and for each query take the cheaper model whenever it scored as well as the expensive one.

Here's what we measured on 15 held-out queries spanning facts, rewrites, code, math and reasoning, with `llama3.1:8b` as the strong model and `llama3.2:1b` as the weak one:

| strategy | quality | cost | sent to 1B |
|---|---|---|---|
| all-strong | 0.900 | $0.169 | 0% |
| all-weak | 0.567 | $0.092 | 100% |
| **oracle** | **0.900** | **$0.144** | **47%** |

The oracle keeps full quality while sending almost half the traffic to the 1B model.

**It saves only 14.7%.**

## Why half the traffic is only 15% of the cost

The queries a 1B model gets right, like "capital of Japan?", "translate 'thank you'" or "25% of 80", have **short answers**. They're cheap on *any* model, so moving them saves almost nothing.

The expensive queries produce long answers: a 500-token Dijkstra implementation, an explanation of Raft leader election, a Bloom filter with its failure modes. Those are exactly the queries the 1B model gets wrong. **On this workload, answer length and difficulty go together**: the same queries are both costly and hard. Routing saves the most when easy queries are also expensive, and here they aren't.

That changes the question. It isn't "how good is my router?" but "how much of my cost sits in queries a small model can handle?" If your traffic is mostly short Q&A, the ceiling is low. If it includes expensive easy work, like summarising long documents, drafting emails, rewriting long text or extracting fields from big inputs, the ceiling rises and a router becomes worth building.

## The router: capturing what's available

With the ceiling known, the question is how much of it a real router can get. mdb-steer's router sees only the query text and combines four kinds of signal:

- **Neighbour vote.** `$vectorSearch` finds the most similar calibration queries, where both models were run and graded. It returns the similarity-weighted share where the strong model clearly won.
- **Difficulty wording.** Words like "explain", "implement", "probability", "how many", "at least".
- **Shape.** Query length, and whether it contains numbers.
- **Self-confidence.** The 1B model's rating of whether it can answer.

A class-balanced logistic model fitted on the calibration set combines these into P(strong wins). Sweeping the threshold on held-out queries:

| threshold | sent to 1B | savings | quality drop | vs. random routing |
|---|---|---|---|---|
| 0.2 | 20% | 1.1% | 0.0% | +0.067 |
| **0.3** | **33%** | **4.9%** | **3.7%** | **+0.078** |
| 0.4 | 40% | 11.2% | 7.4% | +0.067 |
| 0.5 | 53% | 15.9% | 14.8% | +0.044 |

The router beats random routing at every threshold from 0.1 to 0.6, so it's picking up real signal. At 0.3 it stays inside a 5% quality guardrail while sending a third of traffic to the 1B model. That captures about a third of the oracle's 14.7%. It's a small result, but it's measured.

## What the signals told us

**Difficulty wording beat semantic similarity.** Multi-step wording, length and numbers all got positive weights. The neighbour vote came out negative: with only 30 calibration queries, a query's nearest neighbours are often on a different topic. Embedding similarity tells you *what a query is about*, not *how hard it is*, and it only helps once there's enough nearby data ([#1](https://github.com/ranfysvalle02/mdb-steer/issues/1)).

**A 1B model can't judge its own competence.** Its self-rated confidence carried no signal and added about half a second per query.

**Some failures can't be predicted from the text.** The 1B model got a basic grammar fix and a simple factual question wrong. No text-only router will catch those. Calibration coverage, or a cheap check after the answer, might.

## Details that make the numbers trustworthy

A few details decide whether routing numbers are real:

- **Use categorical verdicts, not scales.** An 8B judge scoring 0–10 gave 8/10 to an answer of 30 for the arrangements of "BANANA" (the correct answer is 60). Asking for CORRECT, PARTIAL or INCORRECT on the final answer fixed that.
- **Count compute, not wall-clock time.** When several models share one Ollama instance they push each other out of memory, and latency ends up measuring loading. Keep them all loaded (`OLLAMA_MAX_LOADED_MODELS`) and price Ollama's reported generation time, not elapsed time.
- **Balance the classes.** Strong-model wins are the minority. An unweighted fit learns to always answer "weak" and still looks accurate.
- **Compare against random routing at the same offload rate.** Offloading 33% of traffic always saves money. The question is whether it loses less quality than choosing that 33% at random.

## Why everything lives in MongoDB

Every answer, verdict, embedding, routing feature and fitted model is stored in MongoDB: `calibration` (vector-indexed), `telemetry`, `router_models` and `benchmark_runs`. That makes the expensive part, running models, a one-time cost:

- **Change the judge?** `regrade` re-grades stored answers.
- **Change the router?** `fit`, then `sweep --rescore` replays stored features against the held-out set.
- **Change the threshold?** `sweep` traces the full cost/quality curve.

None of these call the models again. On a laptop running CPU inference, that's what makes iterating in minutes practical.

## Try it on your traffic

Everything runs locally in Docker (MongoDB Atlas Local with Vector Search, plus Ollama). No API keys.

```bash
git clone https://github.com/ranfysvalle02/mdb-steer && cd mdb-steer
docker compose up -d mongodb ollama
docker compose run --rm app calibrate
docker compose run --rm app benchmark
docker compose run --rm app sweep --rescore
```

Replace `data/benchmark.jsonl` with a sample of your own queries and look at the `oracle` row first. If the oracle doesn't save much, stop there. If it does, the router shows how much of that saving you can actually capture.
