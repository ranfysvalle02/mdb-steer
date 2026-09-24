# mdb-steer: Project Review

**Score: 7 / 10.** The engineering is sound and every result can be reproduced. The router works, and the oracle analysis shows the savings ceiling on this workload is low. The main barrier to 9+ is data volume and a workload with more room to save ([#1](https://github.com/ranfysvalle02/mdb-steer/issues/1)), not code.

---

## TL;DR

- **The key finding:** on this workload even an **oracle** (a perfect router) saves only **14.7%**, while sending 47% of traffic to the 1B model. The queries a small model can handle are also the cheap ones, so traffic mix caps the savings more than router quality does.
- The fitted router **beats random routing at every threshold from 0.1 to 0.6**. At threshold 0.3 it **passes the 5% quality guardrail** while sending **33%** of traffic to the 1B model, for a **4.9%** saving, about a third of the oracle's.
- Every number is measured: judge-graded quality, cost from server-side compute time, results stored in MongoDB, and a guardrail that sets the exit code. It all runs locally in Docker with no API keys.

## Executive summary

mdb-steer is a learned LLM router. For each query it decides whether `llama3.2:1b` is good enough or `llama3.1:8b` is worth the cost, using only the query text: an Atlas Vector Search neighbour vote over graded calibration queries, text difficulty features, and the weak model's self-confidence, combined by a class-balanced logistic model.

The benchmark compares the router against all-strong, all-weak, random routing at the same offload rate, and an oracle. That comparison produced the project's main finding. The oracle shows the most routing can save here is about 15%, because the queries the 1B model gets right have short, cheap answers. The router captures part of that and beats random routing. The small dataset (30 calibration, 15 benchmark) is the main limit on how far the evidence goes.

Everything (answers, verdicts, embeddings, features, fitted models, run summaries) is stored in MongoDB. Changing the judge, the router or the threshold replays stored data instead of calling the models again, so iteration takes minutes even on CPU.

## Results

Setup: `llama3.1:8b` (strong, also the judge), `llama3.2:1b` (weak), `nomic-embed-text` for embeddings. Everything ran on CPU in Docker on a Mac. 30 calibration queries, 15 held-out benchmark queries. Scores: 1.0 correct, 0.5 partial, 0.0 incorrect.

**Baselines (run `run-1790288468`)**

| strategy | quality | cost (USD) | offload to 1B |
|---|---|---|---|
| all_strong | 0.900 | 0.1691 | 0% |
| all_weak | 0.567 | 0.0922 | 100% |
| oracle | 0.900 | 0.1443 | 47% |

**Router (threshold sweep; router cost includes the self-confidence call)**

| threshold | offload | cost savings | quality drop | vs. random | guardrail (≤5%) |
|---|---|---|---|---|---|
| 0.10 | 13% | −0.8% | 0.00% | +0.044 | PASS |
| 0.20 | 20% | 1.1% | 0.00% | +0.067 | PASS |
| **0.30** | **33%** | **4.9%** | **3.70%** | **+0.078** | **PASS** |
| 0.40 | 40% | 11.2% | 7.41% | +0.067 | FAIL |
| 0.50 | 53% | 15.9% | 14.81% | +0.044 | FAIL |
| 0.60 | 80% | 34.1% | 29.63% | +0.000 | FAIL |

Fitted weights (standardised features): `knn_p_strong −1.83`, `multi_step +0.88`, `length +0.86`, `numeric +0.52`, `weak_uncertainty −0.41`. Training accuracy 80%, against a 67% majority-class baseline.

## Key insights

1. **The oracle ceiling matters more than router accuracy.** Perfect routing saves only 14.7% here, because the queries the 1B model can answer (facts, short rewrites) produce short, cheap answers. Most of the cost is in long answers to hard queries, and those need the strong model anyway. Before building a router, measure the oracle: it tells you the most routing can ever save on your traffic.
2. **Small judges need categorical verdicts.** On a 0–10 scale an 8B judge gave 8/10 to `BANANA → 30` (the correct answer is 60), and scores bunched together. With CORRECT, PARTIAL or INCORRECT verdicts, the strong model clearly wins 10 of 30 calibration queries, which gives the router something to learn from.
3. **Price compute, not wall-clock time.** When models share one Ollama instance they push each other out of memory, and wall-clock latency mostly measures loading. With all models kept loaded and cost based on server-reported compute time, all-weak is 45% cheaper than all-strong. On CPU, the 1B model generates only about 1.5× faster (11.3 vs 7.6 tokens/sec).
4. **Difficulty wording beat topic similarity at this data size.** The multi-step, length and numeric features got positive weights. The leave-one-out kNN vote came out *negative*: with 30 points, a query's nearest neighbours are often off-topic. The kNN signal should improve with calibration size.
5. **A 1B model can't judge its own competence.** Its self-rated confidence carries no useful signal (the weight is small and wrong-signed) and adds about 0.5 s of routing overhead. It is a candidate for removal.
6. **Weak-model failures are partly unpredictable.** The 1B model failed a grammar fix (b05) and a factual question (b03) that look trivially easy. No text-only router will catch those. Only calibration coverage or a cheap post-hoc check will.
7. **Class balancing is essential.** Strong wins are the minority class, so an unweighted fit gets 67% accuracy by always predicting "weak" and routes everything to the 1B model.

## Caveats

- **Small sample.** 15 benchmark queries means one query changes quality by about 6.7%. Treat the numbers as directional.
- **Threshold chosen on the test set.** The 0.3 threshold was picked by sweeping the held-out set, which is optimistic. The default in config stays at 0.5. A proper setup needs a separate validation split, which needs more data.
- **Judge self-bias.** The strong model grades its own answers. `JUDGE_MODEL` can point at an independent model.
- **Replayed router scores.** The router rows come from `sweep --rescore` on stored benchmark features. The fit uses only calibration data, so this matches what a fresh `benchmark` with the same model would produce.
- **CPU-only hardware.** On a GPU, the 1B-vs-8B speed gap and therefore the savings would likely be much larger than the 1.5× measured here.

## Scorecard

| Area | Score | Notes |
|---|---|---|
| Honesty and measurement | 9 | Every number is measured and stored in MongoDB. Failures are reported, and the guardrail is enforced. |
| Architecture and code quality | 8 | Small, typed modules with clear boundaries. Replay tooling (`regrade`, `sweep --rescore`) makes iteration cheap. |
| Reproducibility | 8 | `docker compose` end to end, no keys. On CPU a full run takes hours. |
| Router effectiveness | 5 | Beats random and passes the guardrail at 0.3, but the savings are small. |
| Evidence strength | 5 | 30 + 15 queries, and the threshold was tuned on the test set. |
| Tests | 2 | Deliberately deferred. |
| **Overall** | **7** | |

## Path to 9+

1. **Data ([#1](https://github.com/ranfysvalle02/mdb-steer/issues/1)).** 150+ calibration queries plus a separate validation split for choosing the threshold.
2. **Try traffic where routing has more room to save.** Mixed workloads where easy queries also produce long answers, e.g. summarisation and drafting, raise the oracle ceiling.
3. **Drop or replace self-confidence.** Try the weak model's token log-probabilities, or a cheap draft-then-verify cascade.
4. **An independent judge** and a small human-graded audit sample to measure the judge's accuracy.
5. **Tests** for the pure pieces: `summarize`, `fit_logistic`, `judge._parse`, `text_features`.
6. **Benchmark on a GPU host.**
