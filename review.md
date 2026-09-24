# mdb-steer: Project Review

**Score: 7 / 10.** The engineering is sound and every result can be reproduced. The router works, but the win is small. The main barrier to 9+ is data volume ([#1](https://github.com/ranfysvalle02/mdb-steer/issues/1)), not code.

---

## TL;DR

- mdb-steer went from a **mock demo** (hardcoded quality scores, the router given the answer, a guardrail that always printed `PASS`) to a **real, measured router**. It runs locally in Docker with no API keys.
- On 15 held-out queries, the fitted router **beats random routing at every threshold from 0.1 to 0.6**. At threshold 0.3 it **passes the 5% quality guardrail** while sending **33%** of traffic to the 1B model.
- The savings are small: **4.9%** at that setting. The **oracle**, which is perfect routing, would save only **14.7%**. Routing value is capped by *where the cost is*, not only by how good the router is.
- Three measurement bugs were found and fixed by looking at real data: a lenient judge, model-swap latency, and a class-imbalanced fit. Each one made the results look different from reality.

## Executive summary

LLM routing promises to cut inference cost by sending easy queries to a small model. The first version of this project claimed a 36.7% cost cut with 99% quality retention, but those numbers came from the script's own assumptions: the router was given each query's difficulty label, the Factuality score was a constant, and failed calls returned the reference answer.

The rebuild replaces every simulated piece with a measured one:

| Concern | Before | Now |
|---|---|---|
| Routing input | hand-labelled difficulty | query text only |
| Router | hand-tuned logit | Atlas Vector Search kNN + text features + weak-model self-confidence, combined by a fitted logistic model |
| Quality | hardcoded 0.98 / 0.65 | LLM judge, categorical verdict against a reference |
| Cost | invented tokens × invented prices | Ollama server-side compute time × GPU $/hr, excluding model load |
| Evaluation | same 10 queries, no baseline | held-out set; compared against all-strong, all-weak, random and oracle |
| Guardrail | always `PASS` | enforced; non-zero exit on failure |

On this dataset the router produces a real but modest gain. It is a solid foundation to scale up, but it doesn't yet show large savings.

## Results

Setup: `llama3.1:8b` (strong, also the judge), `llama3.2:1b` (weak), `nomic-embed-text` for embeddings. Everything ran on CPU in Docker on a Mac. 30 calibration queries, 15 held-out benchmark queries. Scores: 1.0 correct, 0.5 partial, 0.0 incorrect.

**Baselines (run `run-1790288468`)**

| strategy | quality | cost (USD) | offload to 1B |
|---|---|---|---|
| all_strong | 0.900 | 0.1691 | 0% |
| all_weak | 0.567 | 0.0922 | 100% |
| oracle | 0.900 | 0.1443 | 47% |

**Router, fitted logistic model (sweep replayed from stored telemetry; the router cost includes the self-confidence call)**

| threshold | offload | cost savings | quality drop | vs. random | guardrail (≤5%) |
|---|---|---|---|---|---|
| 0.10 | 13% | −0.8% | 0.00% | +0.044 | PASS |
| 0.20 | 20% | 1.1% | 0.00% | +0.067 | PASS |
| **0.30** | **33%** | **4.9%** | **3.70%** | **+0.078** | **PASS** |
| 0.40 | 40% | 11.2% | 7.41% | +0.067 | FAIL |
| 0.50 | 53% | 15.9% | 14.81% | +0.044 | FAIL |
| 0.60 | 80% | 34.1% | 29.63% | +0.000 | FAIL |

Fitted weights (standardised features): `knn_p_strong −1.83`, `multi_step +0.88`, `length +0.86`, `numeric +0.52`, `weak_uncertainty −0.41`. Training accuracy 80%, against a 67% majority-class baseline.

**Earlier runs, kept for the record**
- v1 (kNN only, 0–10 judge): quality drop 33% at threshold 0.5, lift over random −0.011. Failed.
- v2 (logistic without class balancing): predicted "weak" for every query and was identical to all-weak. Failed.

## Key insights

1. **The oracle ceiling matters more than router accuracy.** Perfect routing saves only 14.7% here, because the queries the 1B model can answer (facts, short rewrites) produce short, cheap answers. Most of the cost is in long answers to hard queries, and those need the strong model anyway. Before building a router, measure the oracle: it tells you the most routing can ever save on your traffic.
2. **Small judges are too lenient on a 0–10 scale.** An 8B judge gave 0.8/1.0 to `BANANA → 30` (the correct answer is 60). Switching to a categorical verdict (CORRECT, PARTIAL or INCORRECT) moved the count of "strong clearly wins" from 2 to 10 out of 30 calibration queries. Without that, the router had nothing to learn from.
3. **Model swapping distorted the cost numbers.** Without `OLLAMA_MAX_LOADED_MODELS`, Ollama reloaded models 28 times during one run, and wall-clock latency mostly measured loading. The measured all-weak cost advantage went from 26% to 45% once cost used server-reported compute time instead of wall-clock.
4. **Difficulty wording beat topic similarity at this data size.** The multi-step, length and numeric features got positive weights. The leave-one-out kNN vote came out *negative*: with 30 points, a query's nearest neighbours are often off-topic. The kNN signal should improve with calibration size.
5. **A 1B model can't judge its own competence.** Its self-rated confidence carried no signal (weight near zero on the first fit, wrong-signed on the second), and it adds about 0.5 s of routing overhead. It is a candidate for removal.
6. **Weak-model failures are partly unpredictable.** The 1B model failed a grammar fix (b05) and a factual question (b03) that look trivially easy. No text-only router will catch those. Only calibration coverage or a cheap post-hoc check will.
7. **Class balancing is essential.** Strong wins are the minority class, so an unweighted fit gets 67% accuracy by always predicting "weak". That's what the v2 run did.

## Caveats

- **Small sample.** 15 benchmark queries means one query changes quality by about 6.7%. Treat the numbers as directional.
- **Threshold chosen on the test set.** The 0.3 threshold was picked by sweeping the held-out set, which is optimistic. The default in config stays at 0.5. A proper setup needs a separate validation split, which needs more data.
- **Judge self-bias.** The strong model grades its own answers. `JUDGE_MODEL` can point at an independent model.
- **Test-set replay.** The logistic model was refit after v2 ran, then replayed with `sweep --rescore` on v2's stored features. Since the fit uses only calibration data, this is legitimate. The weights shown are exactly what a fresh `benchmark` would use.
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
