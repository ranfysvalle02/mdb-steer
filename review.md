# mdb-steer: Project Review

**Score: 8 / 10.** The system is sound, tested and reproducible, and every claim is measured on held-out data. The router beats random routing and captures about half of the achievable savings within a 5% quality budget. What keeps it from 10: the savings ceiling on this hardware is low, 45 benchmark queries still leave wide error bars, and the threshold chosen without the test set missed the guardrail on held-out data.

---

## TL;DR

- **Key finding: two factors set the savings ceiling before any router is trained.** They are the share of cost in queries the small model can handle, and how much cheaper the small model really is on your hardware. On 45 held-out queries the **perfect-hindsight router** sends **64%** of traffic to the 1B model and still saves only **17.8%**.
- **The label mattered more than the features.** Predicting "did strong beat weak?" was at chance (AUC ≈ 0.50). Predicting "did the weak model fail?" gave AUC ≈ 0.65, and took the router from worse than random to better than random at every threshold up to 0.65.
- **Router on held-out data:** at threshold 0.40, **24%** offload, **8.6%** savings, **3.8%** quality drop, **PASS**. That's about half the hindsight router's savings. The threshold chosen by cross-validation (0.45) predicted a 4.5% drop but delivered **7.6%** and **fails** the guardrail. We report that miss.
- 150 calibration and 45 benchmark queries, no overlap between them, 32 tests, CI, and it all runs locally in Docker with no API keys.

## Executive summary

mdb-steer is a learned LLM router. For each query it decides whether `llama3.2:1b` is good enough or `llama3.1:8b` is needed, using only the query text: an Atlas Vector Search neighbour vote over graded calibration queries plus text features, combined by a class-balanced logistic model. The routing threshold is chosen by 5-fold cross-validation on the calibration set, so the benchmark is a true held-out test.

The main result is about the workload rather than the router. Per-category analysis of the perfect-hindsight router shows the savings ceiling is set by where the cost is and by the real cost gap between the models. On CPU, a 1B model is only about 1.35–1.6× cheaper per query than an 8B, so even tasks the 1B always handles (summaries, rewrites, extraction) save only about 26%. Teams should measure the hindsight router and the all-weak cost on their own traffic and hardware before investing in a router.

Within that ceiling the router works, but the effect is modest and noisy at n=45. It beats random routing at every threshold up to 0.65 and captures about half of the hindsight router's savings at 3.8% quality loss. Everything (answers, verdicts, embeddings, neighbour lists, features, fitted models) is stored in MongoDB, so changes to the judge, label, features or threshold are evaluated by replaying stored data rather than re-running the models.

## Results

Setup: `llama3.1:8b` (strong, also the judge), `llama3.2:1b` (weak), `nomic-embed-text` for embeddings. Everything ran on CPU in Docker on a Mac. 150 calibration queries, 45 held-out benchmark queries across factual, rewrite, coding, math, reasoning and long_easy. Scores: 1.0 correct, 0.5 partial, 0.0 incorrect. Cost is Ollama's server-side compute time.

**Baselines (run `run-1790302862`)**

| strategy | quality | cost (USD) | offload to 1B |
|---|---|---|---|
| all_strong | 0.878 | 0.3462 | 0% |
| all_weak | 0.611 | 0.2184 | 100% |
| hindsight | 0.889 | 0.2845 | 64% |

**Perfect-hindsight router by category**

| category | 1B good enough | hindsight saves |
|---|---|---|
| factual | 5 / 7 | 50.8% |
| long_easy | 6 / 6 | 26.1% |
| reasoning | 3 / 8 | 19.2% |
| rewrite | 4 / 6 | 16.5% |
| coding | 8 / 10 | 15.1% |
| math | 3 / 8 | 3.6% |

**Token-priced (cloud API) ceiling.** Same queries re-priced by measured token counts, output at 3× input:

| strong : weak price per token | hindsight saves (95% CI) | all-weak saves | router @ 0.40 saves (95% CI) |
|---|---|---|---|
| 1.35× | 20.1% (12.6–28.5) | 39.6% | 8.5% (3.3–15.9) |
| 2× | 33.0% (22.6–43.5) | 59.2% | 11.3% (4.3–20.2) |
| 5× | 49.1% (32.8–63.9) | 83.7% | 14.9% (6.5–25.6) |
| 10× | 54.5% (37.6–70.4) | 91.8% | 16.1% (6.9–28.6) |
| 20× | 57.2% (40.1–75.3) | 95.9% | 16.6% (6.7–29.5) |

**Router (label `weak_fails`; replayed with `sweep --rescore` from stored features and neighbours)**

| threshold | offload | savings | quality drop | vs. random | guardrail (≤5%) |
|---|---|---|---|---|---|
| 0.30 | 4% | 0.2% | 0.00% | +0.012 | PASS |
| 0.35 | 18% | 5.4% | 3.80% | +0.014 | PASS |
| **0.40** | **24%** | **8.6%** | **3.80%** | **+0.032** | **PASS** |
| 0.45 *(cross-validated)* | 38% | 13.3% | 7.59% | +0.034 | FAIL |
| 0.50 | 56% | 18.2% | 7.59% | +0.081 | FAIL |
| 0.60 | 71% | 24.1% | 18.99% | +0.023 | FAIL |
| 0.70 | 91% | 32.9% | 30.38% | −0.024 | FAIL |

**Bootstrap 95% CIs (2,000 resamples of the 45 queries).** At 0.40: router saves 8.6% (2.9–15.5), quality drop 3.8% (0.0–10.0), within the 5% budget in 70% of resamples, lift over random +0.032 (−0.01 to +0.08, positive in 91%). At 0.45: drop 7.6% (1.2–15.9), within budget in only 29%.

The fit: 150 queries, 69 labelled as needing the strong model. Cross-validated accuracy 64%. Weights on standardised features: `knn_p_strong +0.44`, `multi_step +0.28`, `length −0.09`, `numeric −0.12`. The same benchmark routed under the old `strong_wins` label had a lift over random of **−0.027**.

**Offline signal analysis (leave-one-out on calibration, AUC)**

| signal | predicting "strong beats weak" | predicting "weak fails" |
|---|---|---|
| kNN vote, k=5 | 0.514 | 0.634 |
| kNN vote, k=3, sharpened | 0.496 | 0.655 |
| category rate (upper bound for a category classifier) | 0.524 | 0.656 |
| query length | 0.559 | 0.556 |
| multi-step wording | 0.518 | 0.544 |

## Key insights

1. **Measure the perfect-hindsight router and the all-weak cost first.** Together they bound what routing can save. Here all-weak saves 36.9% and the hindsight router 17.8%. No router can do better.
2. **Model size isn't cost.** On CPU, the 1B model is only about 1.35–1.6× cheaper per query than the 8B. The long_easy category, where the 1B was always good enough, still saves only 26%. On a GPU the gap, and the ceiling, would likely be larger.
3. **Predict the weak model's failure, not the strong model's win.** A difference of two noisy verdicts is close to unlearnable. A single verdict isn't. This one change took the router from worse than random to better than random.
4. **Small calibration sets create false signals.** At 30 queries, text features looked predictive and the router appeared to beat random. At 150, those features were at chance. Scale data before trusting features.
5. **Embeddings capture topic, and topic partly predicts failure.** The kNN vote reaches about the same AUC as a per-category failure rate (0.65 vs 0.66). It learns "rewrites and reasoning are risky for the 1B" without being told the categories.
6. **Judges need categorical verdicts.** A 0–10 scale let an 8B judge give 8/10 to wrong answers.
7. **Measure compute, not wall-clock time.** Model swapping inflated the costs until all models were kept loaded and cost came from server-side timings.
8. **Report when the held-out result misses the cross-validated prediction.** The threshold chosen by cross-validation missed the guardrail on held-out data (7.6% drop vs 4.5% predicted). At n=45, a single query moves quality by 1–2 points.

## Caveats

- **Noise.** 45 benchmark queries give wide error bars (see the bootstrap CIs above), and a couple of queries can flip a guardrail result.
- **Judge self-bias.** The strong model grades its own answers. `JUDGE_MODEL` can point at an independent model, and no human audit of the judge has been done.
- **Replayed router scores.** The router rows come from `sweep --rescore` after refitting with the new label. The fit uses only calibration data, and the kNN vote is recomputed from each benchmark query's stored neighbours. A fresh `benchmark` would make the same decisions.
- **Hardware.** Everything ran on CPU. On a GPU the cost gap between the models would likely be much larger, which changes factor 2 of the savings ceiling.

## Scorecard

| Area | Score | Notes |
|---|---|---|
| Honesty and measurement | 10 | Held-out evaluation, threshold chosen by cross-validation, misses reported, every number stored. |
| Architecture and code quality | 9 | Small typed modules, one shared scoring function, retries and resume, replay tooling. |
| Reproducibility | 8 | `docker compose` end to end, no keys. A full run takes several hours on CPU. |
| Tests and CI | 8 | 32 fast tests (judge, features, fitting, evaluation, router, client retries, data integrity). Lint and tests run in CI. No integration test against live containers. |
| Router effectiveness | 6 | Beats random robustly and captures about half the ceiling, but the cross-validated threshold missed on held-out data. |
| Evidence strength | 6 | 150/45 split, but error bars at n=45 are wide and the judge is unaudited. |
| **Overall** | **8** | |

## Path to 10

1. **A GPU host run.** This is the biggest lever: it tests factor 2 of the ceiling directly.
2. **200+ benchmark queries,** with confidence intervals (bootstrap over queries) on every reported number ([#2](https://github.com/ranfysvalle02/mdb-steer/issues/2)).
3. **Judge audit:** an independent judge model plus about 50 human-graded answers to measure the judge's accuracy.
4. **A cascade baseline:** try the 1B first and escalate on a cheap check. Compare it with prediction-only routing.
5. **An integration test** that runs a tiny calibrate → fit → benchmark against the Docker stack in CI.
