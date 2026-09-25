# mdb-steer vs RouteLLM

*How mdb-steer and RouteLLM compare: what each one answers, how each works, pros, cons, when to use which, and how
they fit together. Background on RouteLLM is in [routellm.md](routellm.md). mdb-steer's own results
are in [key-insights.md](key-insights.md) and [review.md](review.md).*

---

## TL;DR

**They answer different questions, so they work well together.**

- **RouteLLM** answers *"which model should this query go to?"* using a router trained on public
  human-preference data. It works on day one, before you have any data of your own.
- **mdb-steer** answers *"is routing worth it on **my** traffic, and how much of the possible saving does
  my router capture?"* It measures a perfect-hindsight savings ceiling on your own graded traffic, in your
  real cost units, then fits a router to that same data in MongoDB.

RouteLLM's paper never measures the ceiling. mdb-steer's main finding is that the ceiling decides the
outcome. On our workload a perfect router saved only 18% on CPU (49–57% at cloud price gaps). The
sensible setup: **use mdb-steer to decide whether to route and to judge routers; treat RouteLLM as one
of the routers being judged.**

---

## 1. Side by side

| | **RouteLLM** | **mdb-steer** |
|---|---|---|
| Main question | Which model for this query? | How much can routing save here, and does my router capture it? |
| Training signal | Chatbot Arena human preferences + benchmark answers + LLM-judge labels | Your own queries, both models' answers, graded by a judge against a reference |
| Label | "strong model preferred" (`P(win_strong \| q)`) | "weak model fails" (`weak_score < 1.0`); `strong_wins` also available |
| Router | `mf` / `sw_ranking` / `bert` / `causal_llm` | kNN vote over Atlas Vector Search + text difficulty features → logistic model |
| Cost unit | share of strong-model calls | measured compute time or tokens × your price |
| Threshold | calibrated to a target strong-call share | 5-fold cross-validated for maximum saving within a quality guardrail |
| Headline metric | PGR / CPT / APGR | savings vs `all_strong`, quality drop, **lift over random**, **hindsight ceiling** |
| Guardrail | none built in | fails with a non-zero exit code if the quality drop exceeds `QUALITY_GUARDRAIL_PCT` (CI-friendly) |
| Data store | none (stateless router) | MongoDB: calibration, telemetry, router fits, benchmark runs |
| Works day one | yes, pre-trained | no, needs a calibration run (~150 queries) |
| Model pair | trained on GPT-4 vs Mixtral; carried over to Claude 3 Opus/Sonnet and Llama 3.1 70B/8B without retraining (paper Table 4) | whatever you configure; the router is fitted to that exact pair |
| Keys / cloud | `mf` and `sw_ranking` need `OPENAI_API_KEY` for embeddings | none: Ollama + Atlas Local in Docker |
| Serving | OpenAI-compatible server / client drop-in | CLI `route`; no OpenAI-compatible server |
| Scale of evidence | large public benchmarks | 150 calibration + 45 held-out queries (CIs wide; [#2](https://github.com/ranfysvalle02/mdb-steer/issues/2)) |

## 2. How each one works

### RouteLLM

```
query ─▶ router (mf | sw_ranking | bert | causal_llm) ─▶ P(win_strong | q)
                                                              │
                                  ≥ α ─────────────┬─────────── < α
                                   ▼                           ▼
                             strong model                 weak model   (via LiteLLM)
```

- The weights are learned **once, offline, on preference data from other people's traffic**.
- `α` is picked so that a target share of queries goes to the strong model.
- A query is scored by how much a human (or GPT-4 judge) would *prefer* the strong model's answer.

### mdb-steer

```
calibrate:  queries ─▶ both models answer ─▶ judge vs reference ─▶ MongoDB (+ embeddings, Vector Search index)
fit:        leave-one-out features ─▶ class-balanced logistic ─▶ CV threshold within guardrail
route:      query ─▶ embed ─▶ $vectorSearch k-NN "weak failed?" vote + text features ─▶ P(strong needed)
benchmark:  held-out queries ─▶ router + all_strong + all_weak + random + hindsight ─▶ guardrail verdict
```

- The weights are learned **from your traffic, graded for correctness**, in the costs you actually pay.
- The threshold is chosen from out-of-fold predictions, never on the benchmark set.
- Every answer, grade, embedding and decision is stored, so re-scoring (new label, new prices, new
  router) needs **no model calls**.

### The closest match: `sw_ranking` vs mdb-steer's kNN vote

Both look at labelled examples that are *semantically close* to the query:

| | `sw_ranking` | mdb-steer kNN |
|---|---|---|
| Neighbours | every Arena battle, weighted by similarity | top-k (default 5) from Atlas `$vectorSearch` |
| Aggregation | weighted Bradley–Terry fit | similarity-weighted share of "weak failed" |
| Used as | the final score | one feature in a logistic model, alongside difficulty features |
| Data | public battles | your graded calibration set |

mdb-steer can be read as `sw_ranking` run on your own data, with a database-backed index, a label
you can actually learn, and a combiner on top.

## 3. Pros and cons

### RouteLLM

**Pros**
- Works immediately, with no calibration data or grading pipeline.
- Trained on a large, varied dataset, so it handles broad chat traffic well.
- Adopting it is almost free: change the model name in your OpenAI client.
- Well-studied metrics (APGR) and published benchmarks.
- Works across model pairs, so a model upgrade doesn't force retraining.

**Cons**
- **Doesn't know your traffic.** Arena-trained weights may not fit specialised workloads, and calibration only moves the threshold.
- **The cost model is the number of strong-model calls.** mdb-steer's data shows why that misleads: routing 64% of
  requests saved only 18% of cost, because the queries a small model can handle have short, cheap answers.
- **A preference label is not a correctness label.** A preferred answer may just be longer or more polished.
- **Requires OpenAI** for the recommended router, which means an external API call per request and data leaving your environment.
- **No ceiling, no guardrail, no "compare to random" check** on your data out of the box.

### mdb-steer

**Pros**
- **Measures the ceiling first**, so it can tell you not to build a router at all.
- Cost measured in real units (compute time, tokens × price), so savings are money saved, not calls avoided.
- Correctness labels from a reference-based judge, and a label that can actually be learned (`weak_fails`; AUC ≈ 0.65 vs ≈ 0.50 for the win label).
- An honest evaluation built in: held-out set, random baseline at the same offload, CV threshold, and a guardrail that can gate CI.
- Fully local with no keys. MongoDB keeps a history you can replay.
- Fitted to your exact model pair and traffic.

**Cons**
- **Needs data first:** you run both models on about 150+ queries, with reference answers, before it routes.
- A small evaluation (45 held-out queries) gives wide confidence intervals.
- A simple router (kNN + 3 text features + logistic) with modest discrimination.
- By default the judge is the strong model grading itself, which biases grades toward the strong model.
- No OpenAI-compatible serving layer, so it isn't a drop-in for production traffic.
- Only two models.

## 4. Considerations

1. **Count the cost you actually pay.** Before trusting any router's savings (including RouteLLM's), convert its
   decisions into tokens or seconds × your price. Counting calls overstates savings whenever easy queries
   have short answers.
2. **Check the ceiling before comparing routers.** If hindsight saves under ~15%, the gap between RouteLLM and
   mdb-steer doesn't matter. See the decision table in [key-insights.md](key-insights.md#then-decide).
3. **Your traffic vs Arena traffic.** The more your queries look like general chat, the better RouteLLM's
   pre-trained weights will do. For narrow domains, expect a router trained on your own data to do better.
4. **Compare labels as well as routers.** Preferences (RouteLLM) and correctness (mdb-steer) can disagree. When
   quality means "gets the right answer", use correctness grades to evaluate both.
5. **Privacy and keys.** `mf` and `sw_ranking` send every query to OpenAI for embeddings. Use `bert` (local) if
   that's unacceptable, or if you want to keep mdb-steer's "no API keys" setup.
6. **Thresholds don't carry over.** RouteLLM's example `0.11593` was calibrated on Arena data. On your traffic,
   sweep `α` and pick it the way mdb-steer does: on held-out data, within a quality guardrail.
7. **Latency.** RouteLLM `mf` adds a remote embedding call; `bert` adds a local forward pass; mdb-steer adds a
   local embedding plus a `$vectorSearch` query. All are small next to generation time, but measure them.
8. **Drift.** RouteLLM's weights are fixed. mdb-steer's telemetry can be fed back into calibration. With either one,
   keep watching how often the weak model fails on live traffic.

## 5. When to use which

| situation | use |
|---|---|
| No data yet, broad chat-like traffic, want something running today | **RouteLLM** (`mf`, or `bert` if you can't send data to OpenAI) |
| Need to decide whether routing is worth building | **mdb-steer** (`calibrate` → `benchmark` → read `hindsight`) |
| Specialised domain with a clear definition of correct | **mdb-steer**, or RouteLLM retrained on your labels |
| Need a CI gate on quality | **mdb-steer** guardrail |
| Need an OpenAI-compatible routing proxy | **RouteLLM** server (optionally wrapping a router trained with mdb-steer) |
| Want to know which router is better on *your* traffic | `scripts/compare_routellm.py` (see §6) |

## 6. How to combine them: the measured comparison

[`scripts/compare_routellm.py`](scripts/compare_routellm.py) scores RouteLLM's pre-trained routers on a
stored benchmark run and compares them with mdb-steer on **the same graded, priced answers**. It makes **no LLM
calls and needs no API key**:

```bash
pip install -e ".[routellm]"
python scripts/compare_routellm.py --ratios 5,10,20     # bert: local, keyless
python scripts/compare_routellm.py --routers bert,mf    # mf: needs OpenAI or Azure OpenAI embeddings
```

It loads RouteLLM's Hugging Face checkpoints directly instead of importing `routellm`, because that
package creates an `OpenAI()` client at import time and fails without a key, even for the local BERT
router. The scoring logic is RouteLLM's own. Scores are cached in the `routellm_scores` collection.

### Results (45 held-out queries, `llama3.1:8b` vs `llama3.2:1b`)

| | mdb-steer | RouteLLM `bert` |
|---|---|---|
| AUC for "weak model fails" (threshold-free) | **0.787** | 0.617 |
| best point within 5% guardrail, CPU cost | 24% offload, **8.6%** saving, 3.8% drop | 24% offload, 5.6% saving, 3.8% drop |
| same, per-token pricing, weak 10× cheaper | **16.1%** saving | 12.9% saving |
| thresholds beating random at the same offload | 81 / 92 | 74 / 99 |
| hindsight ceiling (CPU / 10×) | 17.8% / 54.5% | 17.8% / 54.5% |

What this shows:
- **RouteLLM `bert` works on day one, with no calibration data:** it beats random routing and passes the guardrail.
- **mdb-steer, fitted on 150 of your own graded queries, ranks better** (AUC 0.79 vs 0.62) and saves
  about 3 points more at the same offload and quality. At equal offload, it sends away queries that cost more.
- **Both are far below the ceiling.** Neither captures more than a third of the hindsight saving. The ceiling
  and the share of hard queries still decide the outcome more than the choice of router.
- Caveat: with 45 queries, one query moves offload by 2.2 points. The two routers offload different
  queries (only 5 of 11 overlap), and their equal quality drop is a coincidence of the discrete grades.
  Treat the gap as suggestive until [#2](https://github.com/ranfysvalle02/mdb-steer/issues/2) grows the benchmark.

Other ways to combine them: use RouteLLM's score as **another feature** in mdb-steer's logistic model, or
serve mdb-steer's router behind RouteLLM's server by implementing `calculate_strong_win_rate` on top of `Router.route`.

---

## Appendix

### A. Glossary

| term | meaning |
|---|---|
| strong / weak model | the expensive, better model / the cheap, weaker one |
| `α`, threshold | cut-off on the router's score; at or above it → strong |
| offload | share of queries sent to the weak model |
| hindsight (oracle) | a router that already knows both grades and picks the cheapest model with the best score; the savings ceiling |
| random | routes the same share of queries at random; the baseline a real router must beat |
| PGR | share of the weak-to-strong quality gap recovered |
| CPT(x%) | smallest share of strong-model calls that reaches PGR x% |
| APGR | area under PGR vs share of strong-model calls |
| guardrail | largest quality drop allowed vs all-strong (mdb-steer default 5%) |

### B. Formulas

```
RouteLLM
  route(q)  = strong if P(win_strong | q) ≥ α else weak
  PGR       = (r_router − r_weak) / (r_strong − r_weak)
  APGR      ≈ (1/10) Σ_i PGR at strong-call share c_i     (10 buckets over [0%, 100%]; paper Eq. 8)

mdb-steer
  label     = weak_score < 1.0                                  (weak_fails)
  knn       = Σ sim_i · label_i / Σ sim_i                       over top-k $vectorSearch neighbours
  p_strong  = σ(w · standardise([knn, multi_step, length, numeric]) + b)
  savings   = 1 − cost(router) / cost(all_strong)
  hindsight = Σ_q  cost(weak_q) if score(weak_q) ≥ score(strong_q) else cost(strong_q)
```

### C. mdb-steer results for reference

Taken from [review.md](review.md) (`llama3.1:8b` vs `llama3.2:1b`, CPU, 150 calibration / 45 held-out queries):

| | value |
|---|---|
| hindsight offload / saving (CPU) | 64% / 17.8% |
| all-weak saving (CPU) | 36.9% |
| hindsight saving at 5–20× cloud price gap | ~49–57% |
| router @ 0.40 | 24% offload, 8.6% saving, 3.8% quality drop (passes 5%) |
| `weak_fails` AUC vs `strong_wins` AUC | ≈ 0.65 vs ≈ 0.50 |

The gap between 64% offload and 17.8% saving is the clearest example of why counting strong-model calls
overstates savings.

### D. How the comparison script works

For each router, `p_strong` is attached to every stored benchmark row. The script sweeps thresholds over a
fixed grid plus every observed score (RouteLLM scores cluster in about 0.2 to 0.8), and for each threshold replays
the rows through mdb-steer's own `summarize` (cost, quality drop, lift over random, guardrail). It
then reports the best passing point per router. Token-priced tables reuse `blog_stats.py`'s pricing
(output tokens 3× input, weak model N× cheaper).

- `bert`: `routellm/bert_gpt4_augmented`, 3-way classifier; `P(strong) = 1 − P(tie) − P(weak wins)`.
- `mf`: `routellm/mf_gpt4_augmented`, scoring the GPT-4 vs Mixtral pair from its training vocabulary,
  on `text-embedding-3-small` embeddings from OpenAI (`OPENAI_API_KEY`) or Azure OpenAI
  (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBED_DEPLOYMENT`).

### E. References

- Ong et al. *RouteLLM: Learning to Route LLMs with Preference Data.* [arXiv:2406.18665](https://arxiv.org/abs/2406.18665), 2024.
- RouteLLM code and docs: [github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM),
  [Routing to Local Models](https://github.com/lm-sys/RouteLLM/blob/main/examples/routing_to_local_models.md)
- mdb-steer: [README](README.md), [key-insights.md](key-insights.md), [review.md](review.md), [blog.md](blog.md)
