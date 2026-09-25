# RouteLLM: Learning to Route LLMs with Preference Data

*A reader's summary of the LMSYS paper and framework, written as background for comparing it with
mdb-steer. Numbers and claims are the authors'; see the sources at the end. For the head-to-head,
read [mdb-steer-vs-routellm.md](mdb-steer-vs-routellm.md).*

---

## Abstract

RouteLLM (Ong et al., 2024, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)) treats routing as a
**binary choice between a strong, expensive model and a weak, cheap one**. A router looks only at the
query and estimates the probability that the strong model would give the better answer. If that
probability is above a *cost threshold*, the query goes to the strong model. Otherwise it goes to the weak one.

The routers are trained on **human preference data from Chatbot Arena**: pairs of model answers where
a person picked a winner. The authors add benchmark data with known correct answers, plus
preferences judged by an LLM. They report that the trained routers cut cost by more than 2× on standard
benchmarks without a big quality loss, and that they still work when the model pair is swapped at
test time. The open-source framework ships four pre-trained routers, an OpenAI-compatible server and an
evaluation harness.

---

## 1. Problem formulation

- **Two models.** A strong model `M_strong` (GPT-4 in the paper) and a weak model `M_weak` (Mixtral 8x7B).
- **Win probability.** For a query `q`, the router estimates `P_θ(win_strong | q)`, the probability
  that the strong model's answer is preferred to the weak model's.
- **Decision rule.** Given a threshold `α`:

  ```
  route(q) = M_strong   if P_θ(win_strong | q) ≥ α
             M_weak     otherwise
  ```

  A higher `α` sends more traffic to the weak model, which costs less but may lose quality. The threshold
  is set per request, so each request can pick its own trade-off.

- **Cost** is counted as the **share of calls sent to the strong model**. It is not measured in tokens or seconds.

## 2. Training data

Routers are trained on labelled comparisons of the form (query, strong vs weak, who won).

| source | what it is | role |
|---|---|---|
| **Chatbot Arena** | ~80k human-judged battles between many models | the main training signal |
| **Model tiers** | Arena models grouped into tiers by Elo score, so a battle between any tier-1 and tier-2 model counts as strong vs weak | fixes the sparsity problem (few direct GPT-4 vs Mixtral battles) |
| **Golden labels** | benchmark questions with known answers (e.g. MMLU validation), where correctness decides the winner | adds exact-match signal |
| **LLM-judge labels** | answers from both models, compared by GPT-4 as the judge | adds a lot of cheap, targeted preference data |

The authors find that **adding this data helps a lot**. Routers trained on Arena data alone are only
modestly better than random on some benchmarks. With the extra data, performance improves noticeably.
The released configs use Arena data augmented with LLM-judge labels.

## 3. The routers

All four routers implement one method, `calculate_strong_win_rate(prompt) → float`.

| router | mechanism | needs |
|---|---|---|
| **`sw_ranking`**, similarity-weighted ranking | For each query, reweight every Arena battle by how similar its prompt is to the query (embedding similarity), then solve a weighted **Bradley–Terry** model for the strong model's win rate. No training step; the work happens at inference. | OpenAI embeddings, the battle dataset |
| **`mf`**, matrix factorisation | Learn a scoring function `s(M, q)` over model and query embeddings (a bilinear, recommender-style model), trained so that `σ(s(M_strong, q) − s(M_weak, q))` matches observed preferences. **The authors' recommended router.** | OpenAI embeddings, small trained weights |
| **`bert`**, BERT classifier | Fine-tune a BERT-base encoder with a classification head on the preference labels. | a local model (runs on CPU or GPU) |
| **`causal_llm`**, LLM classifier | Fine-tune Llama 3 8B to predict the preference label as next-token output. | a GPU, an 8B model |
| `random` | Baseline: route to strong with a fixed probability. | none |

The `mf` and `sw_ranking` routers embed queries with OpenAI's API, so **an `OPENAI_API_KEY` is required
even when both models are local**. `bert` and `causal_llm` run without that key.

## 4. Evaluation metrics

Quality is measured on a benchmark as `r(·)`: accuracy for MMLU and GSM8K, judge score for MT-Bench.

- **Performance Gap Recovered (PGR):** how much of the quality gap between the weak and strong
  models the router recovers:

  ```
  PGR = (r(router) − r(weak)) / (r(strong) − r(weak))
  ```

- **Call-Performance Threshold, CPT(x%):** the smallest share of strong-model calls that reaches
  a PGR of x%. For example, CPT(50%) = 30% means the router recovers half the gap while sending only 30% of
  queries to the strong model.
- **APGR:** the area under the PGR vs strong-call share curve, as one summary number.
  Random routing gives a straight line with an APGR of about 0.5.

## 5. Results (as reported)

- On **MT-Bench**, the best routers reach 95% of GPT-4's quality while cutting cost by up to **85%**
  (per the README). Across benchmarks the headline is **more than 2× cost reduction** without a big quality loss.
- **MMLU** and **GSM8K** show smaller gains. Routing on knowledge and math questions is harder, and
  the extra data matters more there.
- The authors report that their routers match commercial routing products while being **more than 40% cheaper**.
- **Swapping the model pair works:** routers trained on GPT-4 vs Mixtral still worked when tested on
  other pairs (for example, Claude 3 Opus vs Llama 3 8B) without retraining. The authors read this as the router
  learning *query difficulty* rather than anything specific to those models.

## 6. The framework

- `Controller`: a drop-in replacement for the OpenAI client. The router and threshold go in the model
  name, e.g. `model="router-mf-0.11593"`.
- `routellm.openai_server`: an OpenAI-compatible HTTP server (default port 6060).
- `routellm.calibrate_threshold`: finds the `α` that sends a target share of calls to the strong
  model on a query set (Chatbot Arena by default; the authors recommend your own traffic).
- `routellm.evals.evaluate`: benchmarks routers on MMLU, GSM8K and MT-Bench, with cached results.
- Model calls go through **LiteLLM**, so any provider works, including Ollama (`ollama_chat/<model>`) and
  any OpenAI-compatible endpoint. [Routing to Local Models](https://github.com/lm-sys/RouteLLM/blob/main/examples/routing_to_local_models.md)
  shows GPT-4 as the strong model and a local Llama 3 on Ollama as the weak one.

## 7. Strengths

1. **You get a router with no data of your own.** The pre-trained routers work on day one.
2. **Public, reusable training data.** Human preferences at scale, plus a recipe for adding more.
3. **Carefully designed metrics.** PGR, CPT and APGR make routers comparable regardless of cost model.
4. **Works across model pairs.** The router appears to learn query difficulty, so it survives model upgrades.
5. **Easy to adopt.** Works as an OpenAI client or server, and threshold calibration is one command.

## 8. Limitations

1. **Cost is a count of strong-model calls.** It assumes every strong-model call costs the same.
   Real cost depends on tokens, and long answers dominate the bill.
2. **Preferences are not correctness.** An Arena vote measures which answer people liked. It can reward style or length.
3. **Distribution shift.** Arena prompts are chat-like. Specialised traffic (your product's
   requests) may look nothing like it. The authors say to calibrate on your own queries, but the router's
   *weights* stay Arena-trained.
4. **Only two models.** Routing among more than two models is outside the paper's scope.
5. **Hosted embeddings.** The recommended router (`mf`) depends on OpenAI's embedding API for every query, which adds latency, cost and a data-egress question.
6. **No savings ceiling.** The framework shows how well a router trades off quality against strong-model calls.
   It doesn't show the most routing could save on your traffic.

---

## Sources

- Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica. *RouteLLM: Learning to Route LLMs with
  Preference Data.* 2024. [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)
- Code: [github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) · Blog: [lmsys.org/blog/2024-07-01-routellm](http://lmsys.org/blog/2024-07-01-routellm/)
- Models and data: [huggingface.co/routellm](https://huggingface.co/routellm)

*Summary based on the paper, README and examples. Check exact figures against the paper before quoting them.*
