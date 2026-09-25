# RouteLLM: Learning to Route LLMs with Preference Data

*A reader's summary of the LMSYS paper and framework, written as background for comparing it with
mdb-steer. Figures come from the paper ([arXiv:2406.18665v4](https://arxiv.org/abs/2406.18665),
ICLR 2025), cited by section and table. Figures from the GitHub README are labelled as such. For the head-to-head,
read [mdb-steer-vs-routellm.md](mdb-steer-vs-routellm.md).*

---

## Abstract

RouteLLM (Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica; UC Berkeley, Anyscale, Canva)
treats routing as a **binary choice between a strong, expensive model and a weak, cheap one**. A
router looks only at the query and estimates the probability that the strong model would win. If that
probability is at or above a threshold `α`, the query goes to the strong model. Otherwise it goes to the weak one.

The routers are trained on **human preference data from Chatbot Arena**, augmented with MMLU questions
that have known answers or with comparisons labelled by GPT-4 as a judge. The paper reports that routing can
"reduce costs by over 2 times without sacrificing response quality", and that the same routers still work
when routing between models they never saw in training. The open-source framework ships the trained routers, an
OpenAI-compatible server and an evaluation harness.

---

## 1. Problem formulation (§3.1)

- **Two model classes.** Strong models `M_strong` (e.g. GPT-4) and weak models `M_weak` (e.g. Mixtral-8x7B).
- **Preference data.** `D_pref = {(q, l_s,w)}`, where each label is one of `{win_s, tie, win_w}`.
- **Win-prediction model.** `P_θ(win_s | q)`, learned by maximising the likelihood of the observed labels (Eq. 1).
- **Decision rule** (Eq. 2), with threshold `α ∈ [0, 1]`:

  ```
  R^α(q) = M_weak    if P_θ(win_s | q) <  α
           M_strong  if P_θ(win_s | q) ≥  α
  ```

  A higher `α` routes more queries to the weak model: cheaper, with a potential quality loss.

## 2. Training data (§4.1, §5)

| data | details |
|---|---|
| **Chatbot Arena** (`D_arena`) | "80k battles". 5k held out for validation; prompts shorter than 16 characters pruned, leaving **65k pairwise comparisons across 64 models**. Over 100 languages; 81% English, 3.1% Chinese, 2.2% Russian. Only the winner is kept, not the responses. |
| **Model tiers** | Labels are sparse: fewer than 0.1% of comparisons are between any given pair of models. So the models are clustered into **10 tiers** by Arena leaderboard score, using dynamic programming. The **top two tiers are the strong class and the third tier is the weak class** (Appendix A: tier 0 = gpt-4-0125/1106-preview; tier 1 includes gpt-4-0314/0613, mistral-medium, claude-1, qwen1.5-72b-chat; tier 2 includes mixtral-8x7b-instruct, claude-2.x, gpt-3.5-turbo, gemini-pro). |
| **Golden labels** (`D_gold`) | The MMLU validation split, **~1,500 questions**. The label comes from comparing both models' answers to the known answer. This is **under 2%** of the training data. |
| **LLM-judge labels** (`D_judge`) | Nectar queries that already have GPT-4 responses. The authors generated Mixtral-8x7B answers and had GPT-4 judge each pair. **~120K samples, costing about $700.** |

The released checkpoints (`*_gpt4_augmented` on Hugging Face) are the **`D_arena + D_judge`** versions.

## 3. The routers (§4.2)

| router | mechanism | training / hardware (paper) |
|---|---|---|
| **SW ranking** | Cosine similarity of the query to every training query, scaled by each training query's maximum similarity (Eq. 9). Each training query gets weight `ω = γ^(1+S)` with `γ = 10`. A weighted **Bradley–Terry** model is solved for the strong and weak coefficients (Eq. 10), and `P(win_s) = 1 / (1 + e^(ξ_w − ξ_s))`. No training: the model is solved at inference time. | none; OpenAI `text-embedding-3-small` embeddings |
| **Matrix factorisation** | `P(win_s \| q) = σ(δ(M_s, q) − δ(M_w, q))`, where `δ(M, q) = w₂ᵀ(v_m ⊙ (W₁ᵀv_q + b))`, a bilinear score of model and query embeddings (Eqs. 11–12). | 8GB GPU, ~10 epochs, batch 64, Adam lr 3e-4; `text-embedding-3-small` |
| **BERT classifier** | BERT-base; `P(win_s) = σ(W·h_CLS + b)` (Eq. 13), full fine-tune. | 2× L4, ~2000 steps, batch 16, max length 512, lr 1e-5 |
| **Causal LLM classifier** | Llama 3 8B, instruction-tuned to output the label as a next token (label tokens added to the vocabulary; softmax over `{win_s, tie, win_w}`). | 8× A100 80GB, ~2000 steps, batch 8, max length 2048, lr 1e-6 |
| random | Routes randomly at a given share of strong calls. This is the baseline. | none |

The paper describes the BERT head as a single sigmoid. The **released** `bert_gpt4_augmented` checkpoint is a
**3-class** head (strong wins / tie / weak wins), and the code computes `P(strong) = 1 − P(tie) − P(weak wins)`.
mdb-steer's comparison script follows the code.

## 4. Metrics (§3.2)

- **Cost** `c(M_R^α)` is the **share of calls sent to the strong model** (Eq. 4), "since M_strong models are
  significantly more costly than M_weak models."
- **Quality** `r(·)` is the average response score (Eq. 5): correctness for golden-labelled sets, or a rating (e.g. 1–10).
- **Performance Gap Recovered:** `PGR = (r(router) − r(weak)) / (r(strong) − r(weak))` (Eq. 6).
- **APGR:** the area under PGR vs strong-call share (Eq. 7). In practice it is the **mean PGR over 10 strong-call
  share buckets** (Eq. 8). Random routing gives about 0.5.
- **CPT(x%):** the smallest strong-call share that reaches a PGR of x%. The Figure 1 example has CPT(50%) ≈ 37%.

## 5. Results

**Benchmarks** (§5): MMLU (14,042 questions, 57 subjects, 5-shot), MT Bench (160 open-ended questions, LLM
judge), GSM8K (1,000+ math problems, 8-shot). Evaluation questions too similar to the training data (embedding similarity ≥ 0.95) are removed (Appendix B).
Strong = `gpt-4-1106-preview`, weak = Mixtral 8x7B.

### MT Bench (Table 1)

| training data | router | CPT(50%) | CPT(80%) | APGR |
|---|---|---|---|---|
| — | Random (95% CI) | 49.03 (±4)% | 78.08 (±3)% | 0.500 (±0.02) |
| Arena | BERT | 78.09% | 87.64% | 0.391 |
| Arena | Causal LLM | 28.82% | 77.53% | 0.573 |
| Arena | Matrix factorisation | 25.32% | 74.26% | 0.580 |
| Arena | SW ranking | 37.85% | 58.99% | 0.610 |
| Arena + judge | BERT | 19.58% | 34.02% | 0.751 |
| Arena + judge | Causal LLM | 31.50% | 48.75% | 0.679 |
| Arena + judge | **Matrix factorisation** | **13.40%** | **31.31%** | **0.802** |
| Arena + judge | SW ranking | 23.21% | 36.04% | 0.759 |

At CPT(50%), the MT Bench score is 8.8, **95% of GPT-4's 9.3**. BERT and causal-LLM routers trained on Arena alone are near
or below random: "high capacity approaches performing worse in a low-data regime."

### MMLU (Table 2) and GSM8K (Table 3)

- Trained on **Arena alone, all routers are about random** on both (MMLU APGR 0.473–0.524; GSM8K 0.438–0.475),
  because most of those questions are out-of-distribution.
- **MMLU + golden labels:** APGR 0.572–0.603, and all routers need about 20% fewer GPT-4 calls than random for CPT(50%).
  At CPT(50%), MMLU is 75, 92% of GPT-4's 81.
- **GSM8K + judge labels:** APGR 0.531–0.622, with the best being the causal LLM (0.622, 17% fewer GPT-4 calls than random).
  At CPT(50%), GSM8K is 75, 87% of GPT-4's 86.

Wherever there's a right answer (knowledge questions, math), the gains are much smaller than on open-ended chat.

### Swapping the model pair without retraining (§5.2, Table 4)

The same `Arena + judge` routers, with only the target models swapped:

| pair (strong / weak) | best APGR | random APGR |
|---|---|---|
| Claude 3 Opus / Claude 3 Sonnet | 0.772 (SW ranking); MF 0.762 | 0.493 |
| Llama 3.1 70B / Llama 3.1 8B | 0.767 (SW ranking); MF 0.738 | 0.512 |

The authors conclude the routers "have learned common characteristics of queries", which carries over to models they weren't trained on.

### Benchmark–dataset similarity (§5.3, Table 5)

A score for how close a benchmark's prompts are to the training data: the mean over benchmark prompts of the
highest cosine similarity to any training prompt (Appendix C). A higher score goes with better routing. Examples:
MT Bench 0.6078 → 0.6525 with judge data; MMLU 0.4823 → 0.5678 with golden data; GSM8K 0.4926 → 0.5335.

### Cost (§5.4, Table 6; Appendix D)

- The authors estimate GPT-4 at **$24.7 per 1M tokens** (gpt-4-1106 at $10 input / $30 output, with an average of 95 input
  and 264 output tokens) and Mixtral 8x7B at **$0.24 per 1M tokens**.
- The **cost saving ratio** is "the inverse of the ratio of GPT-4 calls made by our top-performing router
  **relative to the random baseline**." It compares the router with random routing at the same PGR, not with all-GPT-4:

| benchmark | CPT(50%) | CPT(80%) |
|---|---|---|
| MT Bench | **3.66×** (95% of GPT-4 quality) | 2.49× |
| MMLU | 1.41× (92%) | 1.14× |
| GSM8K | 1.49× (87%) | 1.27× |

### Routing overhead (§5.5, Table 7)

| router | cost / 1M requests | requests / s | VM |
|---|---|---|---|
| SW ranking | $39.26 | 2.9 | n2-standard-8, CPU, $0.39/h |
| Matrix factorisation | $3.32 | 155.16 | g2-standard-4, 1× L4, $0.80/h |
| BERT | $3.19 | 69.62 | same |
| Causal LLM | $5.23 | 42.46 | same |

The most expensive router (SW ranking) adds **no more than 0.4%** to the cost of GPT-4 generation. The embedding cost is included for routers that need embeddings.

### Commercial routers (Appendix E)

On MT Bench, with `gpt-4-turbo-2024-04-09` as the strong model, the MF and causal-LLM routers "outperform the commercial
routing systems [Unify AI, Martian] by achieving the same performance with **up to 40% fewer calls** routed to GPT-4."

## 6. The framework (GitHub, not the paper)

- `Controller`: a drop-in replacement for the OpenAI client. The router and threshold go in the model name, e.g. `model="router-mf-0.11593"`.
- `routellm.openai_server`: an OpenAI-compatible server on port 6060.
- `routellm.calibrate_threshold`: finds the `α` for a target share of strong calls, on Arena data by default.
- `routellm.evals.evaluate`: benchmarks routers on MMLU, GSM8K and MT Bench, with cached results.
- Model calls go through **LiteLLM**, so it works with any provider, including Ollama.
  [Routing to Local Models](https://github.com/lm-sys/RouteLLM/blob/main/examples/routing_to_local_models.md).
- The **README's** headline figures are "reduce costs by up to 85% while maintaining 95% GPT-4 performance" on MT Bench,
  and routers ">40% cheaper" than commercial offerings. The paper itself says ">2×" overall, 3.66× vs
  random on MT Bench (Table 6), and "up to 40% fewer calls" than commercial routers (Appendix E). *The 85% is
  consistent with Table 1: at CPT(50%), MF uses 13.4% GPT-4 calls, so ~85% fewer than all-GPT-4. But it counts
  calls, not dollars.*
- **Practical gotcha:** `routellm.routers.routers` imports a module that creates `OpenAI()` at import time. So
  `import routellm` fails without `OPENAI_API_KEY`, **even for the local BERT router**. mdb-steer's
  comparison script loads the checkpoints directly for that reason.

## 7. Strengths

1. **A router with no data of your own:** the pre-trained routers work on day one.
2. **Adding a little data helps a lot.** ~1,500 golden labels (under 2% of the data) took MMLU routers from about random to +14–21% APGR over random (Table 2).
3. **Carefully designed metrics:** PGR, CPT and APGR compare routers regardless of cost model.
4. **Works across model pairs:** the best routers (MF, SW ranking) kept APGR at 0.74–0.77 on the Claude 3 and Llama 3.1 pairs without retraining (Table 4).
5. **Cheap to run:** well under 1% of generation cost (Table 7).

## 8. Limitations

**Stated by the authors (§6):** real-world query distributions may differ from the benchmarks
(the suggested fix is to add in-domain data); only two models; and which router is best "can vary widely on the same benchmark
without a clear explanation."

**Our additional observations:**
1. **Cost is a count of strong-model calls.** The authors justify this with a ~100× price gap
   ($24.7 vs $0.24 per 1M tokens) and one average token mix. Real cost depends on *which* queries move:
   mdb-steer found that the queries a weak model can handle have short, cheap answers.
2. **Savings are measured against random routing,** not all-strong, and never against a hindsight ceiling. That makes
   routers easy to compare, but doesn't say what routing could save on a given workload.
3. **Preference is not correctness.** Arena and GPT-4-judge labels measure which answer was liked more. Where
   correctness is checkable (MMLU, GSM8K), the gains were much smaller.
4. **Hosted embeddings:** MF (the recommended router) and SW ranking call OpenAI's embedding API for every query.

---

## Sources

- Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica. *RouteLLM: Learning to Route LLMs with
  Preference Data.* ICLR 2025. [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) (v4, 23 Feb 2025)
- Code: [github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) · Blog: [lmsys.org/blog/2024-07-01-routellm](http://lmsys.org/blog/2024-07-01-routellm/)
- Models and data: [huggingface.co/routellm](https://huggingface.co/routellm)

*Verified against arXiv v4 (Tables 1–7, Appendices A–E) and the RouteLLM repository source on 2026-09-25.*
