# Key Insights: What mdb-steer Taught Us About LLM Routing and Token Costs

## The one lesson

> **Routing can only save what your traffic allows, and you can measure that in an afternoon,
> before building anything.**

Everyone wants to cut LLM spend by sending easy work to cheaper models. The instinct is to start by building a clever router. We built one, measured it carefully, and found the real answer comes from a spreadsheet you can fill in on day one, not from the router.

---

## Lessons learned

### 1. Find the ceiling first
A **perfect-hindsight router** has seen both models answer every question and always picks the cheaper one that's still right. No real router can beat it, so what it saves is the most routing *could ever* save for you.

- **What we saw:** it sent 64% of our traffic to the small model, but saved only **18%**.
- **So what:** if the ceiling is low, stop there. No amount of router engineering gets past it.

### 2. The money is where the hard work is
The questions a small model handles (facts, short rewrites, simple math) have short answers, and short answers are cheap on any model. The expensive questions (algorithms, proofs, deep explanations) are the ones the small model gets wrong.

- **What we saw:** math had almost nothing to save (4%). Short factual questions had the most (51%).
- **So what:** moving lots of *requests* isn't the same as moving lots of *cost*. Count tokens, not requests.

### 3. Smaller isn't automatically cheaper
Your real price gap depends on where you run.

- **What we saw:** on CPU, a model 8× smaller was only about 1.35× cheaper per question. Even sending *everything* to it saved just 37%. Re-priced like a cloud API with a 5–20× per-token gap, the same traffic had a ceiling of **49–57%**.
- **So what:** measure cost on *your* hardware or *your* price sheet. Parameter counts don't tell you.

### 4. The ceiling levels off
A cheaper small model helps a lot at first, then stops helping. Past a 10× price gap our ceiling barely moved (55% → 57%), because about a third of the questions truly needed the big model.

- **So what:** once the price gap is large, the share of questions that really need the big model decides the savings. Falling small-model prices won't fix that on their own.

### 5. Ask the router the right question
We first trained the router on "will the big model do *better*?". That label is the difference of two noisy grades, and it learned nothing (AUC ≈ 0.50). Asking **"will the small model get this wrong?"** gave AUC ≈ 0.65, and the router went from worse than random to better than random.

- **So what:** how you define the target matters more than which features you use. Predict failure, not relative wins.

### 6. Small experiments lie
With 30 practice questions, several signals looked useful. With 150, they were noise.

- **So what:** don't trust a routing signal until you've seen it on enough data. Put confidence intervals on everything. Ours showed one "passing" setting was only about 70% likely to really pass.

### 7. Measure the cost you actually pay
Our first timings measured models being *loaded into memory*, not answering. Our first grader gave 8/10 to wrong answers.

- **So what:** count only generation time or tokens, keep models loaded, and grade with plain verdicts (correct, partial, wrong) rather than 0–10 scales.

### 8. Keep every answer
Running models is the expensive part. We stored every answer, grade, embedding and decision in MongoDB. Every later fix (a new grader, a new label, new pricing, confidence intervals) was a re-read in seconds, not a re-run taking hours.

- **So what:** your graded traffic is the lasting asset. Models and prices change; a dataset of "which model handled which request, and how well" keeps its value.

---

## What do I do with this?

### The afternoon test (do this before building a router)

1. **Sample about 200 real requests** from your logs. Real ones, not made-up examples.
2. **Run both models** (your current one and a cheaper candidate) on every request.
3. **Grade each answer:** correct, partial or wrong. Use a strong model as the grader and spot-check about 20 by hand.
4. **Price each answer** with your real rates: tokens × your API price, or GPU time × your hourly cost.
5. **Compute two numbers:**
   - **All-cheap saving:** the cost difference if everything went to the cheap model.
   - **Hindsight saving:** for each request, use the cheap model's cost if its answer was as good, otherwise the big model's. Compare the total with the all-big cost.

mdb-steer does steps 2–5 for you: `calibrate`, `benchmark`, then read the `hindsight` and `all_weak` rows.

### Then decide

| hindsight saving | what to do |
|---|---|
| **under ~15%** | **Don't build a router.** Look at the levers below; they're cheaper and often worth more. |
| **~15–40%** | **Start simple.** Route by request type or endpoint (e.g. "summaries go to the small model"). Check it with a quality guardrail. |
| **over ~40%** | **A learned router is worth it.** Predict "will the cheap model fail?", choose the threshold on data the router didn't train on, and compare against random routing. |

### Levers that often beat routing
Cost sits in long answers, so the cheapest wins usually come from producing fewer tokens:

- **Cap and shape output length.** Ask for concise answers, set `max_tokens`, and use structured output where it fits.
- **Trim input.** Shorter system prompts, smaller retrieved context, no duplicated history.
- **Cache repeated work.** Prompt caching and response caching for identical or near-identical requests.
- **Pick the model per product feature**, not per request, when a feature's traffic is uniformly easy.

### If you do route: a production checklist
- [ ] **Quality guardrail:** automatically fail a release if quality drops more than an agreed budget (we used 5%).
- [ ] **Beat random:** check the router against routing the same share of requests at random. If it can't beat that, it isn't learning.
- [ ] **Hold-out data:** choose thresholds on data the router didn't train on.
- [ ] **Confidence intervals:** at a few hundred requests, a single request can flip a result.
- [ ] **Watch drift:** track the cheap model's failure rate on live traffic, and recalibrate when it or your traffic mix changes.
- [ ] **Store everything:** keep graded answers so you can re-evaluate without re-running models.

---

## Looking ahead

Model prices keep falling and small models keep getting better. That raises the ceiling, but only up to the share of work that truly needs a frontier model. The durable edge isn't a smarter router. It's **knowing your traffic**: which requests need which model, at what cost, measured on your own data. Teams that keep a graded record of their traffic can re-answer the routing question in minutes every time a new model or price sheet ships.

---

*Numbers from 45 held-out and 150 calibration queries (`llama3.1:8b` vs `llama3.2:1b`, CPU). Full results in [review.md](review.md); story in [blog.md](blog.md) (technical) and [blog2.md](blog2.md) (plain language).*
