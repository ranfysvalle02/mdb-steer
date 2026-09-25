# Executive Summary: mdb-steer

## The question

Can we cut LLM costs by sending easy requests to a cheaper model, and how much would that actually save?

## The answer

**It depends on your traffic, and you can measure it before building anything.**

mdb-steer runs a large and a small model on real queries, grades every answer, and reports the
**ceiling**: what a perfect router would save. On our workload:

| | saving |
|---|---|
| Perfect router, measured CPU cost | **18%** (even though 64% of requests could go to the small model) |
| Perfect router, cloud pricing (small model 5–20× cheaper) | **49–57%** |
| mdb-steer's learned router, within a 5% quality limit (CPU / 10× pricing) | **8.6% / 16.1%** |
| RouteLLM (LMSYS), off the shelf, same limit (CPU / 10× pricing) | 5.6% / 12.9% |

## What we learned

1. **Measure the ceiling first.** If a perfect router saves little, no router is worth building.
2. **Requests aren't dollars.** The queries a small model handles have short, cheap answers. Moving 64% of
   requests saved 18% of cost.
3. **A router trained on your own data beats a generic one, but only somewhat.** mdb-steer, trained on 150 of our
   own graded queries, ranked difficulty better than RouteLLM's pre-trained router (AUC 0.79 vs 0.62) and saved
   about 3 points more. Neither captured more than half of the ceiling on CPU, or a third of it at cloud prices. RouteLLM's own paper found the same thing: where answers
   are right or wrong (math, exams), routing gains were far smaller than on open-ended chat.
4. **Keep every graded answer.** Storing answers, grades and costs in MongoDB turned every later
   question (new prices, new labels, a new router like RouteLLM) into a re-read in seconds, with no model re-runs.

## Recommendation

| perfect-router saving on your traffic | action |
|---|---|
| under ~15% | Don't route. Shorten outputs, trim prompts, cache. |
| ~15–40% | Route by request type or endpoint, with a quality check. |
| over ~40% | Build a learned router. Start with RouteLLM on day one, then train on your own data. |

## Caveats

These results come from 45 held-out queries on CPU with local Llama models. The direction is clear, but the
exact numbers have wide error bars. Expanding the benchmark is tracked in
[#2](https://github.com/ranfysvalle02/mdb-steer/issues/2).

## Read more

[key-insights.md](key-insights.md) (lessons and playbook) ·
[mdb-steer-vs-routellm.md](mdb-steer-vs-routellm.md) (head-to-head) ·
[routellm.md](routellm.md) (RouteLLM paper summary) · [review.md](review.md) (full results)
