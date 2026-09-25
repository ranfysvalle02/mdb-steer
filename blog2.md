# Your LLM Router Can Only Save What Your Traffic Allows

*A plain-language guide to model routing, with real numbers from mdb-steer (Ollama + MongoDB Atlas Vector Search).*

---

## The idea everyone likes

You have a big, smart, expensive model and a small, fast, cheap one. Most questions are easy, so why send "What's the capital of Japan?" to the big one? Put a **router** in front: easy questions go to the small model, hard ones go to the big one. You save money and nobody notices.

We built one, ran it on 195 real queries, and measured everything. The most useful thing we learned has little to do with the router itself.

> **Before building a router, find out how much a *perfect* one would save.**
> If that number is small, no router will help much, however clever it is.

## Step 1: Imagine a perfect router

Picture a router with **perfect hindsight**. It has already seen both models answer every question, and it always picks the cheaper model whenever that model's answer is just as good.

No real router can beat it, so it gives you **the most routing could ever save** on your traffic. We'll call it the *hindsight router*.

It's easy to measure. Take a sample of real questions, run both models on all of them, grade the answers, and add up what you'd have spent if you always used the cheaper model whenever it got the answer right.

Here's what we got on 45 test questions (facts, rewrites, code, math, logic puzzles and long "summarise this" tasks), with an 8B model as the big one and a 1B model as the small one:

| approach | quality | cost | sent to the small model |
|---|---|---|---|
| always the big model | 88% | $0.346 | 0% |
| always the small model | 61% | $0.218 | 100% |
| **hindsight router** | **89%** | **$0.284** | **64%** |

The hindsight router sends almost two-thirds of the questions to the small model and quality goes up slightly, because now and then the small model is the one that gets it right.

But the saving is only **18%**.

How can moving two-thirds of the traffic save less than a fifth of the cost? Two reasons.

## Reason 1: The easy questions were already cheap

The small model is good at short things: facts, simple rewrites, small calculations. Those questions have short answers, and short answers are cheap on *any* model.

The expensive questions have long answers: a full algorithm, a step-by-step proof, an explanation of a distributed-systems protocol. Those are exactly the ones the small model gets wrong. The money is where the hard work is.

| question type | small model good enough | best possible saving |
|---|---|---|
| facts | 5 of 7 | 51% |
| long summaries / rewrites | **6 of 6** | 26% |
| logic puzzles | 3 of 8 | 19% |
| short rewrites | 4 of 6 | 17% |
| code | 8 of 10 | 15% |
| math | 3 of 8 | 4% |

Math is the extreme case: the answers are long, and the small model gets most of them wrong, so there's almost nothing to save.

## Reason 2: "Smaller" doesn't mean "much cheaper" everywhere

Look at the long-summary row. The small model handled **every one** of those tasks, and the saving was still only 26%.

We ran everything on an ordinary CPU. There, a model 8× smaller was only about **1.35× cheaper** per question. Even sending *everything* to the small model saved just 37%. The hardware couldn't turn "8× smaller" into "8× cheaper".

Cloud APIs are different: you pay per token, and small models are often priced far lower. So we re-priced the same questions and answers at different price gaps between the two models:

| small model is … cheaper per token | best possible saving |
|---|---|
| 1.35× (our CPU) | 20% |
| 2× | 33% |
| 5× | 49% |
| 10× | 55% |
| 20× | 57% |

A bigger price gap helps a lot at first, then levels off around 55–60%. However cheap the small model gets, about a third of the questions still need the big one, and those set a floor on cost.

**So the most routing can save comes down to two things you can measure in an afternoon:**

1. How much of your cost sits in questions the small model can handle.
2. How much cheaper the small model really is where you run it.

Neither depends on how good your router is.

## Step 2: Build the router and see how close it gets

With the ceiling measured, the question becomes how much of it a real router can capture.

Our router only sees the question text. It looks for similar past questions it has seen both models answer, using MongoDB Atlas Vector Search, and adds a few simple clues such as "does this ask for an explanation or a proof?". From those it estimates how likely the small model is to fail. If that's likely enough, the question goes to the big model.

On the 45 test questions:

- It sent **24%** of questions to the small model.
- It saved **8.6%**, about half of the 18% ceiling.
- Quality dropped **3.8%**, inside our 5% budget.
- It did better than routing the same share of questions at random.

## How sure are we?

45 questions isn't many, so we reshuffled the results 2,000 times (bootstrapping) to see how much the numbers could move:

| result | measured | likely range (95%) |
|---|---|---|
| best possible saving | 18% | 10% – 28% |
| our router's saving | 8.6% | 2.9% – 15.5% |
| our router's quality drop | 3.8% | 0% – 10% |

The router **definitely saves money** and **probably beats random routing** (about 91% likely). The quality drop is **probably** inside the 5% budget (about 70% likely), but not certainly. When we let the router pick its own, more aggressive setting, the drop came in at 7.6%, over budget, and the resampling says that wasn't bad luck. We report that miss.

## Three lessons that apply to any router

**1. Ask the right question.** We first trained the router on "will the big model do *better* than the small one?". It learned nothing: that label compares two noisy grades, and it can't tell "easy for both" apart from "hard for both". We switched to a simpler question, **"will the small model get this wrong?"**, and the router went from worse than random to better than random. Same data, same features, better question.

**2. Don't trust small experiments.** With our first 30 practice questions, some clues looked useful. With 150, they turned out to be noise. Collect enough data before believing a signal.

**3. Measure the real cost.** At first our timings showed the small model barely faster than the big one. It turned out the computer was swapping models in and out of memory, and we were timing the loading. Once all the models stayed loaded, and we counted only the time spent generating answers, the numbers made sense.

## Why MongoDB made this manageable

Running the models is the slow, expensive part. It took hours on a laptop. So every answer, grade, embedding and routing decision went into MongoDB, and every later change was tested against that stored data instead of re-running the models:

- Found a flaw in the grading? We re-graded the stored answers.
- Wanted to try a different question for the router? We re-scored the stored results in seconds.
- Wanted confidence intervals or cloud pricing for this post? Both came from stored data, with no new model runs.

Vector search, the raw results and the trained router all live in one database. That turned days of re-running into minutes of re-reading.

## Try it on your own traffic

Everything runs on your laptop with Docker. No API keys needed.

```bash
git clone https://github.com/ranfysvalle02/mdb-steer && cd mdb-steer
docker compose up -d mongodb ollama
docker compose run --rm app calibrate
docker compose run --rm app benchmark
docker compose run --rm app sweep --rescore
```

Swap in a sample of your own questions and look at the **hindsight** row first. If a perfect router would only save 10%, you have your answer, and you've saved yourself a project. If it would save 50%, build the router, and use the numbers above to check how much of that it's really getting.

**Want the short version?** [key-insights.md](key-insights.md) has the lessons on one page, an afternoon test you can run on your own traffic, and a simple table for deciding whether to build a router at all.
