# mdb-steer

----


# Building mdb-steer: Dynamic LLM Routing and Telemetry with RouteLLM, Ollama, and MongoDB Atlas

Running high-capability large language models like `llama3.3:70b` across every production request is economically wasteful. While complex tasks—such as lock-free C++ concurrency or dynamic programming—demand deep reasoning, standard tasks like basic string manipulation or factual lookups are handled just as well by compact, high-throughput models like `llama3.2:3b`.

To bridge this efficiency gap, we built **`mdb-steer`**: an open-source dynamic model routing framework. `mdb-steer` evaluates incoming query difficulty in real time, routes requests to the optimal local LLM via Ollama, validates output quality using Braintrust `autoevals`, and persists real-time operational telemetry inside **MongoDB Atlas**.

---

## The Core Problem: Over-provisioning Compute

In standard AI deployments, system architects face a forced trade-off:

1. **Static Strong Routing (`llama3.3:70b`)**: Guarantees high quality and low hallucination rates, but incurs significant compute costs and high latency overhead ($T_{\text{latency}} \sim 450\text{ ms}$).
2. **Static Weak Routing (`llama3.2:3b`)**: Delivers low-latency ($T_{\text{latency}} \sim 80\text{ ms}$) and low compute costs, but fails on complex code generation and multi-step reasoning.

`mdb-steer` solves this by introducing a **dynamic sidecar router** that dynamically estimates the win probability of the strong model ($P_{\text{strong}}$) before execution.

```
                  +-------------------+
                  |   Incoming Query  |
                  +---------+---------+
                            |
                            v
               +-------------------------+
               |  mdb-steer Router      |
               |  (Matrix Factorization) |
               +------------+------------+
                            |
         +------------------+------------------+
         | P(Strong) >= 0.50                   | P(Strong) < 0.50
         v                                     v
+------------------+                  +------------------+
|  llama3.3:70b    |                  |   llama3.2:3b    |
|  (Strong Model)  |                  |   (Weak Model)   |
+--------+---------+                  +--------+---------+
         |                                     |
         +------------------+------------------+
                            |
                            v
               +-------------------------+
               |  Braintrust AutoEvals   |
               |  (Quality Verification) |
               +------------+------------+
                            |
                            v
               +-------------------------+
               |  MongoDB Atlas ODL      |
               |  (Telemetry & Datasets) |
               +-------------------------+

```

---

## Technical Architecture of `mdb-steer`

`mdb-steer` consists of four primary components:

### 1. Matrix Factorization Router Logic

The router computes a logit score based on difficulty heuristics and domain-specific keywords (e.g., C++, Rust, DP algorithms). The probability of requiring the strong model is calculated using the sigmoid function:

$$P(\text{Strong}) = \frac{1}{1 + e^{-\text{logit}}}$$

* **Easy Queries** ($\text{logit} = -0.8 \rightarrow P \approx 0.31$): Routed to `llama3.2:3b`.
* **Medium Queries** ($\text{logit} = 0.0 \rightarrow P \approx 0.50$): Evaluated dynamically based on domain context.
* **Hard/Coding Queries** ($\text{logit} \ge 1.4 \rightarrow P \ge 0.80$): Escorted to `llama3.3:70b`.

### 2. Ollama Local Execution Engine

By leveraging Ollama's OpenAI-compatible local endpoints (`http://localhost:11434/v1`), `mdb-steer` manages execution across local instances without external cloud API dependencies.

### 3. Automated Quality Guardrails (AutoEvals)

To prevent quality regression, every output is evaluated in real time across three automated evaluation metrics via Braintrust `autoevals`:

* **Exact Match (EM)**: Binary scoring for deterministic queries.
* **Levenshtein Similarity**: String distance matching against target expectations.
* **Factuality Score**: Semantic verification to catch off-model hallucinations.

A weighted **Composite Quality Score** out of $10.0$ is generated:

$$\text{Composite Score} = (0.25 \times \text{Levenshtein} + 0.15 \times \text{ExactMatch} + 0.60 \times \text{Factuality}) \times 10$$

### 4. MongoDB Atlas Operational Data Layer (ODL)

All routing metadata, decision overhead latencies, token consumption, compute costs, and `autoevals` outputs are saved directly to MongoDB Atlas (`routellm_ops_db.query_telemetry`). This operational telemetry serves two purposes:

1. **Real-time Monitoring**: Tracking cost reduction vs. quality drift.
2. **Retraining Dataset Creation**: Query pairs where the weak model scores unexpectedly high are used as fine-tuning targets for future router iterations.

---

## Python Pipeline Implementation

Below is the core implementation of the `mdb-steer` pipeline, showing how routing decisions are evaluated and stored:

```python
import os
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import pymongo
from autoevals import Levenshtein, ExactMatch, Factuality

STRONG_MODEL = "llama3.3:70b"
WEAK_MODEL = "llama3.2:3b"

MODEL_COMPUTE_COST = {
    "llama3.3:70b": {"input_cost_per_1m": 0.80, "output_cost_per_1m": 2.40},
    "llama3.2:3b":  {"input_cost_per_1m": 0.04, "output_cost_per_1m": 0.12}
}

class RouteLLMMFRouter:
    """Matrix Factorization & Heuristic Router for mdb-steer."""
    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold
        self.code_keywords = {"code", "function", "python", "rust", "c++", "algorithm", "lock-free", "dp", "raft"}

    def route(self, query_text: str, difficulty: str) -> Tuple[str, float, float]:
        start = time.perf_counter()
        text_lower = query_text.lower()
        is_coding = any(kw in text_lower for kw in self.code_keywords)
        
        logit = -0.8
        if difficulty == "medium": 
            logit += 0.8
        elif difficulty == "hard": 
            logit += 2.2
            
        if is_coding and difficulty in ("medium", "hard"): 
            logit += 1.1

        p_strong = 1.0 / (1.0 + math.exp(-logit))
        overhead_ms = (time.perf_counter() - start) * 1000.0
        chosen_model = STRONG_MODEL if p_strong >= self.threshold else WEAK_MODEL
        
        return chosen_model, round(p_strong, 3), round(overhead_ms, 3)

```

---

## Benchmark Results: Baseline vs. `mdb-steer`

When executing our $10$-query benchmark suite (spanning standard QA, regular expressions, complex C++ lock-free data structures, and Raft consensus implementations), `mdb-steer` demonstrated substantial compute savings with negligible impact on quality.

### Pipeline Decision Breakdown

| ID | Difficulty | $P(\text{Strong})$ | Model Chosen | Factuality | Composite Quality Score |
| --- | --- | --- | --- | --- | --- |
| **q01** | easy | 0.310 | `llama3.2:3b` | 0.950 | 7.20 / 10.0 |
| **q02** | medium | 0.500 | `llama3.3:70b` | 0.980 | 7.38 / 10.0 |
| **q03** | easy | 0.310 | `llama3.2:3b` | 0.950 | 7.20 / 10.0 |
| **q04** | hard | 0.924 | `llama3.3:70b` | 0.950 | 7.20 / 10.0 |
| **q05** | hard | 0.924 | `llama3.3:70b` | 0.950 | 7.20 / 10.0 |
| **q06** | medium | 0.500 | `llama3.3:70b` | 0.980 | 7.38 / 10.0 |
| **q07** | easy | 0.310 | `llama3.2:3b` | 0.950 | 7.20 / 10.0 |
| **q08** | hard | 0.924 | `llama3.3:70b` | 0.950 | 7.20 / 10.0 |
| **q09** | easy | 0.310 | `llama3.2:3b` | 0.950 | 7.20 / 10.0 |
| **q10** | medium | 0.750 | `llama3.3:70b` | 0.980 | 7.38 / 10.0 |

### System Metric Comparison

| Performance Metric | Baseline (100% 70B) | `mdb-steer` Dynamic Routing | Impact Delta |
| --- | --- | --- | --- |
| **Traffic Offloaded to 3B** | 0.0% | **40.0%** | +40.0% compute capacity |
| **Total Compute Cost** | $0.00392 USD | **$0.00248 USD** | **36.73% Cost Reduction** |
| **Average Quality Score** | 7.33 / 10.0 | **7.26 / 10.0** | **99.05% Quality Retention** |
| **Quality Regression** | 0.0% | **0.95%** | **PASS** (< 2.0% Guardrail) |

---

## Operational Intelligence via MongoDB Atlas

By funneling all telemetry into MongoDB Atlas, `mdb-steer` provides actionable insights through structured document schemas. Here is an example document logged during runtime:

```json
{
  "_id": { "$oid": "66f332a81b2e4c8f12a39b01" },
  "query_id": "q03",
  "category": "coding",
  "difficulty": "easy",
  "model_chosen": "llama3.2:3b",
  "p_strong_win": 0.31,
  "decision_latency_ms": 1.24,
  "execution_latency_ms": 84.2,
  "prompt_tokens": 35,
  "completion_tokens": 45,
  "compute_cost_usd": 0.0000068,
  "autoeval_metrics": {
    "exact_match": 0.0,
    "levenshtein_similarity": 0.60,
    "factuality_score": 0.95,
    "composite_quality_score": 7.20,
    "rationale": "AutoEvals Score -> Factuality: 0.95, Levenshtein: 0.60, ExactMatch: 0.0"
  },
  "timestamp": 1727194618.322
}

```

This telemetry stream enables dynamic monitoring via Atlas Charts to track real-time routing breakdown, model accuracy drift, and cost savings over time.

---

## Next Steps for `mdb-steer`

1. **Atlas Vector Search Integration**: Replacing simple logit heuristics with semantic embedding lookups stored in MongoDB Atlas to classify query intent and complexity automatically.
2. **Online Router Preference Learning**: Using saved telemetry from MongoDB Atlas to fine-tune a specialized router model via Direct Preference Optimization (DPO).
