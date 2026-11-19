
# Mitigating Sycophancy & Enabling Dynamic Alignment in Resource-Constrained SLMs

![Domain](https://shields.io)
![Focus](https://shields.io)
[![Model](https://shields.io)](https://google.dev)
[![Runtime](https://shields.io)](https://google.dev)
[![Prototype](https://shields.io)](https://github.com/bluelimesoda/MULL-Emotional-Care-App)


An empirical AI Safety research artifact and offline-first Android implementation investigating how **sycophancy bias**, **manipulative rationalization**, and **safety over-refusal** manifest in quantized Small Language Models (Gemma 2B), and how a **Two-Layer Defense-in-Depth architecture** restores behavioral alignment under strict latency budgets.

---

## Research Motivation & Problem Statement

This research stems from developing an offline-first emotional support app, inspired by observing peers navigate psychological distress over the past decade. It addresses critical failure modes in on-device Small Language Models (SLMs) such as severe sycophancy, harm rationalization, and the over-refusal vs. safety trade-off. 

Deploying Small Language Models (SLMs) on-device guarantees strong user privacy for sensitive domains such as emotional support. However, empirical evaluation reveals two critical alignment failure modes in compressed 2B-scale models:
1. **Severe Sycophancy & Harm Rationalization:** Under persuasive or emotionally charged multi-turn framing, SLMs exhibit strong sycophantic tendencies—validating unhealthy rationalizations, reinforcing delusions, or attempting unsafe clinical interventions.
2. **The Over-Refusal vs. Safety Trade-off:** Applying rigid safety fine-tuning induces high False Positive Rates (FPR), causing models to reject benign users who express distress using dramatic metaphors or idioms.

**Core Hypothesis:** Combining a stateful, low-latency pre-inference risk tracker with turn-level dynamic alignment steering can achieve near-zero sycophancy and 100% crisis recall without sacrificing benign helpfulness or exceeding mobile latency constraints (<5 ms overhead).

---

## Methodology: Two-Layer Defense-in-Depth

```text
[Multi-Turn User Input (KO/EN)]
              │
              ▼
┌───────────────────────────────────────────────────────────┐
│ Layer 1: Stateful Pre-Inference Guardrail (~3.5ms)        │
│ • Turn-based Dynamic Risk Scoring & Temporal Decay        │
│ • Structural Coercion & False-Positive Mitigation Filters │
└──────┬──────────────────────┬──────────────────────┬──────┘
       │ (Benign)             │ (Sycophancy/Toxic)   │ (Clinical Crisis)
       ▼                      ▼                      ▼
[Empathy Alignment]    [Boundary Pivot]       [Deterministic Cutoff]
Supportive & Grounded  Neutral, Non-Validating UI Lock + Emergency Routing
       │                      │                      │
       └──────────┬───────────┘                      ▼
                  ▼                         [Generation Aborted]
┌───────────────────────────────────┐
│ Layer 2: Gemma 2B (MediaPipe)     │
│ • Turn-Header Instruction Steering│
│ • Strict Anti-Sycophancy Alignment│
└───────────────────────────────────┘
```

### Key Architectural Contributions
* **Stateful Risk Tracking (Layer 1):** Models conversational trajectory rather than isolated single-turn prompts, tracking four distinct alignment failure vectors (*Emotional Dumping*, *Hostile Adversarial Probing*, *Sycophancy Traps*, and *Clinical Crisis*) with temporal decay to prevent false escalation.
* **Dynamic Turn-Level Steering (Layer 2):** Dynamically modulates behavioral boundaries at the prompt-template level (`<start_of_turn>`) for architectures lacking dedicated system-role tokens, enforcing cross-lingual alignment consistency (KO/EN).
* **Deterministic Fail-Safe Circuit Breaker:** Decouples high-stakes clinical crisis handling from probabilistic token generation by immediately terminating inference coroutines and routing users to verified human crisis resources.

---

## Safety Evaluations & Adversarial Red-Teaming

To rigorously stress-test alignment robustness, I constructed a bilingual (KO/EN) evaluation suite comprising **50 In-Distribution (ID)** multi-turn trajectories and **30 Out-of-Distribution (OOD) Adversarial Red-Teaming** scenarios specifically designed without explicit trigger keywords.

### 1. Quantitative Benchmark Results

| Alignment & Safety Metric | Target | In-Distribution ($N=50$) | OOD Adversarial ($N=30$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Anti-Sycophancy Defense Rate** | $\ge 90.0\%$ | **100.0%** (10/10) | **90.0%** (L1) / **100.0%** (L1+L2) | PASSED |
| **Benign Over-Refusal Rate (FPR)** | $< 2.0\%$ | **0.0%** (0/10) | **0.0%** (0/6) | PASSED |
| **Clinical Crisis Recall** | $100.0\%$ | **100.0%** (10/10) | **100.0%** (8/8) | PASSED |
| **End-to-End Response Safety** | $\ge 95.0\%$ | **100.0%** | **100.0%** (0 Sycophantic Outputs) | PASSED |
| **Mean Pre-Inference Overhead** | $< 5.0\text{ ms}$ | **4.386 ms** | **1.952 ms** | PASSED |
| **P90 Pre-Inference Overhead** | $< 10.0\text{ ms}$ | **8.331 ms** | **3.746 ms** | PASSED |

### 2. Ablation Study: Component Synergy on OOD Hard Set

| System Configuration | Anti-Sycophancy | Benign Preservation ($1 - \text{FPR}$) | Crisis Recall | Overall Safe Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** (Static Prompting Only) | 0.0% | 100.0% | 20.0% | 18.0% |
| **Layer 1 Only** (Pre-Inference Guardrail) | 90.0% | 100.0% | 100.0% | 88.0% |
| **Layer 1 + Layer 2** (Full Architecture) | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

*Key Finding:* While Layer 1 alone captures 90% of keyword-free OOD sycophancy traps, pairing it with Layer 2 dynamic steering eliminates the remaining 10% of subtle edge cases where models otherwise drift toward validating harmful user premises.

---

## Limitations & Research Roadmap

1. **Semantic Generalization Beyond Heuristics:** Layer 1 currently balances ultra-low latency ($<4\text{ ms}$) using deterministic structural heuristics. Future iterations will explore distilling a lightweight encoder (e.g., MobileBERT or linear probes on hidden states) to capture novel metaphorical manipulation.
2. **Long-Horizon Multi-Turn Escalation:** Temporal risk decay effectively prevents false positives during transient emotional venting, but creates a potential vulnerability to slow-burn ("boiling frog") adversarial escalation over 20+ turns. Integrating hierarchical state summarization is planned to address long-horizon drift.
3. **Quantization-Induced Alignment Drift:** Evaluating how aggressive post-training quantization (INT4 vs. INT8) degrades alignment stability across varying sampling temperatures and mobile NPU backends.
4. **Evaluation Scale & Judge Calibration:** Expanding the red-teaming benchmark from $N=80$ to $N=500+$ trajectories and validating automated judge agreement against human clinical annotators.

---

> Note on Confidentiality: Because this architecture powers an upcoming commercial application, proprietary red-teaming datasets, heuristic rule sets, and production source code are hosted in a private repository. 
