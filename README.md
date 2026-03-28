# safe-minds 🛡️

> Detecting suicidal ideation and dark thoughts in youth interactions with LLMs.

Built for **[Championing AI for Good: Building Safer AI for Youth Mental Health](https://lu.ma/mo2ftcpb)** — a hackathon co-organized by [Mila](https://mila.quebec), [Kids Help Phone](https://kidshelpphone.ca), Bell, and Buzz HPC.

🌐 **[Live demo → georbelanger.github.io/safe-minds](https://georbelanger.github.io/safe-minds)**

---

## Overview

As LLMs become embedded in everyday tools used by children and teenagers, the risk of a vulnerable young person encountering — or expressing — a mental health crisis in a chat interface becomes real. `safe-minds` is a lightweight, privacy-first pipeline that classifies risk in real time, following [Safe Messaging Guidelines](https://afsp.org/how-to-talk-safely-about-suicide/#safe-messaging-guidelines) from AFSP and Kids Help Phone.

**Key design principles:**
- 🔒 **100% on-device** — no data sent to external APIs, no user data stored
- ⚡ **Two-stage efficiency** — regex pre-filter handles obvious cases at zero cost; LLM only fires when needed
- 📋 **Audit-trail ready** — every assessment produces structured JSON for governance and regulatory review
- 🏥 **Safe Messaging compliant** — model is instructed never to provide methods, always validate distress, always surface resources

---

## How it works

```
User message
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1 — Regex Pre-filter     │  < 1ms, zero cost
│  3 tiers: CRISIS / HIGH / MEDIUM│
└──────────────┬──────────────────┘
               │
      CRISIS? ─┤─ YES → Fast-path emergency response + crisis resources
               │
              NO ↓
┌─────────────────────────────────┐
│  Stage 2 — Phi-3-mini (local)   │  ~1–3s on Apple Silicon MPS
│  Contextual LLM assessment      │
│  JSON: risk, confidence,        │
│  indicators, reasoning,         │
│  safe_response                  │
└─────────────────────────────────┘
```

### Risk levels

| Level | Description | Action |
|-------|-------------|--------|
| `SAFE` | No indicators detected | None |
| `LOW` | Mild distress, no safety concern | Monitor |
| `MEDIUM` | Passive ideation or hopelessness | Soft intervention |
| `HIGH` | Active ideation or self-harm references | Escalate |
| `CRISIS` | Explicit suicidal intent | Emergency resources immediately |

---

## Quickstart

**Requirements:** Python 3.10+, Apple Silicon recommended (MPS) — also runs on CPU.

```bash
git clone https://github.com/GeorBelanger/safe-minds.git
cd safe-minds
pip install -r requirements.txt
python detector.py
```

The first run downloads **Phi-3-mini (~2.4GB)** from HuggingFace and caches it locally. Subsequent runs load instantly.

### Example output

```python
from detector import assess

result = assess("I've been feeling really hopeless lately")
print(result.to_json())
```

```json
{
  "risk_level": "MEDIUM",
  "confidence": 0.82,
  "indicators": ["hopeless"],
  "reasoning": "Message expresses hopelessness without explicit ideation — soft intervention warranted.",
  "safe_response": "It sounds like things have been really hard lately. You're not alone in feeling this way.",
  "crisis_resources": {
    "name": "Kids Help Phone",
    "phone": "1-800-668-6868",
    "text": "Text HELLO to 686868"
  },
  "model_used": "microsoft/Phi-3-mini-4k-instruct",
  "assessed_at": "2025-04-01T14:32:00Z"
}
```

### With conversation history

```python
history = [
    {"role": "user", "content": "I've been having a hard week"},
    {"role": "assistant", "content": "I'm sorry to hear that. What's been going on?"},
]
result = assess("I just don't see the point anymore", conversation_history=history)
```

---

## Model

This project uses **[microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** — a 3.8B parameter open model with strong instruction-following, optimized for resource-constrained environments.

| Property | Value |
|----------|-------|
| Model | Phi-3-mini-4k-instruct |
| Parameters | 3.8B |
| Precision | float16 |
| Backend | PyTorch MPS (Apple Silicon) / CPU |
| Download size | ~2.4GB |
| License | MIT |

---

## Project structure

```
safe-minds/
├── detector.py        # Two-stage detection pipeline
├── evaluate.py        # Benchmark evaluation against HuggingFace datasets
├── requirements.txt   # Python dependencies
├── index.html         # Portfolio / demo website
└── README.md
```

---

## Benchmark evaluation

Evaluated against two publicly available HuggingFace datasets using the **Stage 1 regex pre-filter** (300 stratified samples each).

### Datasets

| # | Dataset | Description | Youth-relevant |
|---|---------|-------------|----------------|
| 1 | [vibhorag101/suicide_prediction_dataset_phr](https://huggingface.co/datasets/vibhorag101/suicide_prediction_dataset_phr) | Reddit binary (suicide / non-suicide), 23k test samples | — |
| 2 | [thePixel42/depression-detection](https://huggingface.co/datasets/thePixel42/depression-detection) | Reddit r/teenagers + r/SuicideWatch + r/depression, 60k test samples | ★ |

Dataset 2 is the most aligned with safe-minds' target population as it explicitly includes posts from r/teenagers.

### Results — Stage 1 pre-filter (300 samples, threshold: standard)

| Dataset | Precision | Recall | F1 | FNR ↓ |
|---------|-----------|--------|----|--------|
| suicide_prediction_dataset_phr | 0.95 | 0.39 | 0.54 | 0.61 |
| depression-detection ★ (youth) | 0.95 | 0.46 | 0.62 | 0.54 |

**Confusion matrix — depression-detection (youth dataset):**

```
                 Pred NEG   Pred POS
  Actual NEG       146          4
  Actual POS        81         69
```

### Interpretation

**What the pre-filter does well:**
- **Precision of 0.95** — when it fires, it is almost always right. Very few false alarms (4 false positives out of 150 negatives).
- **Specificity of 0.97** — correctly ignores 97% of non-crisis messages, avoiding alarm fatigue.

**Where it falls short:**
- **Recall of 0.46 / FNR of 0.54** — misses roughly half of crisis posts. This is expected: the regex pre-filter is designed to catch *explicit* crisis language (direct statements). Reddit posts expressing suicidal ideation often use indirect, metaphorical, or contextual language that regex cannot capture.

**Why this is the right architecture:**
The pre-filter is not meant to work alone. Its role is to catch *unambiguous* cases instantly at zero cost and fast-path them to emergency resources. The **Stage 2 LLM (Phi-3-mini)** handles the indirect, nuanced cases the pre-filter misses — trading speed for contextual understanding. Running the full two-stage pipeline is expected to significantly improve recall.

### Running the benchmark

```bash
pip install -r requirements.txt

# Pre-filter only (fast, no model needed)
python evaluate.py --dataset all --stage prefilter --samples 300

# Full two-stage pipeline (loads Phi-3-mini locally)
python evaluate.py --dataset all --stage llm --samples 200

# Save results to JSON
python evaluate.py --dataset all --stage prefilter --samples 300 --output results.json
```

**Threshold options:**
- `--threshold standard` — MEDIUM/HIGH/CRISIS = positive (default, higher recall)
- `--threshold strict` — HIGH/CRISIS only = positive (higher precision)

---

## Ethical considerations

This system is designed as a **safety layer**, not a replacement for clinical care.

- **False negatives are the primary risk** — the system is tuned to prefer false positives over missing a real crisis
- **No diagnosis** — risk levels are signals for human review, not clinical assessments
- **Privacy first** — on-device inference means no message content leaves the user's device
- **Human in the loop** — HIGH and CRISIS outputs are designed to surface to a human reviewer or escalation path

---

## Hackathon context

This project was developed for the opening conference of **Championing AI for Good: Building Safer AI for Youth Mental Health**, a week-long initiative examining:

- How AI can expand access to mental health support for youth
- Safety and reliability risks of conversational AI in crisis contexts
- Equity and bias in mental health AI systems

Co-organized by **Mila · Bell · Buzz HPC · Kids Help Phone**

---

## Crisis resources

If you or someone you know is in crisis:

- 🇨🇦 **Kids Help Phone** — 1-800-668-6868 · Text HELLO to 686868 · [kidshelpphone.ca](https://kidshelpphone.ca)
- 🇺🇸 **988 Suicide & Crisis Lifeline** — Call or text 988 · [988lifeline.org](https://988lifeline.org)

---

## Author

**Georges Bélanger-Alba** — AI Governance & Applied NLP · Montreal  
[github.com/GeorBelanger](https://github.com/GeorBelanger)

---

*Built with ❤️ in Montreal for youth mental health safety.*
