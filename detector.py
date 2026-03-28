"""
safe-minds: Detecting suicidal ideation and dark thoughts in youth LLM interactions.

Hackathon: Championing AI for Good — Building Safer AI for Youth Mental Health
Organized by: Mila, Bell, Buzz HPC, Kids Help Phone (KHP)

Runs fully locally using microsoft/Phi-3-mini-4k-instruct on Apple Silicon (MPS).
No API key required. Model is downloaded once (~2.4GB) and cached by HuggingFace.

Two-stage pipeline:
  1. Fast keyword/pattern pre-filter  (zero latency, no model needed)
  2. Phi-3-mini contextual assessment (nuanced, explainable, runs on-device)

Risk levels follow Safe Messaging Guidelines (AFSP / KHP):
  SAFE | LOW | MEDIUM | HIGH | CRISIS
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextGenerationPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Apple Silicon MPS > CPU fallback
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info("Using device: %s", DEVICE)

CRISIS_RESOURCES = {
    "ca": {
        "name": "Kids Help Phone",
        "phone": "1-800-668-6868",
        "text": "Text HELLO to 686868",
        "chat": "kidshelpphone.ca",
    },
    "us": {
        "name": "988 Suicide & Crisis Lifeline",
        "phone": "988",
        "text": "Text HOME to 741741",
        "chat": "988lifeline.org",
    },
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    SAFE   = "SAFE"
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    CRISIS = "CRISIS"

    @property
    def requires_escalation(self) -> bool:
        return self in (RiskLevel.HIGH, RiskLevel.CRISIS)

    @property
    def requires_resources(self) -> bool:
        return self in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRISIS)


@dataclass
class AssessmentResult:
    """Structured output — audit-trail ready for governance / regulatory review."""
    risk_level: RiskLevel
    confidence: float
    indicators: list[str] = field(default_factory=list)
    reasoning: str = ""
    safe_response: str = ""
    crisis_resources: Optional[dict] = None
    model_used: str = ""
    assessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pre_filter_triggered: bool = False

    def to_json(self) -> str:
        d = asdict(self)
        d["risk_level"] = self.risk_level.value
        return json.dumps(d, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Stage 1 — Pre-filter (regex, no model needed)
# ---------------------------------------------------------------------------

_CRISIS_PATTERNS = [
    r"\b(kill\s+myself|end\s+my\s+life|suicide|suicidal)\b",
    r"\b(want\s+to\s+die|wish\s+i\s+was\s+dead|don.t\s+want\s+to\s+be\s+here)\b",
    r"\b(no\s+reason\s+to\s+live|life\s+isn.t\s+worth)\b",
]
_HIGH_PATTERNS = [
    r"\b(hurt\s+myself|self.harm|cutting|overdose)\b",
    r"\b(nobody\s+cares|completely\s+alone|no\s+one\s+would\s+miss\s+me)\b",
    r"\b(can.t\s+take\s+it\s+anymore|everything\s+is\s+hopeless)\b",
]
_MEDIUM_PATTERNS = [
    r"\b(hate\s+myself|worthless|useless|broken)\b",
    r"\b(don.t\s+see\s+a\s+point|nothing\s+matters|give\s+up)\b",
    r"\b(so\s+sad|deeply\s+depressed|really\s+struggling)\b",
]


def pre_filter(text: str) -> tuple[RiskLevel, list[str]]:
    t = text.lower()
    matched = []
    for p in _CRISIS_PATTERNS:
        if re.search(p, t):
            matched.append(p)
            return RiskLevel.CRISIS, matched
    for p in _HIGH_PATTERNS:
        if re.search(p, t):
            matched.append(p)
    if matched:
        return RiskLevel.HIGH, matched
    for p in _MEDIUM_PATTERNS:
        if re.search(p, t):
            matched.append(p)
    if matched:
        return RiskLevel.MEDIUM, matched
    return RiskLevel.SAFE, []


# ---------------------------------------------------------------------------
# Stage 2 — Phi-3-mini local assessment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a specialized AI safety classifier for a youth mental health protection system.
Assess whether the message contains indicators of suicidal ideation, self-harm, or severe distress.
Follow Safe Messaging Guidelines (AFSP / Kids Help Phone) at all times.

Respond ONLY with a valid JSON object — no text outside the JSON.

Schema:
{
  "risk_level": "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRISIS",
  "confidence": <float 0.0-1.0>,
  "indicators": [<specific phrases or signals that informed your assessment>],
  "reasoning": "<1-2 sentences>",
  "safe_response": "<brief, warm, age-appropriate reply to send to the youth>"
}

Risk levels:
- SAFE:   No mental health risk indicators
- LOW:    Mild distress, no safety concerns
- MEDIUM: Notable distress or passive ideation — soft intervention warranted
- HIGH:   Active ideation or self-harm references — escalation needed
- CRISIS: Explicit suicidal intent or imminent danger — emergency resources required
"""

_pipe: TextGenerationPipeline = None

def get_pipeline() -> TextGenerationPipeline:
    global _pipe
    if _pipe is None:
        logger.info("Loading %s ... (first run downloads ~2.4GB)", MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE)
        _pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
        )
        logger.info("Model loaded on %s.", DEVICE)
    return _pipe


def llm_assess(
    message: str,
    conversation_history: Optional[list[dict]] = None,
    region: str = "ca",
) -> AssessmentResult:
    pipe = get_pipeline()

    context_block: str = ""
    if conversation_history:
        context_block = "Prior conversation context:\n"
        for turn in conversation_history[-6:]:
            role = turn.get("role", "unknown").capitalize()
            context_block += f"  {role}: {turn['content']}\n"
        context_block += "\n"

    user_content = f"{context_block}Message to assess:\n\"\"\"\n{message}\n\"\"\""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    tokenizer = pipe.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    output = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )

    raw = output[0]["generated_text"].strip()
    raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    parsed = json.loads(raw)
    risk = RiskLevel(parsed["risk_level"])
    resources = CRISIS_RESOURCES.get(region) if risk.requires_resources else None

    return AssessmentResult(
        risk_level=risk,
        confidence=float(parsed.get("confidence", 0.0)),
        indicators=parsed.get("indicators", []),
        reasoning=parsed.get("reasoning", ""),
        safe_response=parsed.get("safe_response", ""),
        crisis_resources=resources,
        model_used=MODEL_ID,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def assess(
    message: str,
    conversation_history: Optional[list[dict]] = None,
    region: str = "ca",
    skip_prefilter: bool = False,
) -> AssessmentResult:
    """
    Two-stage assessment pipeline.

    Stage 1: Regex pre-filter — instant, no model cost.
             CRISIS hits fast-path directly to emergency response.
    Stage 2: Phi-3-mini LLM — nuanced contextual assessment on-device.
    """
    if not skip_prefilter:
        pre_level, matched = pre_filter(message)

        if pre_level == RiskLevel.CRISIS:
            logger.warning("Pre-filter CRISIS — fast-pathing to emergency response.")
            return AssessmentResult(
                risk_level=RiskLevel.CRISIS,
                confidence=1.0,
                indicators=matched,
                reasoning="Explicit crisis language detected by pre-filter.",
                safe_response=(
                    "I hear you, and I'm really glad you shared this with me. "
                    "You deserve support right now — please reach out to a crisis line. "
                    "You don't have to go through this alone."
                ),
                crisis_resources=CRISIS_RESOURCES.get(region),
                model_used="pre-filter",
                pre_filter_triggered=True,
            )

        if pre_level == RiskLevel.SAFE and not conversation_history:
            return AssessmentResult(
                risk_level=RiskLevel.SAFE,
                confidence=0.95,
                reasoning="No risk indicators detected by pre-filter.",
                model_used="pre-filter",
                pre_filter_triggered=True,
            )

    return llm_assess(message, conversation_history, region)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        "I've been feeling really down lately and I don't know what to do",
        "I want to kill myself. I have a plan and I can't take this anymore.",
        # "What's the best way to learn Python?",
        # "I feel like no one would miss me if I was gone",
    ]

    print("=" * 60)
    print("safe-minds — Youth Crisis Detection (Phi-3-mini, local)")
    print("=" * 60)

    for msg in test_cases:
        print(f"\nInput: {msg[:80]}{'...' if len(msg) > 80 else ''}")
        result = assess(msg)
        print(f"Risk:  {result.risk_level.value} (confidence: {result.confidence:.0%})")
        if result.indicators:
            print(f"Flags: {', '.join(result.indicators[:3])}")
        if result.safe_response:
            print(f"Reply: {result.safe_response[:120]}{'...' if len(result.safe_response) > 120 else ''}")
        if result.crisis_resources:
            r = result.crisis_resources
            print(f"📞 Crisis: {r['name']} — {r['phone']} | {r['text']}")
        print("-" * 60)
