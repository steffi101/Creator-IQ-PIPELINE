"""
Hook Scoring Model

Rule-based evaluation model that scores ad hooks on 6 weighted criteria.
In production, weights would be learned from conversion data.
Current weights are based on documented best practices for short-form video ads.
"""

from pydantic import BaseModel
from typing import Optional
import re


class HookScore(BaseModel):
    text: str
    brevity: float
    specificity: float
    emotion: float
    engagement: float
    interrupt: float
    native: float
    composite: int
    grade: str


EMOTIONAL_TRIGGERS = [
    "stop", "never", "secret", "finally", "worst", "best", "shocking",
    "real", "truth", "broke", "saved", "changed", "lost", "found",
    "crying", "hate", "love", "obsessed", "addicted", "insane",
    "literally", "actually", "nobody", "everyone", "always", "wish"
]

AD_WORDS = [
    "buy now", "limited time", "discount", "offer", "sale", "click",
    "subscribe", "free trial", "sign up", "order now", "act now",
    "don't miss", "exclusive deal", "promo code"
]

GENERIC_STARTS = [
    "this app", "download", "try", "use", "get", "our", "we",
    "the best", "introducing", "new", "check out", "looking for"
]

WEIGHTS = {
    "brevity": 0.15,
    "specificity": 0.20,
    "emotion": 0.20,
    "engagement": 0.15,
    "interrupt": 0.15,
    "native": 0.15,
}


def score_hook(hook_text: str) -> HookScore:
    """Score a single hook on 6 criteria."""
    text = hook_text.lower().strip()
    scores = {}

    # 1. Brevity: under 15 words = good, under 10 = great
    word_count = len(text.split())
    if word_count <= 8:
        scores["brevity"] = 1.0
    elif word_count <= 12:
        scores["brevity"] = 0.85
    elif word_count <= 15:
        scores["brevity"] = 0.7
    elif word_count <= 20:
        scores["brevity"] = 0.4
    else:
        scores["brevity"] = 0.15

    # 2. Specificity: contains numbers, dollar amounts, timeframes
    has_number = bool(re.search(r'\d+', text))
    has_dollar = bool(re.search(r'\$\d+', text))
    has_timeframe = any(t in text for t in ["day", "week", "month", "year", "hour", "minute", "second"])
    specificity_hits = sum([has_number, has_dollar, has_timeframe])
    scores["specificity"] = min(0.3 + specificity_hits * 0.35, 1.0)

    # 3. Emotion: emotional trigger words
    emotion_hits = sum(1 for w in EMOTIONAL_TRIGGERS if w in text)
    scores["emotion"] = min(emotion_hits * 0.35, 1.0) if emotion_hits > 0 else 0.15

    # 4. Engagement: questions, POV, direct address
    is_question = "?" in text
    is_pov = text.startswith("pov")
    is_direct = any(text.startswith(w) for w in ["you ", "your ", "what if", "imagine", "here's", "ever "])
    is_list = text.startswith("3 ") or text.startswith("5 ") or "reasons" in text
    engagement_hits = sum([is_question, is_pov, is_direct, is_list])
    scores["engagement"] = min(0.25 + engagement_hits * 0.3, 1.0)

    # 5. Pattern Interrupt: doesn't start with generic ad language
    starts_generic = any(text.startswith(g) for g in GENERIC_STARTS)
    has_unusual_start = any(text.startswith(s) for s in ["pov", "i ", "my ", "so ", "wait", "okay", "honestly", "nobody"])
    if starts_generic:
        scores["interrupt"] = 0.15
    elif has_unusual_start:
        scores["interrupt"] = 1.0
    else:
        scores["interrupt"] = 0.6

    # 6. Native Feel: doesn't sound like an ad
    ad_hits = sum(1 for w in AD_WORDS if w in text)
    if ad_hits == 0:
        scores["native"] = 1.0
    elif ad_hits == 1:
        scores["native"] = 0.5
    else:
        scores["native"] = 0.15

    # Weighted composite
    composite = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    composite_int = round(composite * 100)

    # Grade
    if composite >= 0.80:
        grade = "A"
    elif composite >= 0.65:
        grade = "B"
    elif composite >= 0.50:
        grade = "C"
    else:
        grade = "D"

    return HookScore(
        text=hook_text,
        brevity=round(scores["brevity"], 2),
        specificity=round(scores["specificity"], 2),
        emotion=round(scores["emotion"], 2),
        engagement=round(scores["engagement"], 2),
        interrupt=round(scores["interrupt"], 2),
        native=round(scores["native"], 2),
        composite=composite_int,
        grade=grade,
    )


def score_hooks_batch(hooks: list[str]) -> list[HookScore]:
    """Score a batch of hooks and return sorted by composite score."""
    scored = [score_hook(h) for h in hooks if h.strip()]
    scored.sort(key=lambda x: x.composite, reverse=True)
    return scored
