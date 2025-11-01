from typing import List, Dict
import re

def _sent_split(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]

def _jaccard(a_tokens: set, b_tokens: set) -> float:
    inter = len(a_tokens & b_tokens)
    uni = len(a_tokens | b_tokens) or 1
    return inter / uni

def summarize_to_notes(
    chunks: List[Dict],
    max_bullets: int = 12,
    max_chars_per_bullet: int = 220,
    dedupe_threshold: float = 0.7,
) -> List[str]:
    """
    Simple rule-based condenser:
      - take 1–2 salient sentences from each chunk (first, middle),
      - score by length-normalized alnum count,
      - dedupe by token Jaccard,
      - trim to limits.
    """
    candidates = []
    for ch in chunks:
        text = ch.get("text","").strip()
        if not text:
            continue
        sents = _sent_split(text)
        if not sents:
            continue
        picks = []
        picks.append(sents[0])
        if len(sents) > 2:
            picks.append(sents[len(sents)//2])
        for s in picks:
            s_clean = re.sub(r'\s+', ' ', s).strip()
            score = sum(c.isalnum() for c in s_clean) / (len(s_clean) + 1e-6)
            candidates.append((s_clean, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    bullets: List[str] = []
    seen = []
    for s, _ in candidates:
        tok = set(re.findall(r"[A-Za-z0-9]+", s.lower()))
        if any(_jaccard(tok, t) >= dedupe_threshold for t in seen):
            continue
        if len(s) > max_chars_per_bullet:
            s = s[:max_chars_per_bullet-1].rstrip() + "…"
        bullets.append("• " + s)
        seen.append(tok)
        if len(bullets) >= max_bullets:
            break

    if not bullets:
        bullets = ["• Key facts not found in local corpus."]
    return bullets
