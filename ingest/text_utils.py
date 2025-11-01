import re
from typing import List, Tuple
from unidecode import unidecode
import nltk

_SENT_SPLIT = nltk.data.load("tokenizers/punkt/english.pickle")

def html_to_text(html: str) -> str:
    # expect pre-cleaned bs4 get_text(); this is a final pass
    text = unidecode(html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text: str) -> List[str]:
    return _SENT_SPLIT.tokenize(text)

def chunk_by_tokens(sentences: List[str], max_chars: int = 1400, overlap_chars: int = 200) -> List[str]:
    """
    Simple char-based packing (robust for MiniLM). Aim ~600 tokens ≈ 1200–1500 chars.
    """
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) + 1 <= max_chars:
            cur.append(s); cur_len += len(s) + 1
        else:
            if cur:
                chunks.append(" ".join(cur))
                # start next with an overlap tail
                tail = " ".join(" ".join(cur)[-overlap_chars:].split(".")[-1:]).strip()
                cur, cur_len = ([tail] if tail else []), len(tail)
            # add current sentence (might exceed if single long, that’s fine)
            cur.append(s); cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    # filter tiny chunks
    return [c for c in chunks if sum(ch.isalnum() for ch in c) >= 200]
