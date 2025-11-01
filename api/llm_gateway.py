import os, re, json
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

DEFAULT_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash-lite")

def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

def _extract_json(text: str) -> Any:
    """
    Robust JSON extraction from model text.
    """
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) fenced ```json ... ```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if fence:
        blk = fence.group(1).strip()
        try:
            return json.loads(blk)
        except Exception:
            last = blk.rfind("}")
            if last != -1:
                try:
                    return json.loads(blk[:last+1])
                except Exception:
                    pass
    # 3) brace-balance from first '{'
    if "{" in text and "}" in text:
        s = text[text.find("{"):]
        depth = 0
        end_idx = None
        for i, ch in enumerate(s):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx is not None:
            candidate = s[:end_idx+1]
            try:
                return json.loads(candidate)
            except Exception:
                last = candidate.rfind("}")
                if last != -1:
                    try:
                        return json.loads(candidate[:last+1])
                    except Exception:
                        pass
    return {"raw": text}

def normalize_task(chat: str, defaults: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> Dict[str, Any]:
    model_name = model or DEFAULT_TEXT_MODEL
    system = (
        "You normalize casual user requests into a compact JSON TaskSpec. "
        "Return ONLY valid JSON (no code fences). Strict keys: topic, audience, language, difficulty, outputs (default ['text','diagram']), "
        "keywords (3-7), image_ideas (1-2). Add 'audio' only if the user asked for voice. Add 'video' only if asked."
    )
    user = f"User message: ```{chat}```\nReturn ONLY valid JSON without code fences. Start with '{{' and end with '}}'."
    client = _client()
    res = client.models.generate_content(
        model=model_name,
        contents=f"{system}\n\n{user}",
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )
    text = res.candidates[0].content.parts[0].text if res.candidates else ""
    data = _extract_json(text)
    if defaults and isinstance(data, dict):
        for k, v in defaults.items():
            data.setdefault(k, v)
    return data

def generate_lesson(task_spec: Dict[str, Any], helpful_notes: List[str], model: Optional[str] = None) -> Dict[str, Any]:
    model_name = model or DEFAULT_TEXT_MODEL
    system = (
        "You create concise, pedagogically sound lessons. Use HelpfulNotes to improve correctness, "
        "but do not cite them. Return ONLY valid JSON (no code fences) with keys: "
        "title, segments[{text, mermaid, image_prompt}], narration? . Include optional fields ONLY if requested."
    )
    notes_block = "\n".join(helpful_notes[:12])
    prompt = (
        f"TaskSpec JSON:\n```json\n{json.dumps(task_spec, ensure_ascii=False)}\n```\n"
        f"HelpfulNotes (optional):\n{notes_block}\n"
        "Produce the lesson now. Return ONLY valid JSON without code fences. Start with '{' and end with '}'."
    )
    client = _client()
    res = client.models.generate_content(
        model=model_name,
        contents=f"{system}\n\n{prompt}",
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )
    text = res.candidates[0].content.parts[0].text if res.candidates else ""
    data = _extract_json(text)
    return data
