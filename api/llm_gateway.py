import os, re, json
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
import itertools

DEFAULT_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")

def _client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)

def _extract_json(text: str) -> Any:
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

# ---------- Sanitizers ----------

_MERMAID_FENCE = re.compile(r"```(?:mermaid)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

def _coerce_str(val) -> Optional[str]:
    return val if isinstance(val, str) else None

def _extract_mermaid(val: Any) -> Optional[str]:
    """Return a clean Mermaid string or None."""
    if not isinstance(val, str):
        return None
    txt = val.strip()
    m = _MERMAID_FENCE.search(txt)
    if m:
        txt = m.group(1).strip()
    # Heuristic: keep only the block that starts with 'graph' or 'flowchart'
    lines = [ln.rstrip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return None
    start_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower().lstrip()
        if low.startswith("graph ") or low.startswith("flowchart "):
            start_idx = i
            break
    if start_idx is None:
        # if first line declares graph without leading whitespace, accept
        if lines and (lines[0].lower().startswith(("graph ", "flowchart "))):
            start_idx = 0
        else:
            return None
    cleaned = "\n".join(lines[start_idx:])
    return cleaned if cleaned else None

def _sanitize_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(seg or {})
    raw_text = s.get("text")
    if not isinstance(raw_text, str):
        raw_text = "" if raw_text is None else str(raw_text)

    # If mermaid leaked into text, pull it out
    text_wo_mermaid, mermaid_in_text = _pop_mermaid_from_text(raw_text)

    # text_format
    if s.get("text_format") not in ("md", "plain"):
        s["text_format"] = "md"

    # mermaid: prefer explicit field, else from text
    mer = _extract_mermaid(s.get("mermaid")) or mermaid_in_text
    if isinstance(mer, str):
        mer = _strip_mermaid_brittle_lines(mer)
    s["mermaid"] = mer

    # final text
    s["text"] = text_wo_mermaid

    # image_prompt
    ip = _coerce_str(s.get("image_prompt"))
    s["image_prompt"] = ip.strip() if isinstance(ip, str) else None

    # alt_text
    at = _coerce_str(s.get("alt_text"))
    s["alt_text"] = at.strip() if isinstance(at, str) else None

    # kind inference
    if s.get("mermaid"):
        s["kind"] = "diagram"
    elif s.get("image_prompt"):
        s["kind"] = "image"
    else:
        s["kind"] = "content"
    return s

def sanitize_lesson(lesson: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(lesson, dict):
        return {"title": "", "segments": []}
    title = lesson.get("title") if isinstance(lesson.get("title"), str) else ""
    segs_in = lesson.get("segments") if isinstance(lesson.get("segments"), list) else []
    segs = [_sanitize_segment(s) for s in segs_in]
    out = {"title": title, "segments": segs}
    if isinstance(lesson.get("narration"), str):
        out["narration"] = lesson["narration"]
    return out

def _strip_mermaid_brittle_lines(code: str) -> str:
    """Reduce renderer breakage: drop fragile style/class lines that often error."""
    lines = [ln for ln in (code or "").splitlines()]
    safe = []
    for ln in lines:
        l = ln.strip()
        if l.lower().startswith("style "):    # common breakage
            continue
        if l.lower().startswith("classdef "): # styling often trips CLI
            continue
        safe.append(ln)
    return "\n".join(safe).strip()

def _pop_mermaid_from_text(md: str) -> tuple[str, Optional[str]]:
    """
    Find a mermaid block inside Markdown text and remove it from the text.
    Support both ```mermaid fenced blocks and unfenced blocks starting with 'graph'/'flowchart'.
    Return (clean_text, mermaid or None).
    """
    if not isinstance(md, str) or not md.strip():
        return md or "", None

    # 1) fenced ```mermaid ... ```
    m = re.search(r"```mermaid\s*([\s\S]*?)```", md, flags=re.IGNORECASE)
    if m:
        code = m.group(1).strip()
        code = _strip_mermaid_brittle_lines(code)
        md2 = md[:m.start()] + md[m.end():]
        return md2.strip(), (code if code else None)

    # 2) unfenced: look for a block that starts with graph/flowchart and ends at first blank-line sequence
    lines = md.splitlines()
    start = None
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith("graph ") or low.startswith("flowchart "):
            start = i
            break
    if start is not None:
        # capture until a blank line separator (two consecutive blank lines or end)
        block = []
        for ln in itertools.islice(lines, start, None):
            if ln.strip() == "" and (len(block) > 0 and block[-1].strip() == ""):
                break
            block.append(ln)
        code = "\n".join(block).strip()
        code = _strip_mermaid_brittle_lines(code)
        # remove that block from text
        kept = lines[:start] + lines[start+len(block):]
        md2 = "\n".join(kept).strip()
        return md2, (code if code else None)

    return md, None


# ---------- Normalization / Generation ----------

def normalize_task(chat: str, defaults: Optional[Dict[str, Any]] = None, model: Optional[str] = None) -> Dict[str, Any]:
    model_name = model or DEFAULT_TEXT_MODEL
    system = (
        "You normalize casual user requests into a compact JSON TaskSpec. "
        "Return ONLY valid JSON (no code fences). Strict keys: "
        "topic, audience, language, difficulty, outputs (default ['text','diagram','image']), "
        "keywords (3-7), image_ideas (1-2), text_depth ('very_detailed' by default), "
        "min_diagrams (int 0-10), min_images (int 0-10). "
        "Infer counts from phrasing (e.g., '3 diagrams', 'several images'→3, 'a lot'→3). "
        "If not stated, default min_diagrams=2 and min_images=2 when the respective output is present."
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

    if isinstance(data, dict):
        data.setdefault("outputs", ["text","diagram","image"])
        data.setdefault("text_depth", "very_detailed")
        has_diag = "diagram" in data["outputs"]
        has_img  = "image" in data["outputs"]
        data.setdefault("min_diagrams", 2 if has_diag else 0)
        data.setdefault("min_images",   2 if has_img  else 0)
        # clamp
        try:
            data["min_diagrams"] = max(0, min(int(data.get("min_diagrams", 2)), 10))
        except Exception:
            data["min_diagrams"] = 2 if has_diag else 0
        try:
            data["min_images"] = max(0, min(int(data.get("min_images", 2)), 10))
        except Exception:
            data["min_images"] = 2 if has_img else 0

    return data

def generate_lesson(task_spec: Dict[str, Any], helpful_notes: List[str], model: Optional[str] = None) -> Dict[str, Any]:
    model_name = model or DEFAULT_TEXT_MODEL

    structure_hint = (
        "Use a clearly structured Markdown layout with headings and subsections. "
        "Target outline (adapt to the topic):\n"
        "1. Introduction & Motivation\n"
        "2. What Is X? (Informal → Formal)\n"
        "3. Core Terminology\n"
        "4. Types / Taxonomy\n"
        "5. Representations / Common Forms\n"
        "6. Worked Examples\n"
        "7. Bridge to a Related Concept\n"
        "8. Guided Tutorial Scenarios\n"
        "9. Practice Problems\n"
        "10. Assignments\n"
        "11. Glossary\n"
        "12. Common Pitfalls\n"
        "13. Summary\n"
        "14. Further Reading\n\n"
        "Ensure the Markdown in 'segments[].text' is preserved verbatim."
    )
    graph_hint = (
        "If the topic is about Graphs, prefer this deeper outline: "
        "Introduction & Motivation; What Is a Graph? (Informal → Formal G=(V,E)); "
        "Core Terminology (Vertices, Edges, Incidence, Degree, Adjacency); "
        "Types of Graphs (Simple, Multigraph, Pseudograph, Directed, Undirected, Mixed); "
        "Graph Representations (Adjacency Matrix, Adjacency List, Quick Comparison); "
        "Worked Examples (Social Network, One-Way Streets); Bridge to Minimal (Vertex) Cover; "
        "Guided Tutorial Scenarios; Practice Problems; Assignments; Glossary; Common Pitfalls; Summary; Further Reading."
    )
    system = (
        "You create comprehensive, pedagogically sound lessons. "
        "Use HelpfulNotes to improve correctness, but do not cite them. "
        "Return ONLY valid JSON (no code fences) with keys: "
        "title, segments[{section?, kind(content|diagram|image), text(md), text_format='md', mermaid?, image_prompt?, alt_text?}], narration?. "
        "CRITICAL: Keep Markdown inside 'text' exactly as you generate it; do not escape or convert. "
        "If 'diagram' is in outputs, include at least 2 segments with valid Mermaid (prefer 'flowchart TD' or 'graph LR'); 'mermaid' MUST be a string containing Mermaid code, not true/false. "
        "If 'image' is in outputs, include at least 2 segments with precise schematic image prompts; 'image_prompt' MUST be a string, not true/false. "
        "Diagrams are for explanation; images are aesthetic yet relevant. "
        "Avoid brittle Mermaid 'style' lines unless confident they render. "
        "Keep images schematic (not photorealistic). "
        f"{structure_hint} {graph_hint}"
    )

    notes_block = "\n".join(helpful_notes[:12])
    prompt = (
        f"TaskSpec JSON:\n```json\n{json.dumps(task_spec, ensure_ascii=False)}\n```\n"
        f"HelpfulNotes (optional):\n{notes_block}\n\n"
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
    # sanitize before returning
    return sanitize_lesson(data)

# ---------- LLM fallbacks ----------

def gen_mermaid_snippet(task_spec: Dict[str, Any], helpful_notes: List[str], model: Optional[str] = None) -> str:
    model_name = model or DEFAULT_TEXT_MODEL
    topic = task_spec.get("topic") or "the topic"
    notes = "\n".join(helpful_notes[:8])
    system = (
        "Produce ONLY a valid, small Mermaid diagram that teaches the topic with high relevance. "
        "Prefer 'flowchart TD' or 'graph LR'. No narrative text. No code fences. "
        "Avoid fragile 'style' lines unless necessary."
    )
    user = (
        f"Topic: {topic}\nHelpfulNotes (optional):\n{notes}\n\n"
        "Constraints:\n- Compact and valid Mermaid\n- Simple nodes/edges with brief labels\n- Output Mermaid only"
    )
    client = _client()
    res = client.models.generate_content(
        model=model_name,
        contents=f"{system}\n\n{user}",
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )
    txt = res.candidates[0].content.parts[0].text if res.candidates else ""
    mer = _extract_mermaid(txt) or "flowchart TD\nA[Start]-->B[Concept 1]\nB-->C[Concept 2]\nC-->D[End]"
    return mer

def gen_image_prompt(task_spec: Dict[str, Any], helpful_notes: List[str], model: Optional[str] = None) -> str:
    model_name = model or DEFAULT_TEXT_MODEL
    topic = task_spec.get("topic") or "the topic"
    notes = "\n".join(helpful_notes[:8])
    system = (
        "Return ONLY one line of text: a clean 2D vector schematic prompt (not a photo). "
        "Style: flat, minimal, white background, thin black outlines, limited accent colors, "
        "clear labels/arrows, resolution ~1024x1024. No people or scenery. "
        "It should be aesthetically pleasing but still relevant to the topic."
    )
    user = (
        f"Topic: {topic}\nHelpfulNotes (optional):\n{notes}\n\n"
        "Return one line describing the schematic content, precise and labeled."
    )
    client = _client()
    res = client.models.generate_content(
        model=model_name,
        contents=f"{system}\n\n{user}",
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )
    text = res.candidates[0].content.parts[0].text if res.candidates else ""
    line = " ".join((text or "").strip().split())
    return line or f"clean 2D vector schematic of {topic}, white background, thin black outlines, clear labels"

def repair_mermaid(mermaid_code: str, error_log: Optional[str] = None, topic: Optional[str] = None,
                   model: Optional[str] = None) -> str:
    model_name = model or DEFAULT_TEXT_MODEL
    sysmsg = (
        "You repair Mermaid diagrams. Output ONLY Mermaid code (no fences). "
        "Use 'flowchart TD' or 'graph LR'. Remove fragile 'style' lines. Keep it small and valid."
    )
    user = (
        f"Topic: {topic or ''}\n"
        f"Broken Mermaid:\n{mermaid_code}\n"
        f"Renderer stderr (optional):\n{(error_log or '').strip()}\n\n"
        "Return fixed Mermaid only."
    )
    client = _client()
    res = client.models.generate_content(
        model=model_name,
        contents=f"{sysmsg}\n\n{user}",
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )
    text = res.candidates[0].content.parts[0].text if res.candidates else ""
    return _extract_mermaid(text) or mermaid_code
