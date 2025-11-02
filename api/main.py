from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import (
    NormalizeRequest, TaskSpec, HelpfulNotesRequest, HelpfulNotesResponse,
    ChunkPayload, GenerateRequest, LessonDraft, FullLessonRequest,
    LessonWithAssets, EnrichedLessonSegment
)
from retrieval.hybrid_search import hybrid_search
from retrieval.summarize import summarize_to_notes

from .llm_gateway import (
    normalize_task, generate_lesson, sanitize_lesson,
    gen_mermaid_snippet, gen_image_prompt, repair_mermaid
)
from .media.pipeline import render_assets_for_lesson
from .media.mermaid import render_mermaid

import os, uuid

app = FastAPI(title="UGTA Pipeline API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("artifacts", exist_ok=True)
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")


@app.post("/normalize", response_model=TaskSpec)
def api_normalize(req: NormalizeRequest):
    try:
        data = normalize_task(req.chat, defaults=req.defaults or {})
        return TaskSpec(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/helpful-notes", response_model=HelpfulNotesResponse)
def api_helpful_notes(req: HelpfulNotesRequest):
    try:
        chunks = hybrid_search(
            queries=req.queries,
            k_mmr=req.mmr_k,
            lambda_mmr=req.lambda_mmr,
            k_final=req.kfinal,
            use_cross_encoder=req.use_cross_encoder
        )
        notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)
        return HelpfulNotesResponse(
            chunks=[ChunkPayload(**c) for c in chunks],
            notes=notes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=LessonDraft)
def api_generate(req: GenerateRequest):
    try:
        # sanitize inside generate_lesson already, but keep consistent
        lesson = generate_lesson(task_spec=req.task_spec.dict(), helpful_notes=req.helpful_notes, model=req.model)
        lesson = sanitize_lesson(lesson)
        return LessonDraft(**lesson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _top_up_assets_with_llm(lesson: dict, task: TaskSpec, notes: list, model: str | None):
    # sanitize first so booleans become None and don’t break counters
    lesson = sanitize_lesson(lesson)
    segs = list(lesson.get("segments", []))

    count_merm = sum(1 for s in segs if isinstance(s.get("mermaid"), str) and s["mermaid"].strip())
    count_img  = sum(1 for s in segs if isinstance(s.get("image_prompt"), str) and s["image_prompt"].strip())

    target_merm = max(0, int(getattr(task, "min_diagrams", 2)))
    target_img  = max(0, int(getattr(task, "min_images", 2)))

    while count_merm < target_merm:
        m = gen_mermaid_snippet(task.dict(), notes, model=model)
        segs.append({
            "section": "Auto-added Diagram",
            "kind": "diagram",
            "text": "Diagram for the current topic.",
            "text_format": "md",
            "mermaid": m,
            "image_prompt": None,
            "alt_text": "Diagram explaining a key concept of the topic."
        })
        count_merm += 1

    while count_img < target_img:
        ip = gen_image_prompt(task.dict(), notes, model=model)
        segs.append({
            "section": "Auto-added Image",
            "kind": "image",
            "text": "Illustrative image to support understanding.",
            "text_format": "md",
            "mermaid": None,
            "image_prompt": ip,
            "alt_text": "Schematic image for the topic."
        })
        count_img += 1

    # sanitize again (just in case)
    lesson["segments"] = segs
    return sanitize_lesson(lesson)


@app.post("/lesson", response_model=LessonDraft)
def api_full_lesson(req: FullLessonRequest):
    """
    chat → TaskSpec → HelpfulNotes → LessonDraft (JSON only)
    Guarantees: ≥min_diagrams Mermaid + ≥min_images image prompts.
    """
    try:
        # 1) normalize (Gemini #1)
        ts = normalize_task(req.chat, defaults={"language": "en"}, model=req.model)
        task = TaskSpec(**ts)

        # 2) helpful notes
        queries = [task.topic] if task.topic else []
        queries.extend(task.keywords[:5])
        if not queries:
            queries = [req.chat]
        chunks = hybrid_search(queries=queries, k_final=10, k_mmr=20, lambda_mmr=0.6)
        notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)

        # 3) lesson (Gemini #2)
        lesson = generate_lesson(task_spec=task.dict(), helpful_notes=notes, model=req.model)

        # 4) ensure targets and sanitize
        lesson = _top_up_assets_with_llm(lesson, task, notes, req.model)

        return LessonDraft(**lesson)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lesson_rendered", response_model=LessonWithAssets)
def api_full_lesson_rendered(req: FullLessonRequest, request: Request):
    """
    chat → TaskSpec → HelpfulNotes → LessonDraft → render Mermaid + Images
    Images generated in parallel (max 5). Ensures ≥ min_diagrams & ≥ min_images.
    Repairs broken Mermaid once if needed.
    """
    try:
        # 1) normalize
        ts = normalize_task(req.chat, defaults={"language": "en"}, model=req.model)
        task = TaskSpec(**ts)

        # 2) helpful notes
        queries = [task.topic] if task.topic else []
        queries.extend(task.keywords[:5])
        if not queries:
            queries = [req.chat]
        chunks = hybrid_search(queries=queries, k_final=10, k_mmr=20, lambda_mmr=0.6)
        notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)

        # 3) lesson draft
        lesson = generate_lesson(task_spec=task.dict(), helpful_notes=notes, model=req.model)
        lesson = _top_up_assets_with_llm(lesson, task, notes, req.model)

        # 4) render assets
        run_id = str(uuid.uuid4())[:8]
        out_root = os.path.join("artifacts", run_id)
        enriched = render_assets_for_lesson(lesson, out_root=out_root, image_concurrency=5)

        # 5) repair failed diagrams once
        for i, seg in enumerate(enriched.get("segments", [])):
            if isinstance(seg.get("mermaid"), str) and seg["mermaid"].strip():
                if not seg.get("diagram_path"):
                    fixed = repair_mermaid(seg["mermaid"], error_log=None, topic=lesson.get("title"), model=req.model)
                    if fixed and fixed.strip() != seg["mermaid"].strip():
                        seg["mermaid"] = fixed
                        ddir = os.path.join(out_root, "diagrams"); os.makedirs(ddir, exist_ok=True)
                        dpath = os.path.join(ddir, f"diagram_{i}.png")
                        ok = render_mermaid(fixed, dpath)
                        seg["diagram_path"] = dpath if ok else ""

        # 6) add public URLs
        base = str(request.base_url).rstrip("/")
        for seg in enriched.get("segments", []):
            p = seg.get("diagram_path")
            if p:
                seg["diagram_url"] = f"{base}/{p.replace('\\', '/')}"
            ip = seg.get("image_path")
            if ip:
                seg["image_url"] = f"{base}/{ip.replace('\\', '/')}"

        segs_out = [EnrichedLessonSegment(**seg) for seg in enriched.get("segments", [])]
        return LessonWithAssets(
            title=enriched.get("title", ""),
            segments=segs_out,
            narration=enriched.get("narration"),
            artifacts_root=out_root
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
