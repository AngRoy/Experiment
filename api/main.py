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

# LLM fallbacks + mermaid renderer
from .llm_gateway import (
    normalize_task, generate_lesson,
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

# Serve generated files
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
        lesson = generate_lesson(task_spec=req.task_spec.dict(), helpful_notes=req.helpful_notes, model=req.model)
        return LessonDraft(**lesson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lesson", response_model=LessonDraft)
def api_full_lesson(req: FullLessonRequest):
    """
    chat → TaskSpec → HelpfulNotes → LessonDraft (JSON only; no file IO)
    Guarantees: at least one Mermaid + at least one image_prompt (via LLM fallback if missing).
    """
    try:
        # 1) normalize (Gemini #1)
        ts = normalize_task(req.chat, defaults={"language": "en"}, model=req.model)
        task = TaskSpec(**ts)

        # simple queries
        queries = []
        if task.topic:
            queries.append(task.topic)
        queries.extend(task.keywords[:5])
        if not queries:
            queries = [req.chat]

        # 2) helpful notes
        chunks = hybrid_search(queries=queries, k_final=10, k_mmr=20, lambda_mmr=0.6)
        notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)

        # 3) lesson (Gemini #2)
        lesson = generate_lesson(task_spec=task.dict(), helpful_notes=notes, model=req.model)

        # 4) ensure ≥1 diagram + ≥1 image_prompt via LLM fallback
        segs = list(lesson.get("segments", []))
        has_mermaid = any(isinstance(s.get("mermaid"), str) and s["mermaid"].strip() for s in segs)
        has_image   = any(isinstance(s.get("image_prompt"), str) and s["image_prompt"].strip() for s in segs)

        if not has_mermaid:
            m = gen_mermaid_snippet(task.dict(), notes, model=req.model)
            segs.append({"text": "Auto-added diagram", "mermaid": m, "image_prompt": None})
        if not has_image:
            ip = gen_image_prompt(task.dict(), notes, model=req.model)
            segs.append({"text": "Auto-added image", "mermaid": None, "image_prompt": ip})

        lesson["segments"] = segs
        return LessonDraft(**lesson)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lesson_rendered", response_model=LessonWithAssets)
def api_full_lesson_rendered(req: FullLessonRequest, request: Request):
    """
    chat → TaskSpec → HelpfulNotes → LessonDraft → render Mermaid + Images
    Images generated in parallel (max 5). Ensures ≥1 diagram & ≥1 image.
    If a Mermaid fails, ask Gemini to repair and retry that diagram once.
    """
    try:
        # 1) normalize (Gemini #1)
        ts = normalize_task(req.chat, defaults={"language": "en"}, model=req.model)
        task = TaskSpec(**ts)

        queries = []
        if task.topic:
            queries.append(task.topic)
        queries.extend(task.keywords[:5])
        if not queries:
            queries = [req.chat]

        # 2) helpful notes
        chunks = hybrid_search(queries=queries, k_final=10, k_mmr=20, lambda_mmr=0.6)
        notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)

        # 3) lesson draft (Gemini #2)
        lesson = generate_lesson(task_spec=task.dict(), helpful_notes=notes, model=req.model)

        # ensure ≥1 diagram + ≥1 image_prompt
        segs = list(lesson.get("segments", []))
        has_mermaid = any(isinstance(s.get("mermaid"), str) and s["mermaid"].strip() for s in segs)
        has_image   = any(isinstance(s.get("image_prompt"), str) and s["image_prompt"].strip() for s in segs)

        if not has_mermaid:
            m = gen_mermaid_snippet(task.dict(), notes, model=req.model)
            segs.append({"text": "Auto-added diagram", "mermaid": m, "image_prompt": None})
        if not has_image:
            ip = gen_image_prompt(task.dict(), notes, model=req.model)
            segs.append({"text": "Auto-added image", "mermaid": None, "image_prompt": ip})

        lesson["segments"] = segs

        # 4) render assets
        run_id = str(uuid.uuid4())[:8]
        out_root = os.path.join("artifacts", run_id)
        enriched = render_assets_for_lesson(lesson, out_root=out_root, image_concurrency=5)

        # 5) repair failed diagrams once (optional)
        for i, seg in enumerate(enriched.get("segments", [])):
            if isinstance(seg.get("mermaid"), str) and seg["mermaid"].strip():
                if not seg.get("diagram_path"):
                    fixed = repair_mermaid(seg["mermaid"], error_log=None, topic=lesson.get("title"), model=req.model)
                    if fixed and fixed.strip() != seg["mermaid"].strip():
                        seg["mermaid"] = fixed
                        ddir = os.path.join(out_root, "diagrams")
                        os.makedirs(ddir, exist_ok=True)
                        dpath = os.path.join(ddir, f"diagram_{i}.png")
                        ok = render_mermaid(fixed, dpath)
                        seg["diagram_path"] = dpath if ok else ""

        # 6) stitch URLs
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
