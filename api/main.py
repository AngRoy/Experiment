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
from .llm_gateway import normalize_task, generate_lesson
from .media.pipeline import render_assets_for_lesson
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
    Convenience endpoint: chat → TaskSpec → HelpfulNotes → LessonDraft
    """
    try:
        # 1) normalize
        ts = normalize_task(req.chat, defaults={"language": "en"} , model=req.model)
        task = TaskSpec(**ts)
        # derive simple queries
        queries = []
        if task.topic: queries.append(task.topic)
        queries.extend(task.keywords[:5])
        if not queries:
            queries = [req.chat]

        # 2) helpful notes (books only; web booster omitted here by design)
        chunks = hybrid_search(queries=queries, k_final=10, k_mmr=20, lambda_mmr=0.6)
        notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)

        # 3) generate
        lesson = generate_lesson(task_spec=task.dict(), helpful_notes=notes, model=req.model)
        return LessonDraft(**lesson)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lesson_rendered", response_model=LessonWithAssets)
def api_full_lesson_rendered(req: FullLessonRequest, request: Request):
    """
    chat → TaskSpec → HelpfulNotes → LessonDraft → (render Mermaid + Images)
    Images generated in parallel, capped at 5 concurrent jobs.
    """
    try:
        # 1) normalize (Gemini #1)
        ts = normalize_task(req.chat, defaults={"language": "en"} , model=req.model)
        task = TaskSpec(**ts)
        queries = []
        if task.topic: queries.append(task.topic)
        queries.extend(task.keywords[:5])
        if not queries:
            queries = [req.chat]

        # 2) helpful notes (hybrid retrieval → bullets)
        chunks = hybrid_search(queries=queries, k_final=10, k_mmr=20, lambda_mmr=0.6)
        notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)

        # 3) lesson draft (Gemini #2)
        lesson = generate_lesson(task_spec=task.dict(), helpful_notes=notes, model=req.model)

        # 4) assets (Mermaid + Images; images in parallel max 5)
        run_id = str(uuid.uuid4())[:8]
        out_root = os.path.join("artifacts", run_id)
        enriched = render_assets_for_lesson(lesson, out_root=out_root, image_concurrency=5)

        # 5) add public URLs for frontend
        base = str(request.base_url).rstrip("/")
        for seg in enriched.get("segments", []):
            p = seg.get("diagram_path")
            if p:
                seg["diagram_url"] = f"{base}/{p.replace('\\', '/')}"
            ip = seg.get("image_path")
            if ip:
                seg["image_url"] = f"{base}/{ip.replace('\\', '/')}"

        # 6) shape response
        segs = [EnrichedLessonSegment(**seg) for seg in enriched.get("segments", [])]
        return LessonWithAssets(
            title=enriched.get("title",""),
            segments=segs,
            narration=enriched.get("narration"),
            artifacts_root=out_root
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
