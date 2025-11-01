# UGTA API (FastAPI)

Endpoints to support your two-call Gemini pipeline.

## Setup
1. `export GEMINI_API_KEY=...`
2. Ensure Qdrant and Whoosh indexes are built (see retrieval README).
3. Install deps:
```bash
pip install -r requirements.txt
```
4. Run:
```bash
uvicorn api.main:app --reload --port 8000
```

## Endpoints

### POST /normalize
Body:
```json
{ "chat": "explain bfs with a diagram", "defaults": {"language":"en"} }
```
Returns `TaskSpec` JSON.

### POST /helpful-notes
Body:
```json
{ "queries": ["Breadth-First Search", "queue frontier bfs"], "kfinal": 10 }
```
Returns `{ "chunks": [...], "notes": ["• ..."] }`

### POST /generate
Body:
```json
{
  "task_spec": {
    "topic":"Breadth-First Search",
    "audience":"undergrad",
    "language":"en",
    "difficulty":"intro",
    "outputs":["text","diagram","image"],
    "keywords":["queue","visited set","level order"],
    "image_ideas":["graph colored by BFS levels"]
  },
  "helpful_notes": ["• BFS explores nodes level-by-level ..."]
}
```
Returns `LessonDraft` JSON (text + optional mermaid/image_prompt/narration).

### POST /lesson
Body:
```json
{ "chat": "teach bfs with a diagram and an image" }
```
Runs: normalize → helpful-notes → generate, then returns `LessonDraft`.

## Notes
- Video/audio are generated only if the `outputs` in TaskSpec request them.
- No visible citations; HelpfulNotes are boost-only.
