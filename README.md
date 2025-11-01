# UGTA Retrieval Helpers (Hybrid Search + Notes)

This folder contains the **next step** in your pipeline:
1) `hybrid_search()` → BM25 (Whoosh) + Semantic (Qdrant/MiniLM) → MMR diversify → (optional cross-encoder) → top-k chunks
2) `summarize_to_notes()` → compress selected chunks into concise HelpfulNotes (bullets) to feed Gemini Call #2

## Prereqs
- Qdrant running on `localhost:6333` with collection `books_corpus` (built earlier)
- Whoosh index at `data/whoosh_index` (built earlier)
- Python deps: `qdrant-client`, `sentence-transformers`, `whoosh`, `numpy`

## Quick run
```bash
python run_helpful_notes.py --query "BFS level order queue frontier" --kfinal 10
```

## Files
- `retrieval/mmr.py` — MMR selection
- `retrieval/hybrid_search.py` — Hybrid retrieval function
- `retrieval/summarize.py` — Simple rule-based condenser
- `run_helpful_notes.py` — CLI runner

## Next
- Wire the output bullets into your **Gemini Call #2** prompt as **HelpfulNotes**.
- Later you can swap `summarize_to_notes` with an LLM-based condenser if you prefer.
