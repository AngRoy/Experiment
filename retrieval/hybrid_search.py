from typing import List, Dict, Any, Tuple
import numpy as np

from whoosh import index
from whoosh.qparser import MultifieldParser

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .mmr import mmr_select

# Paths & constants
WHOOSH_INDEX_DIR = "data/whoosh_index"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "books_corpus"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _normalize_scores(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin + 1e-12:
        return [0.5 for _ in vals]
    return [float((v - vmin) / (vmax - vmin)) for v in vals]

def bm25_topk(query: str, k: int = 30) -> List[Tuple[str, float, Dict[str, Any]]]:
    ix = index.open_dir(WHOOSH_INDEX_DIR)
    qp = MultifieldParser(["title", "text"], schema=ix.schema)
    q = qp.parse(query)
    out = []
    with ix.searcher() as s:
        res = s.search(q, limit=k)
        for r in res:
            out.append((r["chunk_id"], float(r.score), {"doc_id": r["doc_id"], "title": r["title"], "text": r["text"]}))
    return out

def semantic_topk(query: str, model: SentenceTransformer, client: QdrantClient, k: int = 30):
    qv = model.encode([query], normalize_embeddings=True)[0]
    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qv.tolist(),
        limit=k,
        with_payload=True
    )
    out = []
    for h in hits:
        payload = dict(h.payload)
        cid = payload.get("chunk_id") or f"{payload.get('doc_id','')}#{payload.get('start_char','?')}"
        out.append((cid, float(h.score), payload))
    return out

def _pool_candidates(bm25_list, sem_list) -> Dict[str, Dict[str, Any]]:
    pooled: Dict[str, Dict[str, Any]] = {}
    for cid, score, payload in bm25_list:
        pooled.setdefault(cid, {"payload": payload, "bm25": -1.0, "sem": -1.0})
        pooled[cid]["bm25"] = max(pooled[cid]["bm25"], float(score))
    for cid, score, payload in sem_list:
        pooled.setdefault(cid, {"payload": payload, "bm25": -1.0, "sem": -1.0})
        pooled[cid]["sem"] = max(pooled[cid]["sem"], float(score))
        if payload and len(payload.get("text","")) > len(pooled[cid]["payload"].get("text","")):
            pooled[cid]["payload"] = payload
    return pooled

def _ensure_text_payload(pooled: Dict[str, Dict[str, Any]]):
    to_drop = []
    for cid, item in pooled.items():
        if not item["payload"].get("text"):
            to_drop.append(cid)
    for cid in to_drop:
        pooled.pop(cid, None)

def hybrid_search(
    queries: List[str],
    topn_bm25: int = 30,
    topm_sem: int = 30,
    k_mmr: int = 20,
    lambda_mmr: float = 0.6,
    k_final: int = 10,
    use_cross_encoder: bool = False,
    cross_encoder_model: str = "BAAI/bge-reranker-base"
) -> List[Dict[str, Any]]:
    """
    Returns a list of up to k_final payload dicts (diverse, high-quality).
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer(EMB_MODEL)

    bm25_all, sem_all = [], []
    for q in queries:
        if not q or not q.strip():
            continue
        bm25_all.extend(bm25_topk(q, k=topn_bm25))
        sem_all.extend(semantic_topk(q, model, client, k=topm_sem))

    pooled = _pool_candidates(bm25_all, sem_all)
    _ensure_text_payload(pooled)
    if not pooled:
        return []

    bm25_scores = [pooled[cid]["bm25"] if pooled[cid]["bm25"] >= 0 else 0.0 for cid in pooled]
    sem_scores  = [pooled[cid]["sem"]  if pooled[cid]["sem"]  >= 0 else 0.0 for cid in pooled]
    bm25_norm = _normalize_scores(bm25_scores)
    sem_norm  = _normalize_scores(sem_scores)

    cids = list(pooled.keys())
    rel = np.array([0.5*bm25_norm[i] + 0.5*sem_norm[i] for i in range(len(cids))], dtype=np.float32)

    texts = [pooled[cid]["payload"]["text"] for cid in cids]
    emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    sel_idx = mmr_select(embeddings=emb, relevance=rel, k=k_mmr, lambda_=lambda_mmr)
    selected = [cids[i] for i in sel_idx]

    if use_cross_encoder:
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(cross_encoder_model)
            main_q = next((q for q in queries if q and q.strip()), "")
            pairs = [(main_q, pooled[cid]["payload"]["text"]) for cid in selected]
            scores = ce.predict(pairs)
            ord_idx = list(np.argsort(scores))[::-1]
            selected = [selected[i] for i in ord_idx]
        except Exception:
            pass

    out_payloads = [pooled[cid]["payload"] for cid in selected[:k_final]]
    return out_payloads
