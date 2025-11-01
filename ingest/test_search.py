from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from whoosh import index
from whoosh.qparser import MultifieldParser

def sem_topk(query, k=5):
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qv = model.encode([query], normalize_embeddings=True)[0]
    hits = client.search(
        collection_name="books_corpus",
        query_vector=qv.tolist(),
        limit=k,
        with_payload=True
    )
    return [(h.payload["chunk_id"], h.score, h.payload["text"][:120]) for h in hits]

def bm25_topk(query, k=5):
    ix = index.open_dir("data/whoosh_index")
    qp = MultifieldParser(["title","text"], schema=ix.schema)
    q = qp.parse(query)
    with ix.searcher() as s:
        res = s.search(q, limit=k)
        return [(r["chunk_id"], r.score, r["text"][:120]) for r in res]

if __name__ == "__main__":
    q = "BFS level order queue frontier"
    print("BM25:", bm25_topk(q))
    print("SEM :", sem_topk(q))
