from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from ingest_epub import parse_epub, make_chunks
from pathlib import Path
from tqdm import tqdm

COLLECTION = "books_corpus"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d

def ensure_collection(client: QdrantClient, dim: int = 384):
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def embed_and_upsert(epub_dir="data/epubs"):
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(EMB_MODEL)
    ensure_collection(client, dim=model.get_sentence_embedding_dimension())

    points = []
    pid = 0
    for fp in tqdm(sorted(Path(epub_dir).glob("*.epub"))):
        entry = parse_epub(fp)
        chunks = make_chunks(entry)
        texts = [c["text"] for c in chunks]
        vecs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        for c, v in zip(chunks, vecs):
            points.append({
                "id": pid,
                "vector": v.tolist(),
                "payload": c
            })
            pid += 1
            if len(points) >= 2048:
                client.upsert(collection_name=COLLECTION, points=points)
                points = []
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    print("Upsert complete.")

if __name__ == "__main__":
    embed_and_upsert()
