"""
Runner that produces HelpfulNotes for a user query using hybrid search.
Assumes you have already built:
  - Qdrant collection "books_corpus" with MiniLM embeddings
  - Whoosh index under data/whoosh_index

Usage:
  python run_helpful_notes.py --query "BFS level order" --kfinal 10
"""
import argparse
from typing import List
from retrieval.hybrid_search import hybrid_search
from retrieval.summarize import summarize_to_notes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True, help="User/topic query")
    ap.add_argument("--kfinal", type=int, default=10, help="final top-k chunks")
    ap.add_argument("--mmr_k", type=int, default=20, help="MMR pool size")
    ap.add_argument("--lambda_mmr", type=float, default=0.6, help="MMR lambda (relevance weight)")
    ap.add_argument("--use_cross_encoder", action="store_true", help="Use cross-encoder for reranking")
    args = ap.parse_args()

    queries: List[str] = [args.query]

    chunks = hybrid_search(
        queries=queries,
        k_mmr=args.mmr_k,
        lambda_mmr=args.lambda_mmr,
        k_final=args.kfinal,
        use_cross_encoder=args.use_cross_encoder
    )
    print(f"[hybrid_search] got {len(chunks)} chunks")

    notes = summarize_to_notes(chunks, max_bullets=12, max_chars_per_bullet=220)
    print("\n=== HelpfulNotes ===")
    for b in notes:
        print(b)

if __name__ == "__main__":
    main()
