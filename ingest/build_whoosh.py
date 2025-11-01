from whoosh import index
from whoosh.fields import Schema, ID, TEXT
from whoosh.analysis import StemmingAnalyzer
from pathlib import Path
from ingest_epub import parse_epub, make_chunks
import os, shutil
from tqdm import tqdm

INDEX_DIR = "data/whoosh_index"

def build_index(epub_dir="data/epubs"):
    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        doc_id=ID(stored=True),
        title=TEXT(stored=True),
        text=TEXT(analyzer=StemmingAnalyzer(), stored=True)
    )
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR, exist_ok=True)
    ix = index.create_in(INDEX_DIR, schema)
    writer = ix.writer(limitmb=512)
    for fp in tqdm(sorted(Path(epub_dir).glob("*.epub"))):
        entry = parse_epub(fp)
        chunks = make_chunks(entry)
        for c in chunks:
            writer.add_document(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                title=c["title"],
                text=c["text"]
            )
    writer.commit()
    print("Whoosh index built.")

if __name__ == "__main__":
    build_index()
