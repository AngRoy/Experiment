from pathlib import Path
from typing import Dict, List
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from text_utils import html_to_text, split_sentences, chunk_by_tokens
from tqdm import tqdm

def parse_epub(epub_path: Path) -> Dict:
    book = epub.read_epub(str(epub_path))
    # metadata
    def first(md, key): 
        vals = book.get_metadata('DC', key)
        return vals[0][0] if vals else None
    meta = {
        "doc_id": epub_path.name,
        "title": first('DC','title') or epub_path.stem,
        "author": first('DC','creator') or "",
        "lang": first('DC','language') or "en",
    }
    chapters: List[Dict] = []
    idx = 0
    for item in book.get_items():
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        # remove scripts/styles/nav
        for tag in soup(["script","style","nav","header","footer"]):
            tag.decompose()
        # best-effort headings
        chapter = None
        for h in soup.find_all(['h1','h2','h3']):
            chapter = h.get_text(strip=True); break
        text = soup.get_text(" ", strip=True)
        text = html_to_text(text)
        if not text or sum(ch.isalnum() for ch in text) < 500:
            continue
        chapters.append({
            "chapter": chapter or f"section-{idx:04d}",
            "section": chapter or "",
            "text": text
        })
        idx += 1
    return {"meta": meta, "chapters": chapters}

def make_chunks(entry: Dict) -> List[Dict]:
    meta = entry["meta"]
    out = []
    pos = 0
    for ch in entry["chapters"]:
        sents = split_sentences(ch["text"])
        pieces = chunk_by_tokens(sents, max_chars=1400, overlap_chars=200)
        for j, piece in enumerate(pieces):
            start = ch["text"].find(piece[:60])  # approximate
            if start < 0: start = pos
            end = start + len(piece)
            out.append({
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "author": meta["author"],
                "lang": meta["lang"],
                "chapter": ch["chapter"],
                "section": ch["section"],
                "chunk_id": f"{meta['doc_id']}#{ch['chapter']}#{j:04d}",
                "start_char": start,
                "end_char": end,
                "text": piece
            })
        pos += len(ch["text"])
    return out

if __name__ == "__main__":
    epub_dir = Path("data/epubs")
    all_chunks = []
    for fp in tqdm(sorted(epub_dir.glob("*.epub"))):
        entry = parse_epub(fp)
        chunks = make_chunks(entry)
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
