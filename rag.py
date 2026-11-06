# Minimal RAG: local embeddings (sentence-transformers) + DeepSeek for answers.
# Setup: put DEEPSEEK_API_KEY=... in a .env file next to this script,
# and drop .txt / .pdf files into the docs/ folder.
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

DOCS_DIR = HERE / "docs"
INDEX_FILE = HERE / "index.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = embedder.tokenizer

# Chunk size is measured in the embedding model's own tokens, not words, so a
# chunk can never exceed what the model reads (MiniLM: 256 incl. [CLS]/[SEP]).
CHUNK_TOKENS = 200
OVERLAP_TOKENS = 40
TOP_K = 6  # chunks sent to the LLM; 6 beat 4 on eval.py (multi-section hit 40% -> 60%)
assert CHUNK_TOKENS + 2 <= embedder.max_seq_length


def n_tokens(text):
    """Tokens the embedder sees for this text, including [CLS] and [SEP]."""
    return len(tokenizer(text)["input_ids"])


def chunk_spans(words, max_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    """(start, end) word ranges, each at most max_tokens, overlapping by ~overlap."""
    # The tokenizer splits on spaces before splitting words into pieces,
    # so per-word token counts add up exactly to the chunk's token count.
    cost = [len(tokenizer.tokenize(w)) for w in words]
    spans, start = [], 0
    while start < len(words):
        end, total = start, 0
        while end < len(words) and total + cost[end] <= max_tokens:
            total += cost[end]
            end += 1
        end = max(end, start + 1)          # a single giant "word" still makes progress
        spans.append((start, end))
        if end == len(words):
            break
        # Step back so the next chunk repeats ~`overlap` tokens of this one.
        back, carried = end, 0
        while back > start + 1 and carried + cost[back - 1] <= overlap:
            back -= 1
            carried += cost[back]
        start = back
    return spans


def doc_files():
    return sorted(p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in (".txt", ".pdf"))


def load(path):
    """Words of the file, and the page each word is on (None for text files)."""
    if path.suffix.lower() != ".pdf":
        words = path.read_text(encoding="utf-8").split()
        return words, [None] * len(words)
    pages = PdfReader(path).pages
    words, page_of, empty = [], [], 0
    for n, page in enumerate(pages, 1):
        w = (page.extract_text() or "").split()
        if not w:
            empty += 1
        words += w
        page_of += [n] * len(w)
    if empty:
        print(f"warning: {path.name}: {empty}/{len(pages)} pages have no text "
              f"(scanned images need OCR first)", file=sys.stderr)
    return words, page_of


def label(name, first, last):
    """'TP3.pdf p.3', or 'TP3.pdf p.3-4' for a chunk that runs across a page break."""
    if first is None:
        return name
    return f"{name} p.{first}" if first == last else f"{name} p.{first}-{last}"


# --- Indexing (cached; rebuilt when a doc is added, removed or changed) ---
def build_index():
    files = doc_files()
    if not files:
        sys.exit(f"No .txt or .pdf files found in {DOCS_DIR}")
    mtimes = {str(p): p.stat().st_mtime for p in files}
    settings = {"model": EMBED_MODEL, "chunk_tokens": CHUNK_TOKENS,
                "overlap_tokens": OVERLAP_TOKENS, "across_pages": True}
    if INDEX_FILE.exists():
        cached = pickle.loads(INDEX_FILE.read_bytes())
        if cached["mtimes"] == mtimes and cached.get("settings") == settings:
            return cached["chunks"], cached["sources"], cached["vecs"]

    print(f"Indexing {len(files)} file(s)...", file=sys.stderr)
    chunks, sources = [], []
    for path in files:
        # Chunk the whole file as one stream so sections split by a page break
        # stay together; the label records every page a chunk covers.
        words, page_of = load(path)
        for start, end in chunk_spans(words):
            chunks.append(" ".join(words[start:end]))
            sources.append(label(path.name, page_of[start], page_of[end - 1]))
    if not chunks:
        sys.exit("No text could be extracted from the documents.")
    vecs = embedder.encode(chunks, normalize_embeddings=True)  # shape (N, d)
    INDEX_FILE.write_bytes(pickle.dumps({
        "mtimes": mtimes, "model": EMBED_MODEL, "settings": settings,
        "max_seq_length": embedder.max_seq_length,
        "chunks": chunks, "sources": sources, "vecs": vecs,
        "tokens": [n_tokens(c) for c in chunks]}))
    return chunks, sources, vecs


chunks, sources, vecs = build_index()
_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                         base_url="https://api.deepseek.com")
    return _client


# --- Querying ---
def retrieve(question, k=TOP_K):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    scores = vecs @ q                      # cosine similarity (vectors normalized)
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def ask(question, k=TOP_K, show=False):
    hits = retrieve(question, k)
    if show:
        print("--- retrieved chunks ---")
        for i, score in hits:
            print(f"[{i}] score={score:.2f} ({sources[i]}) {chunks[i][:150]}...")
        print("------------------------\n")
    context = "\n\n".join(f"[{i}] ({sources[i]}) {chunks[i]}" for i, _ in hits)
    prompt = (f"Answer using only the context below, and cite the chunk numbers you used. "
              f"If any chunk contains the answer, give it directly, even if the chunk "
              f"words it differently from the question. If no chunk addresses the "
              f"question, reply with only: \"That isn't in the document.\" Never add "
              f"that sentence to an answer.\n\n"
              f"{context}\n\nQuestion: {question}")
    resp = get_client().chat.completions.create(
        model=CHAT_MODEL, max_tokens=800,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY is not set. Add it to .env (see .env.example).")
    args = sys.argv[1:]
    show = "--show" in args
    question = " ".join(a for a in args if a != "--show") or "What is the vacation policy?"
    print(ask(question, show=show))
