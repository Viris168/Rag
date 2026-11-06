# Show what's stored in index.pkl (built by rag.py).
#   python show_index.py            -> summary + one line per chunk
#   python show_index.py 3          -> full text and vector of chunk 3
#   python show_index.py 3 --tokens -> ...plus the tokens the model sees
#   python show_index.py --find wal -> chunks containing "wal"
#   python show_index.py --check    -> re-count tokens, flag any chunk that would be cut off
import pickle
import sys
from pathlib import Path

import numpy as np

INDEX_FILE = Path(__file__).parent / "index.pkl"
if not INDEX_FILE.exists():
    sys.exit("index.pkl not found - run rag.py once to build it.")

idx = pickle.loads(INDEX_FILE.read_bytes())
chunks, sources, vecs = idx["chunks"], idx["sources"], idx["vecs"]
tokens = idx.get("tokens") or ["?"] * len(chunks)
limit = idx.get("max_seq_length", 256)
args = sys.argv[1:]


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(f"sentence-transformers/{idx['model']}")


if args and args[0].isdigit():
    i = int(args[0])
    print(f"chunk [{i}]  source: {sources[i]}  words: {len(chunks[i].split())}  "
          f"tokens: {tokens[i]}/{limit}\n")
    print(chunks[i])
    np.set_printoptions(precision=3, suppress=True)
    print(f"\nvector ({vecs[i].shape[0]} numbers), first 10: {vecs[i][:10]}")
    if "--tokens" in args:
        tok = load_tokenizer()
        print("\ntokens the model sees:\n" + " ".join(tok.tokenize(chunks[i])))
elif args and args[0] == "--find":
    term = " ".join(args[1:]).lower()
    for i, c in enumerate(chunks):
        if term in c.lower():
            print(f"[{i}] ({sources[i]}) {c[:120]}...")
elif args and args[0] == "--check":
    tok = load_tokenizer()
    counts = [len(tok(c)["input_ids"]) for c in chunks]
    over = [i for i, n in enumerate(counts) if n > limit]
    print(f"limit: {limit} tokens   chunks: {len(chunks)}   "
          f"min/avg/max: {min(counts)}/{sum(counts) // len(counts)}/{max(counts)}")
    for i in over:
        print(f"  TRUNCATED [{i}] {sources[i]}: {counts[i]} tokens, "
              f"{counts[i] - limit} lost from the end")
    print("OK - every chunk fits, nothing is cut off." if not over
          else f"{len(over)} chunk(s) are cut off when embedded.")
    sys.exit(1 if over else 0)
else:
    print(f"embedding model: {idx['model']}   settings: {idx.get('settings', 'old index')}")
    print(f"files: {', '.join(Path(p).name for p in idx['mtimes'])}")
    print(f"chunks: {len(chunks)}   vectors: {vecs.shape[0]} x {vecs.shape[1]}\n")
    for i, (s, c) in enumerate(zip(sources, chunks)):
        flag = " !" if isinstance(tokens[i], int) and tokens[i] > limit else "  "
        print(f"[{i:>3}] {s:<20} {len(c.split()):>4} words {tokens[i]:>4} tok{flag}| {c[:70]}")
