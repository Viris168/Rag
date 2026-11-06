# Score the RAG pipeline against eval_set.json.
#   python eval.py                  -> retrieval + refusal + citation scores (calls DeepSeek)
#   python eval.py --retrieval-only -> retrieval scores only, no API calls
#   python eval.py -k 4 -v          -> top-4 retrieval (default rag.TOP_K), print every answer
#
# Retrieval, per answerable question (a "relevant" chunk covers an expected page;
# a chunk can span a page break, e.g. "p.3-4"):
#   hit@k  - a relevant chunk is in the top k. Questions with "need_all" need every
#            expected page in the top k (the answer is spread across sections).
#   hit@1  - the top chunk is relevant.
#   MRR    - mean of 1/rank of the first relevant chunk (0 if none in top k).
#   hit@1 and MRR skip "need_all" questions, where no single page is the answer.
# Refusal accuracy: unanswerable questions should get "not in the document".
# False refusals: answerable questions the model wrongly refused. Reported so that
#   a model that refuses everything can't score 100% on refusal accuracy.
# Citations: of the answered questions, how many cite a chunk from an expected page.
#
# Each question has an optional "kind" (direct / paraphrase / multi / near-miss);
# the summary breaks scores down by kind.
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import rag

HERE = Path(__file__).parent
EVAL_FILE = HERE / "eval_set.json"
DOC = "TP3.pdf"

# Lowercased substrings that mean the model declined to answer.
REFUSAL_PHRASES = [
    "not in the", "isn't in the", "is not in the", "not contain", "doesn't contain",
    "does not contain", "not mentioned", "isn't mentioned", "no information",
    "not provided", "doesn't provide", "does not provide", "not found",
    "cannot answer", "can't answer", "unable to answer", "isn't there",
    "not covered", "doesn't say", "does not say", "doesn't specify",
    "does not specify", "not specified", "no mention", "not state", "n't state",
    "not include", "n't include", "doesn't answer", "does not answer",
]


def is_refusal(answer):
    a = answer.lower().replace("’", "'")
    return any(p in a for p in REFUSAL_PHRASES)


def cited_chunks(answer):
    """Chunk numbers cited as [3], [3, 5], [3][5] or 'chunk 3'."""
    nums = []
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        nums += [int(n) for n in re.findall(r"\d+", group)]
    nums += [int(n) for n in re.findall(r"chunks?\s*\[?(\d+)", answer, re.I)]
    return sorted({n for n in nums if n < len(rag.chunks)})


def pages_of(source):
    """'TP3.pdf p.5' -> {5}, 'TP3.pdf p.3-4' -> {3, 4}, anything else -> empty."""
    name, _, pages = source.rpartition(" p.")
    first, _, last = pages.partition("-")
    if name != DOC or not first.isdigit():
        return set()
    return set(range(int(first), int(last or first) + 1))


def main():
    args = sys.argv[1:]
    k = int(args[args.index("-k") + 1]) if "-k" in args else rag.TOP_K
    retrieval_only = "--retrieval-only" in args
    verbose = "-v" in args
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    answerable, unanswerable = data["answerable"], data["unanswerable"]
    # by_kind[kind][metric] = [score, count]
    by_kind = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    def tally(item, metric, value):
        for kind in (item.get("kind", "direct"), "ALL"):
            by_kind[kind][metric][0] += value
            by_kind[kind][metric][1] += 1

    print(f"=== Retrieval (top {k}) ===")
    for item in answerable:
        want = item["pages"]
        got = [pages_of(rag.sources[i]) for i, _ in rag.retrieve(item["q"], k)]
        if item.get("need_all"):
            hit = set(want) <= set().union(*got)
        else:
            hit = any(g & set(want) for g in got)
        rank = next((r for r, g in enumerate(got, 1) if g & set(want)), None)
        tally(item, "hit@k", hit)
        if not item.get("need_all"):     # rank of "the" right page is meaningless here
            tally(item, "hit@1", rank == 1)
            tally(item, "mrr", 1 / rank if rank else 0)
        tag = "HIT " if hit else "MISS"
        shown = ", ".join(f"{min(g)}-{max(g)}" if len(g) > 1 else str(min(g, default="?"))
                          for g in got)
        print(f"{tag} want p.{want}{' (all)' if item.get('need_all') else ''} "
              f"got p.[{shown}]  {item['q']}")

    if not retrieval_only:
        if not rag.os.environ.get("DEEPSEEK_API_KEY"):
            sys.exit("DEEPSEEK_API_KEY is not set. Use --retrieval-only or add it to .env.")

        print("\n=== Refusal (unanswerable questions) ===")
        for item in unanswerable:
            answer = rag.ask(item["q"], k)
            ok = is_refusal(answer)
            tally(item, "refusal", ok)
            print(f"{'OK  ' if ok else 'FAIL'} {item['q']}")
            if verbose or not ok:
                print("     " + answer.replace("\n", "\n     "))

        print("\n=== Answers (answerable questions) ===")
        for item in answerable:
            answer = rag.ask(item["q"], k)
            refused = is_refusal(answer)
            tally(item, "false_refusal", refused)
            cited = cited_chunks(answer)
            cited_pages = sorted(set().union(*(pages_of(rag.sources[i]) for i in cited)))
            cite_ok = any(p in item["pages"] for p in cited_pages)
            if not refused:
                tally(item, "citation", cite_ok)
            problem = ("REFUSED" if refused else
                       "NO CITE" if not cited else
                       "BAD CITE" if not cite_ok else None)
            if verbose or problem:
                print(f"{problem or 'ok'}  {item['q']}")
                print(f"     expected: {item['a']}  (p.{item['pages']})")
                print(f"     cited: {cited} -> p.{cited_pages}")
                print("     " + answer.replace("\n", "\n     "))

    print("\n=== Summary ===")
    labels = [("hit@k", f"hit@{k}"), ("hit@1", "hit@1"), ("mrr", "MRR"),
              ("refusal", "refusal acc"), ("false_refusal", "false refusals"),
              ("citation", "citations ok")]
    kinds = ["ALL"] + sorted(kd for kd in by_kind if kd != "ALL")
    print(f"{'':<16}" + "".join(f"{kd:>14}" for kd in kinds))
    for metric, label in labels:
        cells = []
        for kd in kinds:
            score, count = by_kind[kd][metric] if metric in by_kind[kd] else (0, 0)
            if not count:
                cells.append("-")
            elif metric == "mrr":
                cells.append(f"{score / count:.2f}")
            else:
                cells.append(f"{round(score)}/{count} {100 * score / count:.0f}%")
        if any(c != "-" for c in cells):
            print(f"{label:<16}" + "".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
