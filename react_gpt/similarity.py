"""Shared lexical component of the paper's hybrid similarity metric."""

import re

STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'that','this','these','those','it','its','as','not','no','nor','so',
    'yet','both','either','each','few','more','most','other','some','such',
    'than','then','there','when','where','which','who','whom','why','how',
    'all','any','can','into','if','up','out','about','what','per','i','we',
    'you','he','she','they','them','us','our','your'
}

def tokenise(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 1]

def ngrams(tokens: list, n: int) -> set:
    return set(zip(*[tokens[i:] for i in range(n)]))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0

def overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

def _lexical_similarity(t1: str, t2: str) -> float:
    tok1, tok2 = tokenise(t1), tokenise(t2)
    j_score  = sum(jaccard(ngrams(tok1, n), ngrams(tok2, n)) for n in (1, 2, 3)) / 3.0
    ov_score = overlap_coef(set(tok1), set(tok2))
    return (j_score + ov_score) / 2.0


class Similarity:
    def __init__(self, mode="hybrid", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.mode = mode
        self.encoder = None
        if mode == "hybrid":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError('Hybrid similarity requires: pip install "react-gpt[semantic]"') from exc
            self.encoder = SentenceTransformer(model_name)

    def pairs(self, pairs):
        # Encode the benchmark in batches, without allocating a quadratic pair matrix.
        vectors = self.encoder.encode([text for pair in pairs for text in pair], normalize_embeddings=True) if self.encoder else None
        result = []
        for i, (left, right) in enumerate(pairs):
            value = _lexical_similarity(left, right)
            if vectors is not None:
                value = 0.4 * value + 0.6 * max(0.0, min(1.0, float(vectors[2*i] @ vectors[2*i+1])))
            result.append(value)
        return result

    def matrix(self, texts):
        vectors = self.encoder.encode(texts, normalize_embeddings=True) if self.encoder and texts else None
        result = [[1.0 if i == j else 0.0 for j in range(len(texts))] for i in range(len(texts))]
        for i in range(len(texts)):
            for j in range(i):
                value = _lexical_similarity(texts[i], texts[j])
                if vectors is not None:
                    value = 0.4 * value + 0.6 * max(0.0, min(1.0, float(vectors[i] @ vectors[j])))
                result[i][j] = result[j][i] = value
        return result


def complete_linkage(matrix, threshold):
    """Deterministic complete linkage using a heap of cluster similarities."""
    import heapq
    groups = {i: [i] for i in range(len(matrix))}
    scores = {(j, i): matrix[i][j] for i in groups for j in groups if j < i}
    heap = [(-score, a, b) for (a, b), score in scores.items() if score >= threshold]
    heapq.heapify(heap)
    next_id = len(groups)
    while heap:
        negative_score, a, b = heapq.heappop(heap)
        if a not in groups or b not in groups:
            continue
        members = groups.pop(a) + groups.pop(b)
        for other in groups:
            score = min(scores[tuple(sorted((a, other)))], scores[tuple(sorted((b, other)))])
            scores[(other, next_id)] = score
            if score >= threshold:
                heapq.heappush(heap, (-score, other, next_id))
        groups[next_id] = members
        next_id += 1
    return sorted((sorted(members) for members in groups.values()), key=lambda members: members[0])


def calibrate(path, similarity):
    """CSV sentence1,sentence2,label; label is a human binary equivalence label."""
    import csv
    from .storage import digest
    from pathlib import Path
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not r.get("sentence1") or not r.get("sentence2") or r.get("label") not in {"0", "1"} for r in rows):
        raise ValueError("Benchmark needs nonempty sentence1,sentence2 and binary label columns")
    if {r["label"] for r in rows} != {"0", "1"}:
        raise ValueError("Calibration requires both positive and negative examples")
    scores = similarity.pairs([(r["sentence1"], r["sentence2"]) for r in rows])
    grid = []
    for step in range(1, 100):
        theta = step / 100
        tp = sum(s >= theta and r["label"] == "1" for s, r in zip(scores, rows))
        fp = sum(s >= theta and r["label"] == "0" for s, r in zip(scores, rows))
        fn = sum(s < theta and r["label"] == "1" for s, r in zip(scores, rows))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        grid.append(dict(threshold=theta, precision=precision, recall=recall, f1=f1))
    best = max(grid, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
    return {"threshold": best["threshold"], "source": "benchmark", "benchmark_hash": digest(Path(path).read_text()),
            "pairs": len(rows), "metrics": grid}
