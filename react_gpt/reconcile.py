"""Per-article ModeX-inspired reconciliation with complete provenance."""
from .similarity import complete_linkage
from .storage import digest


def reconcile(articles, candidates, config, similarity, threshold):
    result = []
    for article in articles:
        pool = [r for r in candidates if r["article_id"] == article["article_id"]]
        if not pool:
            continue
        texts = [" ".join(r[k] for k in ("recommendation", "positive_impact", "evidence")) for r in pool]
        matrix = similarity.matrix(texts)
        for i in range(len(pool)):
            for j in range(i):
                if pool[i]["model"] == pool[j]["model"]:
                    matrix[i][j] *= config.intra_model_alpha
                    matrix[j][i] = matrix[i][j]
        for indices in complete_linkage(matrix, threshold):
            models = sorted({pool[i]["model"] for i in indices})
            if len(models) < config.min_support:
                continue
            centroid = max(indices, key=lambda i: sum(matrix[i][j] for j in indices if i != j))
            item = pool[centroid]
            members = [pool[i] for i in sorted(indices)]
            result.append({**{k: v for k, v in article.items() if k != "text"},
                "cluster_id": digest(sorted(r["candidate_id"] for r in members))[:20],
                "support": len(models), "support_fraction": len(models) / len(config.models),
                "models_present": models, "actionable": item["recommendation"],
                "impact": item["positive_impact"], "evidence": item["evidence"],
                "centroid_confidence": item["confidence"],
                "avg_confidence": sum(r["confidence"] for r in members) / len(members),
                "members": members})
    return result
