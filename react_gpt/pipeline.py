"""Ordered pipeline stages; all expensive model requests are resumable."""
import html
import json
import importlib.util
import logging
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from . import __version__, prompts
from .corpus import load_corpus, chunks
from .models import ModelClient, FEATURES
from .taxonomy import discover_taxonomy
from .reconcile import reconcile
from .similarity import Similarity, calibrate
from .storage import digest, read_json, write_json, write_csv, write_text, RunLock

STAGES = ("preprocess", "extract", "reconcile", "score", "categorize", "features", "judge", "export")
COLUMNS = ["rank", "quality_tier", "article_id", "article_title", "venue", "article_link", "source", "cluster_id", "support",
           "support_fraction", "models_present", "actionable", "impact", "evidence", "avg_confidence",
           "centroid_confidence", "SOUND", "PRECISE", "CATEGORY", "FEATURES", "JUDGE_EVIDENCE", "JUDGE_SUPPORT",
           "JUDGE_CATEGORY_FIT", "JUDGE_DUPLICATE_OF", "JUDGE_NOTE"]
log = logging.getLogger(__name__)


def majority(votes):
    return "YES" if votes.count("YES") > len(votes) / 2 else "NO"


def extraction_prompt(text, method, domain="oss"):
    template = {"IP": prompts.IP_template, "CoT": prompts.CoT_template, "RA": prompts.RA_template}[method]
    return prompts.domain_extraction(domain) + "\n<article>\n" + text + "\n</article>\n" + template


def run(config, until="export", client=None):
    config.validate()
    if until not in STAGES:
        raise ValueError(f"Unknown stage: {until}")
    out = Path(config.output)
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out / ".lock"), logging_redirect_tqdm(), tqdm(total=STAGES.index(until) + 1, desc="Pipeline", unit="stage", dynamic_ncols=True) as progress:
        return _run(config, until, client or ModelClient(config), progress)


def _run(config, until, client, progress):
    out = Path(config.output)
    manifest_path = out / "manifest.json"
    log.info("Loading corpus: %s", config.corpus)
    articles = load_corpus(config)
    identity = {"version": __version__, "config": config.to_dict(), "articles": articles,
                "benchmark": Path(config.benchmark).read_text() if config.benchmark else None}
    fingerprint = digest(identity)
    if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
        raise ValueError("Output belongs to a different corpus/configuration; choose a new output directory")
    previous = read_json(manifest_path) if manifest_path.exists() else {}
    started = datetime.now(timezone.utc).isoformat()
    manifest = {"started_at": previous.get("started_at") or started, "last_run_started_at": started, "fingerprint": fingerprint, "version": __version__, "config": config.to_dict(),
                "status": "running", "article_count": len(articles), "completed_stages": []}
    write_json(manifest_path, manifest)

    def completed(stage, data):
        write_json(out / f"{stage}.json", data)
        manifest["completed_stages"].append(stage)
        manifest["last_stage"] = stage
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        if stage == "export":
            manifest["completed_at"] = manifest["updated_at"]
        if stage == until:
            manifest["status"] = "complete" if stage == "export" else "stopped"
        write_json(manifest_path, manifest)
        progress.set_postfix_str(stage)
        progress.update(1)
        log.info("%s: %d records", stage, len(data))
        return stage == until

    try:
        if completed("preprocess", articles):
            return manifest
        if config.similarity == "hybrid" and until != "extract" and importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError('Hybrid similarity requires: pip install "react-gpt[semantic]"')
        client.preflight(sorted(set(config.models + config.evaluators + ([config.judge] if config.judge else []))))
        candidates = []
        for model in config.models:
            for number, article in enumerate(articles, 1):
                log.info("Extracting %s: article %d/%d (%s)", model, number, len(articles), article["article_title"])
                seen = set()
                for index, text in enumerate(tqdm(list(chunks(article["text"], config.chunk_chars)), desc=f"Extract {model} [{number}/{len(articles)}]", unit="chunk", dynamic_ncols=True)):
                    rows = client.ask(model, extraction_prompt(text, config.prompting, config.domain), "extract")
                    for row in rows:
                        key = digest([article["article_id"], model, row])
                        if key not in seen:
                            candidates.append({**row, "candidate_id": key, "article_id": article["article_id"],
                                               "model": model, "chunk": index})
                            seen.add(key)
        if completed("extract", candidates):
            return manifest
        similarity = Similarity(config.similarity, config.embedding_model) if candidates or config.benchmark else None
        calibration = calibrate(config.benchmark, similarity) if config.benchmark else {
            "source": "configured (not benchmark-calibrated)", "threshold": config.threshold}
        write_json(out / "calibration.json", calibration)
        rows = reconcile(articles, candidates, config, similarity, calibration["threshold"])
        if completed("reconcile", rows):
            return manifest
        for row in rows:
            row["quality_votes"] = {}
        sound_definition, precise_definition = prompts.domain_quality(config.domain)
        for model in config.evaluators:
            for row in tqdm(rows, desc=f"Score {model}", unit="recommendation", dynamic_ncols=True):
                votes = {}
                for kind, definition in (("sound", sound_definition), ("precise", precise_definition)):
                    prompt = (f"{definition}\nEvaluate this ReACT:\n" + json.dumps({k: row[k] for k in ("actionable", "impact", "evidence")})
                              + '\nReturn only JSON: {"verdict": "YES"} or {"verdict": "NO"}.')
                    votes[kind] = client.ask(model, prompt, kind)
                row["quality_votes"][model] = votes
        for row in rows:
            for kind in ("sound", "precise"):
                row[kind.upper()] = majority([v[kind] for v in row["quality_votes"].values()])
        if completed("score", rows):
            return manifest
        taxonomy = discover_taxonomy(rows, config, client)
        write_json(out / "taxonomy.json", taxonomy)
        categories = taxonomy["categories"]
        category_prompt = ("Assign only applicable " + config.domain + " categories. Select every matching category; multiple labels per action are allowed. Do not select unrelated categories.\n" + json.dumps(categories)
                           + '\nReturn only JSON: {"categories": ["exact category name"]}; use [] if none apply.\nActionable: ')
        for row in rows:
            row["category_votes"] = {}
        for model in config.evaluators:
            for row in tqdm(rows, desc=f"Categorize {model}", unit="recommendation", dynamic_ncols=True):
                row["category_votes"][model] = client.ask(model, category_prompt + row["actionable"], "categories",
                    **({"categories": categories} if config.category_mode == "dynamic" else {}))
        for row in rows:
            row["categories"] = [category for category in categories
                if sum(category in vote for vote in row["category_votes"].values()) > len(config.evaluators) / 2]
            row["CATEGORY"] = " | ".join(row["categories"]) or "NONE"
        if completed("categorize", rows):
            return manifest
        for row in rows:
            row["feature_votes"] = {}
        if config.features:
            for model in config.evaluators:
                for row in tqdm(rows, desc=f"Features {model}", unit="recommendation", dynamic_ncols=True):
                    prompt = prompts.FEATURE_PROMPT + row["actionable"] + '\nOverride output format: return only JSON {"features": ["exact feature name"]}, or {"features": []}.'
                    row["feature_votes"][model] = client.ask(model, prompt, "features")
        for row in rows:
            row["FEATURES"] = " | ".join(f for f in sorted(FEATURES)
                if sum(f in vote for vote in row["feature_votes"].values()) > len(config.evaluators) / 2) if config.features else ""
        if completed("features", rows):
            return manifest
        if config.judge:
            from .judge import judge_rows
            judge_rows(rows, articles, categories, config, client)
        if completed("judge", rows):
            return manifest
        rows = rank_actionables(rows)
        write_csv(out / "final_set.csv", rows, COLUMNS)
        write_json(out / "final_set.json", rows)
        manifest["summary"] = {"articles": len(articles), "candidates": len(candidates), "actionables": len(rows),
            "sound": sum(r["SOUND"] == "YES" for r in rows), "precise": sum(r["PRECISE"] == "YES" for r in rows)}
        processed_at = datetime.now(timezone.utc).isoformat()
        export_report(out / "report.html", rows, articles=articles, config=config, processed_at=processed_at, taxonomy=taxonomy)
        completed("export", rows)
        return manifest
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        write_json(manifest_path, manifest)
        raise


def rank_actionables(rows):
    """Order by quality verdicts, then panel support and extraction confidence."""
    def tier(row):
        return (2 if row.get("SOUND") == "YES" else 0) + (1 if row.get("PRECISE") == "YES" else 0)

    labels = {3: "Sound and precise", 2: "Sound; needs specificity",
              1: "Precise; soundness concern", 0: "Needs review"}
    ordered = sorted(rows, key=lambda row: (
        -tier(row), -row.get("support_fraction", 0), -row.get("avg_confidence", 0),
        row["actionable"].casefold(), row.get("cluster_id", "")))
    return [{**row, "rank": index, "quality_tier": labels[tier(row)]}
            for index, row in enumerate(ordered, 1)]


def export_report(path, rows, *, articles=None, config=None, processed_at=None, taxonomy=None):
    """Include source context even when an article yields no recommendations."""
    esc = lambda value: html.escape(str(value), quote=True)
    sources = []
    display_titles = {}
    source_records = []
    for article in articles or []:
        source = article["source"]
        display_titles[article["article_id"]] = article["article_title"]
        links = []
        file_url = ""
        if config:
            base = Path(config.corpus)
            local = base / source if base.is_dir() else base
            if not local.is_file() and ":" in source:
                local = base / source.rsplit(":", 1)[0] if base.is_dir() else base
            if local.is_file():
                file_url = local.resolve().as_uri()
                links.append(f'<a href="{esc(file_url)}">Open original article</a>')
        url = article.get("article_link", "")
        if urlparse(url).scheme in {"http", "https"}:
            links.append(f'<a href="{esc(url)}">Publisher / source link</a>')
        source_records.append({"article_id": article["article_id"], "article_title": display_titles[article["article_id"]],
                               "file_url": file_url, "article_link": url, "source": source})
        sources.append(f'<section class="source" id="source-{esc(article["article_id"])}">'
                       f'<h3>{esc(display_titles[article["article_id"]])}</h3><p>{esc(source)}</p>'
                       + ' · '.join(links)
                       + f'<details><summary>Read extracted article text</summary><pre>{esc(article["text"])}</pre></details></section>')
    domain_note = ("Medical research literature extraction; model rankings are not clinical validation or patient-specific advice."
                   if config and config.domain == "medicine" else "")
    cards = []
    for row in rank_actionables(rows):
        cards.append(f'<article><p class="rank">#{row["rank"]} · {esc(row["quality_tier"])}</p><h2>{esc(row["actionable"])}</h2>'
                     f'<p>Base article: <a href="#source-{esc(row["article_id"])}">{esc(display_titles.get(row["article_id"], row["article_title"]))}</a></p>'
                     f'<p>{esc(row["CATEGORY"])} · SOUND: {esc(row["SOUND"])} · PRECISE: {esc(row["PRECISE"])}</p>'
                     f'<p>Model support: {esc(row.get("support", 0))} · Extraction confidence: {row.get("avg_confidence", 0):.0%}</p>'
                     f'<h3>Impact</h3><p>{esc(row["impact"])}</p><h3>Evidence</h3><p>{esc(row["evidence"])}</p>'
                     f'<small>Source: {esc(row["source"])} · Models: {esc(row["models_present"])}</small></article>')
    timestamp_label = datetime.fromisoformat(processed_at).strftime("%B %d, %Y at %H:%M:%S UTC") if processed_at else ""
    timestamp = (f'<time datetime="{esc(processed_at)}">{esc(timestamp_label)}</time>'
                 if processed_at else 'Not recorded for this historical run')
    from .report import render_graph
    taxonomy = taxonomy or {"mode": "fixed", "categories": prompts.domain_categories(config.domain if config else "oss")}
    render_graph(path, rank_actionables(rows), source_records, taxonomy,
                 config.domain if config else "oss", processed_at, ''.join(sources),
                 f'<p>{esc(domain_note)}</p><p>Processed (UTC): {timestamp}</p>' + ''.join(cards))
