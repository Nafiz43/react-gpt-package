"""Read PDF, Markdown, plain text, CSV, and JSONL corpora."""
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm
from .storage import digest, write_json, read_json

SUPPORTED = {".pdf", ".md", ".txt", ".csv", ".jsonl"}


def discover(corpus):
    path = Path(corpus)
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED) if path.is_dir() else [path]
    if not files or any(p.suffix.lower() not in SUPPORTED for p in files):
        raise ValueError("No supported corpus files (PDF, MD, TXT, CSV, JSONL)")
    return files


def load_corpus(config):
    articles = []
    base = Path(config.corpus)
    for path in tqdm(discover(base), desc="Read corpus", unit="file", dynamic_ncols=True):
        relative = str(path.relative_to(base)) if base.is_dir() else path.name
        suffix = path.suffix.lower()
        if suffix in {".csv", ".jsonl"}:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle)) if suffix == ".csv" else [json.loads(line) for line in handle if line.strip()]
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(f"Article must be an object: {relative}:{index+1}")
                text = row.get("text", row.get("Text"))
                title = row.get("title", row.get("Title", f"{path.stem}-{index+1}"))
                link = row.get("link", row.get("Link", ""))
                articles.append(_article(f"{relative}:{index+1}", title, text, link, row.get("venue", relative)))
        else:
            pdf_title = None
            if suffix == ".pdf":
                cache = Path(config.output) / "cache" / "pdf" / (digest([path.read_bytes().hex(), "pdf-v2"]) + ".json")
                if not cache.exists():
                    write_json(cache, read_pdf(path))
                pdf = read_json(cache)
                raw, pdf_title = pdf["text"], pdf["title"]
            else:
                raw = path.read_text(encoding="utf-8-sig")
            title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
            link_match = re.search(r"^\*\*Source:\*\*\s*(.+)$", raw, re.MULTILINE)
            # Only strip the legacy metadata envelope, never arbitrary Markdown sections.
            if "**Original file:**" in raw and link_match:
                parts = re.split(r"(?m)^---\s*$", raw, maxsplit=1)
                body = parts[-1].strip()
            else:
                body = raw.strip()
            articles.append(_article(relative, pdf_title or (title_match.group(1) if title_match else path.stem),
                                     body, link_match.group(1) if link_match else "", relative))
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if suffix not in {".csv", ".jsonl"} and metadata_path.exists():
            metadata = read_json(metadata_path)
            if not isinstance(metadata, dict) or any(k not in {"title", "link", "venue"} for k in metadata):
                raise ValueError(f"Metadata must contain only title, link, or venue: {metadata_path}")
            for key, value in metadata.items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Metadata {key} must be a nonempty string: {metadata_path}")
                articles[-1][{"title": "article_title", "link": "article_link", "venue": "venue"}[key]] = value
            if "title" in metadata:
                articles[-1]["title_explicit"] = True
        if config.limit and len(articles) >= config.limit:
            break
    articles = articles[:config.limit] if config.limit else articles
    if not articles:
        raise ValueError("Corpus contains no articles")
    return articles


def read_pdf(path):
    """Markdown text (OCR for scanned pages on macOS) plus the best available title."""
    try:
        import pymupdf
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError('PDF input requires: pip install "react-gpt[pdf]"') from exc
    with pymupdf.open(path) as document:
        native = sum(len(page.get_text().strip()) for page in document)
        pages = document.page_count
        title = pdf_title(document)
    # ponytail: whole-document heuristic; mixed scanned/native PDFs keep their native text only.
    if native >= 100 * max(pages, 1):
        return {"text": pymupdf4llm.to_markdown(str(path)), "title": title, "extractor": "pymupdf4llm"}
    return {"text": ocr_pdf(path), "title": title, "extractor": "Apple Vision OCR"}


def pdf_title(document):
    """Embedded metadata title, else the largest-font line on page 1."""
    title = (document.metadata or {}).get("title", "").strip()
    if title and not re.search(r"(?i)^untitled|\.(pdf|docx?|tex)$|^microsoft word", title):
        return title
    if not document.page_count:
        return None
    spans = [(round(span["size"], 1), span["text"].strip(), line["bbox"][1])
             for block in document[0].get_text("dict")["blocks"] for line in block.get("lines", [])
             for span in line["spans"] if span["text"].strip()]
    if not spans:
        return None
    largest = max(size for size, _, _ in spans)
    words = " ".join(text for size, text, _ in sorted(spans, key=lambda s: s[2]) if size >= largest - 0.5)
    return words if 5 <= len(words) <= 300 else None


def ocr_pdf(path):
    if sys.platform != "darwin" or not shutil.which("swift"):
        raise ValueError(f"{path.name} is scanned (no text layer). Automatic OCR needs macOS; otherwise supply a .txt or .md version.")
    print(f"Recognizing scanned PDF locally: {path.name}", flush=True)
    process = subprocess.run(["swift", "-module-cache-path", str(Path(tempfile.gettempdir()) / "react-ocr-swift-cache"),
                              str(Path(__file__).with_name("ocr_macos.swift")), str(path)], capture_output=True, timeout=600)
    if process.returncode:
        raise RuntimeError(f"macOS OCR failed for {path.name}: {process.stderr.decode(errors='replace')[-300:]}")
    text = "\n\n".join(json.loads(process.stdout))
    if not text.strip():
        raise ValueError(f"OCR found no text in {path.name}")
    return text


def _article(source, title, text, link, venue):
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Empty or missing article text: {source}")
    return {"article_id": digest([source, text]), "source": source, "article_title": str(title),
            "article_link": str(link), "venue": str(venue), "text": text.strip()}


def chunks(text, size):
    # Non-overlapping bounded chunks preserve every character; split at whitespace when possible.
    while text:
        end = min(len(text), size)
        if end < len(text):
            boundary = text.rfind(" ", max(0, end // 2), end)
            if boundary > 0:
                end = boundary + 1
        yield text[:end]
        text = text[end:]
