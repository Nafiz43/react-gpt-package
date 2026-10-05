# ReACT-GPT

ReACT-GPT extracts evidence-based recommendations from research literature. Several language models read each article independently, their recommendations are reconciled per article, an evaluation panel votes on soundness and precision, actions are categorized, and an optional final judge checks every recommendation against its source text. Results land in CSV, JSON, and a self-contained interactive HTML report.

Models can run locally through [Ollama](https://ollama.com/) or remotely through Anthropic, OpenAI, or Typesafe Jev.

> **License: noncommercial research and education only.** ReACT-GPT is not open source. You may install and run the unmodified software for noncommercial research and education. Modifying, redistributing, or using it commercially requires written permission from UC Davis. **Publications that use ReACT-GPT must cite it** (see [Citation](#citation)). Full terms: [LICENSE](https://github.com/Nafiz43/react-gpt-package/blob/main/LICENSE).

## Install

Python 3.10 or newer.

```bash
python -m pip install "react-gpt[pdf,semantic]"
```

| Extra | Adds |
| --- | --- |
| `pdf` | PDF input (PyMuPDF4LLM) |
| `semantic` | Hybrid lexical + sentence-embedding reconciliation (the default `--similarity hybrid`) |
| `anthropic` | `anthropic:<model>` models |
| `openai` | `openai:<model>` models |
| `all` | Everything above |

Text-only corpora with `--similarity lexical` need no extras. Typesafe Jev needs no extra.

For local models, install and start Ollama, then pull at least one JSON-capable model:

```bash
ollama pull qwen3:8b
ollama pull llama3:8b
```

## Quick start

```bash
react-gpt run \
  --corpus ./papers \
  --output ./runs/my-corpus \
  --models qwen3:8b llama3:8b \
  --open
```

Or answer a few questions and let the wizard write a reusable `react-gpt.json`:

```bash
react-gpt init --run
```

`--open` shows the report in your browser when the run finishes. `react-gpt report ./runs/my-corpus --open` regenerates it later without model calls.

## Models and providers

Every model role accepts any provider. Bare names are Ollama models; prefixes select a hosted provider.

| Name | Provider | Roles |
| --- | --- | --- |
| `qwen3:8b`, `ollama:qwen3:8b` | Ollama (`--ollama-url`, default `http://localhost:11434`) | all |
| `anthropic:claude-opus-5-5` | Anthropic | all |
| `openai:<model>` | OpenAI | all |
| `jev:jev-1.13.0` | Typesafe Jev | evaluation and judge only |

Typesafe Jev answers multiple-choice questions and cannot generate text, so it cannot extract recommendations or propose categories. A run with Jev evaluators uses the first text-capable model for `--category-mode dynamic`.

```bash
react-gpt run --corpus ./papers --output ./runs/mixed \
  --models qwen3:8b anthropic:claude-opus-5-5 \
  --evaluators qwen3:8b anthropic:claude-opus-5-5 jev:jev-1.13.0 \
  --judge anthropic:claude-opus-5-5
```

**API keys.** ReACT-GPT reads `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `TYPESAFE_API_KEY` when set. Otherwise it asks for the key at a hidden prompt in the terminal where the run is happening. Keys are kept in memory only and never written to disk or logs.

**Before any remote call** the run lists the remote models, roughly how many requests and input tokens extraction will use, and a reminder that article text will leave your computer. It then asks for confirmation. Pass `--yes` for unattended runs. Local-only runs never ask.

**Thinking models.** Ollama models that support reasoning (such as qwen3) answer much faster with reasoning off, which is the default. Add `--think` to let them reason first. Hosted models use their own defaults.

## Final judge

`--judge <model>` adds one last pass over every final recommendation. For each one, the judge reads the exact source chunks the extraction models saw and answers four checks:

| Column | Check | Values |
| --- | --- | --- |
| `JUDGE_EVIDENCE` | Is the stated evidence actually in the source? | `YES`, `PARTIAL`, `NO` |
| `JUDGE_SUPPORT` | Does the action follow from that evidence? | `YES`, `NO`, `UNCERTAIN` |
| `JUDGE_CATEGORY_FIT` | Are the assigned categories right? | `YES`, `NO`, `UNCERTAIN` |
| `JUDGE_DUPLICATE_OF` | Cluster ID of a near-duplicate among the five most similar recommendations, if any | ID or empty |

`JUDGE_NOTE` holds a one-sentence rationale (empty for Jev). The judge only annotates: nothing is removed or re-ranked. The report shows the verdict in each action's detail panel.

## Corpus formats

Input is one file or a directory searched recursively.

- `.pdf`: converted to Markdown locally. Scanned PDFs with no text layer are recognized with Apple Vision OCR on macOS; on other systems, supply a `.txt` or `.md` version. The title comes from the PDF's metadata, else the largest text on page 1.
- `.md` / `.txt`: UTF-8 article text; the first Markdown H1 becomes the title.
- `.csv` / `.jsonl`: one article per row or line, with `text` required and `title`, `link`, `venue` optional.

Use a dedicated input folder; every supported file in it is ingested. The output directory must be outside it. Example corpus and configuration: [examples/](https://github.com/Nafiz43/react-gpt-package/tree/main/examples).

To fix a title or add a link, put a sidecar next to the file, such as `paper.pdf.metadata.json`:

```json
{"title": "Full paper title", "link": "https://doi.org/...", "venue": "Journal, year"}
```

## Options

| Option | Purpose |
| --- | --- |
| `--evaluators m1 m2 m3` | Scoring and categorization panel (defaults to the extraction models) |
| `--judge model` | Optional final judge |
| `--domain oss\|medicine` | Prompts and fixed categories for OSS sustainability (default) or medical research |
| `--category-mode fixed\|dynamic` | Predefined categories, or a taxonomy discovered from this run's actions |
| `--features` | OSS only: map actions to seven socio-technical features |
| `--similarity hybrid\|lexical` | Reconciliation metric |
| `--threshold 0.65` / `--benchmark pairs.csv` | Fixed similarity threshold, or calibrate it from labeled `sentence1,sentence2,label` pairs |
| `--min-support 2` | Keep only actions proposed by at least two extraction models |
| `--think` / `--no-think` | Ollama reasoning on or off (default off) |
| `--yes` | Skip the remote-provider confirmation |
| `--limit 10` | First ten articles in file order |
| `--until extract` | Stop after a stage (`preprocess`, `extract`, `reconcile`, `score`, `categorize`, `features`, `judge`, `export`) |
| `--dry-run` | Validate configuration and list inputs without model calls or writes |
| `--timeout 600 --retries 2` | Per-request timeout and extra attempts |
| `--num-ctx 32768 --num-predict 4096 --chunk-chars 24000` | Context, output, and input-chunk sizes |

The similarity threshold of 0.65 is an illustrative starting value, not a calibrated default. For research use, calibrate it with a representative human-labeled `--benchmark`. Run `react-gpt run --help` for every option. `python -m react_gpt` works without the console script.

## Outputs and resuming

Each output directory contains:

- `final_set.csv` and `final_set.json`: the ranked catalog with votes, categories, judge annotations, and full provenance.
- `report.html`: an offline, searchable graph of actions and categories with evidence, votes, and source text. It makes no network requests.
- `manifest.json`: settings, corpus fingerprint, stage completion, and a summary. Inspect it with `react-gpt status <dir>`.
- `preprocess.json` through `judge.json`: intermediate artifacts for every stage, plus `taxonomy.json` and `calibration.json`.
- `cache/requests/`: every successful model response with its prompt. `errors/`: every failed attempt.

Rerun the same command to resume: successful requests are reused, so nothing is paid for twice. A failed request stops the run after bounded retries and is never treated as an empty answer or a negative vote. Changing the corpus or configuration requires a new output directory.

Recommendations are ranked sound-and-precise first, then sound only, precise only, and neither, with ties broken by model support and extraction confidence.

## Methodology and scope

Hybrid reconciliation uses 40% lexical and 60% sentence-embedding similarity with a same-model penalty (default 0.15) and complete linkage; the most central original candidate represents each cluster. Model support counts distinct full model names. Every verdict and category needs a strict majority of the evaluation panel, so an even split does not pass.

Soundness, precision, categories, features, and judge verdicts are model judgments, not independent verification. Medical mode is a literature-extraction workflow, not clinical validation or patient advice. `RA` is a historical prompt name for a reasoning/action-style prompt, not a retrieval system. Chunking preserves every input character but can separate evidence from its context.

## Development

```bash
python -m unittest discover -s tests -v
```

The suite runs against a simulated Ollama server and needs no GPU, models, or network access.

## Citation

Citing ReACT-GPT is a condition of the license. Cite it as:

> Khan, N. I., & Filkov, V. (2026). *ReACT-GPT* (Version 0.1.0) [Computer software]. University of California, Davis. https://github.com/Nafiz43/react-gpt-package

```bibtex
@software{khan_filkov_react_gpt_2026,
  author  = {Khan, Nafiz Imtiaz and Filkov, Vladimir},
  title   = {{ReACT-GPT}},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/Nafiz43/react-gpt-package}
}
```

Machine-readable metadata: [CITATION.cff](https://github.com/Nafiz43/react-gpt-package/blob/main/CITATION.cff).

## Permissions

For commercial use, modification, redistribution, or any use the license does not cover, contact the [UC Davis Technology Transfer Office](https://research.ucdavis.edu/about/offices/technology-transfer-office/).

Copyright © 2026 The Regents of the University of California. All rights reserved.
