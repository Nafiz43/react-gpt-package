"""Interactive setup and automation-friendly command line interface."""
import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from . import __version__
from .config import Config
from .corpus import discover
from .pipeline import STAGES, run
from .models import ModelClient, REMOTE, split_model
from .storage import read_json, write_json, RunLock


def parser():
    root = argparse.ArgumentParser(prog="react-gpt", description="Extract evidence-based actions from a research corpus.")
    root.add_argument("--version", action="version", version="%(prog)s " + __version__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Interactively select options and save a reusable JSON configuration")
    init.add_argument("--config", default="react-gpt.json")
    init.add_argument("--run", action="store_true", help="Run immediately after setup")
    execute = commands.add_parser("run", help="Run end-to-end, automatically resuming successful model calls")
    execute.add_argument("--config", help="JSON configuration; relative paths resolve beside this file")
    execute.add_argument("--corpus", help="PDF/MD/TXT directory, CSV/JSONL file, or individual document")
    execute.add_argument("--output", help="Dedicated output directory outside the corpus")
    execute.add_argument("--models", nargs="+", help="Extraction models: Ollama names (qwen3:8b), or anthropic:<model>, openai:<model>")
    execute.add_argument("--evaluators", nargs="+", help="Evaluation panel (defaults to extraction models); jev:<model> is also allowed")
    execute.add_argument("--judge", help="Optional final judge model; any provider, e.g. qwen3:14b, anthropic:claude-opus-5-5, jev:jev-1.13.0")
    execute.add_argument("--think", action=argparse.BooleanOptionalAction, default=None, help="Let Ollama thinking models reason first (slower; default off)")
    execute.add_argument("--yes", action="store_true", help="Skip the confirmation before sending text to remote providers")
    execute.add_argument("--category-mode", choices=["fixed", "dynamic"], help="Use predefined categories or discover overlapping categories from this run")
    execute.add_argument("--domain", choices=["oss", "medicine"], help="Research domain (default: oss)")
    execute.add_argument("--ollama-url")
    execute.add_argument("--prompting", choices=["IP", "CoT", "RA"])
    execute.add_argument("--similarity", choices=["hybrid", "lexical"])
    execute.add_argument("--embedding-model")
    execute.add_argument("--benchmark", help="Calibration CSV: sentence1,sentence2,label")
    for name in ("threshold", "temperature", "timeout", "intra-model-alpha"):
        execute.add_argument("--" + name, type=float)
    for name in ("retries", "num-ctx", "num-predict", "chunk-chars", "min-support", "limit"):
        execute.add_argument("--" + name, type=int)
    execute.add_argument("--keep-alive")
    execute.add_argument("--features", action=argparse.BooleanOptionalAction, default=None)
    execute.add_argument("--until", choices=STAGES, default="export", help="Stop after this stage; earlier requests are reused")
    execute.add_argument("--open", action="store_true", help="Open the graph report after a complete run")
    execute.add_argument("--dry-run", action="store_true", help="Validate configuration and list inputs without model calls or writes")
    report = commands.add_parser("report", help="Regenerate an offline graph report from an existing run without inference")
    report.add_argument("output")
    report.add_argument("--open", action="store_true")
    status = commands.add_parser("status", help="Inspect an existing run")
    status.add_argument("output")
    return root


def load_config(args):
    values = {}
    if args.config:
        file = Path(args.config).resolve()
        values = read_json(file)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        for key in ("corpus", "output", "benchmark"):
            if values.get(key):
                values[key] = str((file.parent / values[key]).resolve())
    for key in Config.__dataclass_fields__:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    missing = [key for key in ("corpus", "output", "models") if key not in values]
    if missing:
        raise ValueError("Required options: " + ", ".join("--" + key for key in missing))
    return Config(**values).validate()


def select_models(client, panel, default=None):
    """Choose installed names or download new names; keep aliases unique."""
    while True:
        try:
            available = client.list_models()
        except RuntimeError as exc:
            print(f"{exc}\nOnly remote models (anthropic:, openai:, jev:) can be selected until Ollama is running.")
            available = []
        print(f"\nAvailable Ollama models for {panel.lower()}:")
        for index, name in enumerate(available, 1):
            print(f"  {index}. {name}")
        if not available:
            print("  No models installed yet.")
        print("Enter numbers or model names, separated by commas.")
        print("New model names will be downloaded automatically from Ollama.")
        if default:
            hint = "Enter = " + ", ".join(default)
        elif available:
            hint = "Enter = 1"
        else:
            hint = "enter a model name, e.g. llama3.1:latest"
        answer = input(f"{panel} models [{hint}]: ").strip()
        if not answer and default:
            return list(default)
        if not answer and available:
            answer = "1"
        selected = []
        try:
            for token in answer.split(","):
                token = token.strip()
                if not token:
                    raise ValueError("Enter at least one model; remove empty comma-separated entries.")
                if token.isdecimal():
                    number = int(token)
                    if not 1 <= number <= len(available):
                        raise ValueError(f"Model number {number} is not in the list.")
                    name = available[number - 1]
                else:
                    if any(c.isspace() for c in token):
                        raise ValueError("Separate model names with commas, not spaces.")
                    name = token if ":" in token.rsplit("/", 1)[-1] else token + ":latest"
                    if token in available:
                        name = token
                if name not in selected:
                    selected.append(name)
        except ValueError as exc:
            print(f"{exc} Try again.")
            continue
        try:
            for name in selected:
                if name not in available and split_model(name)[0] == "ollama":
                    print(f"\nInstalling {name} on {client.config.ollama_url}...", flush=True)
                    previous = None
                    def progress(event):
                        nonlocal previous
                        status = str(event.get("status", "Downloading"))
                        total, done = event.get("total", 0), event.get("completed", 0)
                        percent = int(100 * done / total) if isinstance(total, (int, float)) and total > 0 and isinstance(done, (int, float)) else None
                        marker = (status, event.get("digest"), percent // 5 if percent is not None else None)
                        if marker != previous:
                            print("  " + status + (f" — {percent}%" if percent is not None else ""), flush=True)
                            previous = marker
                    client.pull_model(split_model(name)[1], progress)
            local = [name for name in selected if split_model(name)[0] == "ollama"]
            if local:
                client.preflight(local)
        except RuntimeError as exc:
            print(f"{exc}\nChoose a different model or retry the download.")
            continue
        print(f"Selected {panel.lower()} models: {', '.join(selected)}")
        return selected


def confirm_remote(config, yes=False):
    """Before any remote call: show what leaves the machine and a rough size, then ask."""
    roles = {"extraction": config.models, "evaluation": config.evaluators, "judge": [config.judge] if config.judge else []}
    remote = {role: [m for m in models if split_model(m)[0] in REMOTE] for role, models in roles.items()}
    if not any(remote.values()):
        return
    sizes = [path.stat().st_size for path in discover(config.corpus)]
    chars = sum(sizes)
    chunks = sum(max(1, -(-size // config.chunk_chars)) for size in sizes)
    print("\nThis run sends corpus text to remote model providers:")
    for role, models in remote.items():
        if models:
            print(f"  {role}: {', '.join(models)}")
    if remote["extraction"]:
        print(f"  Extraction: about {chunks} requests and ~{(chars + 4000 * chunks) // 4:,} input tokens per remote model "
              "(file bytes / 4; PDFs are usually smaller as text).")
    per_item = 2 + 1 + (1 if config.features else 0)
    if remote["evaluation"]:
        print(f"  Evaluation: {per_item} requests per final recommendation per remote evaluator (count known after extraction).")
    if remote["judge"]:
        print("  Judge: 1 request per final recommendation, each with up to two source chunks.")
    print("  Article text leaves this computer and is subject to each provider's data terms. Cached requests are not re-sent.")
    if yes:
        return
    if not sys.stdin.isatty():
        raise ValueError("Remote models need confirmation; rerun with --yes to proceed non-interactively")
    if input("Continue? [y/N]: ").strip().lower() not in {"y", "yes"}:
        raise ValueError("Cancelled; nothing was sent")


def wizard(path):
    target = Path(path).resolve()
    if target.exists():
        raise ValueError(f"Configuration already exists: {target}; choose another --config path")
    def ask(label, default="", check=lambda value: value):
        """Re-ask until `check` accepts the answer, so mistakes surface at the question that caused them."""
        while True:
            value = input(f"{label}" + (f" [{default}]" if default else "") + ": ").strip() or default
            try:
                return check(value)
            except (ValueError, RuntimeError) as exc:
                print(f"{exc} Try again.")

    def one_of(*options):
        def check(value):
            for option in options:
                if value.lower() == option.lower():
                    return option
            raise ValueError(f"Enter one of: {', '.join(options)}.")
        return check

    def required(value):
        if not value:
            raise ValueError("This answer is required.")
        return value

    def corpus_path(value):
        if not Path(required(value)).expanduser().exists():
            raise ValueError(f"{value} does not exist.")
        discover(Path(value).expanduser())
        return str(Path(value).expanduser())

    def settings_check(**changed):
        # Config.validate is the single source of truth for paths and URLs.
        Config(**{"corpus": corpus, "output": output, "models": ["x"], **changed}).validate()
        return next(iter(changed.values()))

    def judge_check(value):
        if value and split_model(value)[0] == "ollama":
            client.preflight([value])  # installed? Remote keys are asked when the run starts.
        return value or None

    def similarity_check(value):
        value = one_of("hybrid", "lexical")(value)
        if value == "hybrid" and importlib.util.find_spec("sentence_transformers") is None:
            raise ValueError('Hybrid needs: pip install "react-gpt[semantic]" (or choose lexical).')
        return value

    def benchmark_check(value):
        if value and not Path(value).expanduser().is_file():
            raise ValueError(f"{value} is not a file.")
        return str(Path(value).expanduser()) if value else None

    def threshold_check(value):
        try:
            number = float(value)
        except ValueError:
            raise ValueError("Enter a number between 0 and 1.") from None
        if not 0 <= number <= 1:
            raise ValueError("Enter a number between 0 and 1.")
        return number

    yes_no = lambda value: one_of("yes", "no", "y", "n")(value) in {"yes", "y"}
    corpus = output = ask("Corpus file or directory", check=corpus_path)
    output = ask("Output directory", "react-output", lambda value: settings_check(output=required(value)))
    url = ask("Ollama URL", "http://localhost:11434", lambda value: settings_check(ollama_url=value))
    client = ModelClient(Config(corpus=corpus, output=output, models=[], ollama_url=url))
    print("Remote models are also accepted: anthropic:<model>, openai:<model> (and jev:<model> for evaluation/judge).")
    models = select_models(client, "Extraction")
    evaluators = select_models(client, "Evaluation", default=models)
    judge = ask("Final judge model (blank = no judge; e.g. qwen3:14b, anthropic:claude-opus-5-5, jev:jev-1.13.0)", check=judge_check)
    think = ask("Let Ollama thinking models reason first? Slower. yes/no", "no", yes_no)
    prompting = ask("Prompt strategy: IP, CoT, RA", "CoT", one_of("IP", "CoT", "RA"))
    similarity = ask("Similarity: hybrid or lexical", "hybrid", similarity_check)
    benchmark = ask("Calibration CSV (blank uses an explicit threshold)", check=benchmark_check)
    threshold = ask("Similarity threshold (uncalibrated unless benchmark supplied)", "0.65", threshold_check) if not benchmark else 0.65
    features = ask("Map socio-technical features? yes/no", "no", yes_no)
    config = Config(corpus=corpus, output=output, models=models,
                    evaluators=evaluators,
                    ollama_url=url, prompting=prompting, similarity=similarity,
                    benchmark=benchmark or None, threshold=threshold, features=features,
                    judge=judge, think=think).validate()
    write_json(target, config.to_dict())
    print(f"Saved {target}")
    return config


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("react_gpt").setLevel(logging.INFO)
    try:
        if args.command == "report":
            from .pipeline import export_report
            out = Path(args.output).resolve()
            with RunLock(out / ".lock"):
                manifest = read_json(out / "manifest.json")
                taxonomy = read_json(out / "taxonomy.json") if (out / "taxonomy.json").exists() else None
                export_report(out / "report.html", read_json(out / "final_set.json"),
                              articles=read_json(out / "preprocess.json"), config=Config(**manifest["config"]),
                              processed_at=manifest.get("completed_at"), taxonomy=taxonomy)
            print(out / "report.html")
            if args.open:
                import webbrowser
                webbrowser.open((out / "report.html").as_uri())
        elif args.command == "status":
            print(json.dumps(read_json(Path(args.output) / "manifest.json"), indent=2))
        elif args.command == "init":
            config = wizard(args.config)
            if args.run:
                confirm_remote(config)
                run(config)
            else:
                print(f'Run with: react-gpt run --config "{Path(args.config).resolve()}"')
        else:
            config = load_config(args)
            if args.dry_run:
                print(json.dumps({"config": config.to_dict(), "inputs": [str(p) for p in discover(config.corpus)],
                                  "until": args.until}, indent=2))
                return
            confirm_remote(config, args.yes)
            result = run(config, args.until)
            print(f'{result["status"]}: {config.output} (through {result["last_stage"]})')
            if args.open and result["status"] == "complete":
                import webbrowser
                webbrowser.open((Path(config.output) / "report.html").as_uri())
    except (ValueError, TypeError, OSError, RuntimeError, ImportError, EOFError) as exc:
        print(f"react-gpt: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Interrupted. Successful requests are saved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130) from None
