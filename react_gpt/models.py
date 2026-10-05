"""Model transports (Ollama, Anthropic, OpenAI, Typesafe Jev), strict response contracts, and resumable caching."""
import getpass
import json
import math
import os
import sys
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from .storage import digest, read_json, write_json
from .prompts import CATEGORIES, domain_categories

FEATURES = {"s_avg_clustering_coef", "s_net_overlap", "t_num_dev_nodes", "t_num_dev_per_file",
            "t_graph_density", "t_net_overlap", "st_num_dev"}


def parse_response(raw, kind, categories=None, choices=None):
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Response must be a JSON object")
    if kind == "taxonomy":
        from .taxonomy import validate_taxonomy
        return validate_taxonomy(value)
    if kind == "extract":
        if value == {"message": "NO ACTIONABLE CAN BE DERIVED"}:
            return []
        rows = value.get("recommendations")
        if not isinstance(rows, list):
            raise ValueError("Missing recommendations array")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each recommendation must be an object")
            for key in ("recommendation", "positive_impact", "evidence"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing nonempty {key}")
            score = row.get("confidence")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("confidence must be a number in [0, 1]")
        return rows
    if kind == "judge":
        for key, spec in choices.items():
            if value.get(key) not in spec["options"]:
                raise ValueError(f"{key} must be one of: " + ", ".join(spec["options"]))
        if not isinstance(value.get("rationale", ""), str):
            raise ValueError("rationale must be a string")
        return {**{key: value[key] for key in choices}, "rationale": value.get("rationale", "").strip()}
    if kind in {"sound", "precise"}:
        if value.get("verdict") not in {"YES", "NO"}:
            raise ValueError("verdict must be YES or NO")
        return value["verdict"]
    key, allowed = ("categories", set(CATEGORIES if categories is None else categories)) if kind == "categories" else ("features", FEATURES)
    values = value.get(key)
    if not isinstance(values, list) or any(not isinstance(v, str) or v not in allowed for v in values):
        raise ValueError(f"Invalid {key}; use an array of exact names")
    return sorted(set(values))


REMOTE = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "jev": "TYPESAFE_API_KEY"}
TEXT_KINDS = {"extract", "taxonomy"}  # Need free-form generation; Jev only answers choice questions.
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_KEYS = {}


def split_model(name):
    """`anthropic:claude-opus-5-5` -> ("anthropic", "claude-opus-5-5"); bare names are Ollama."""
    provider, _, rest = name.partition(":")
    if provider in REMOTE or provider == "ollama":
        if not rest:
            raise ValueError(f"Missing model name after '{provider}:'")
        return provider, rest
    return "ollama", name


def api_key(provider):
    """Environment variable first, else a hidden terminal prompt; never written to disk."""
    key = os.environ.get(REMOTE[provider]) or _KEYS.get(provider)
    if not key:
        if not sys.stdin.isatty():
            raise RuntimeError(f"Set {REMOTE[provider]}; no terminal is available to prompt for the {provider} API key")
        key = getpass.getpass(f"{provider} API key (hidden input, not saved): ").strip()
        if not key:
            raise RuntimeError(f"No {provider} API key entered")
    _KEYS[provider] = key
    return key


def response_schema(kind, allowed=None, choices=None):
    if kind == "taxonomy":
        return {"type": "object", "properties": {"categories": {
            "type": "array", "minItems": 1, "maxItems": 12, "items": {
                "type": "object", "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "criteria": {"type": "string", "minLength": 1, "maxLength": 400}},
                "required": ["name", "criteria"], "additionalProperties": False}}},
            "required": ["categories"], "additionalProperties": False}
    if kind in {"categories", "features"}:
        return {"type": "object", "properties": {kind: {
            "type": "array", "items": {"type": "string", "enum": sorted(FEATURES) if kind == "features" else list(allowed)}}},
            "required": [kind], "additionalProperties": False}
    if kind in {"sound", "precise"}:
        return {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["YES", "NO"]}},
                "required": ["verdict"], "additionalProperties": False}
    if kind == "judge":
        properties = {key: {"type": "string", "enum": list(spec["options"])} for key, spec in choices.items()}
        properties["rationale"] = {"type": "string"}
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    return None


def portable(schema):
    """Drop count/length limits hosted structured-output modes may reject; parse_response still enforces them."""
    if isinstance(schema, dict):
        return {k: portable(v) for k, v in schema.items() if k not in {"minItems", "maxItems", "minLength", "maxLength"}}
    if isinstance(schema, list):
        return [portable(v) for v in schema]
    return schema


def jev_questions(kind, names, choices):
    """Map each response contract onto Typesafe choice questions; the prompt travels as `state`."""
    yes_no = {"YES": "The statement holds.", "NO": "The statement does not hold."}
    ask = lambda instructions, criteria: {"type": "choice", "instructions": instructions, "criteria": criteria}
    if kind in {"sound", "precise"}:
        return {"verdict": ask(f"Using the definition in the state, is this recommendation {kind.upper()}?", yes_no)}
    if kind in {"categories", "features"}:
        noun = "category" if kind == "categories" else "socio-technical feature"
        return {f"q{i}": ask(f"Does the action in the state belong to the {noun} '{name}'?", yes_no) for i, name in enumerate(names)}
    return {key: ask(spec["question"], spec["options"]) for key, spec in choices.items()}


def jev_answer(reply, kind, questions, names):
    answers = reply.get("answers") if isinstance(reply, dict) else None
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Jev response is missing question answers")
    picked = {}
    for key, question in questions.items():
        choice = answers[key].get("choice") if isinstance(answers[key], dict) else None
        if choice not in question["criteria"]:
            raise ValueError(f"Invalid Jev choice for {key}")
        picked[key] = choice
    if kind in {"sound", "precise"}:
        return {"verdict": picked["verdict"]}
    if kind in {"categories", "features"}:
        return {kind: [name for i, name in enumerate(names) if picked[f"q{i}"] == "YES"]}
    return {**picked, "rationale": ""}


class ModelClient:
    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.cache_hits = 0
        self.sdk = {}

    def list_models(self):
        """Return installed model names from the configured Ollama server."""
        url = self.config.ollama_url.rstrip("/") + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=self.config.timeout) as response:
                models = json.load(response)["models"]
            if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("name"), str) for m in models):
                raise ValueError("Invalid model list")
            return sorted({m["name"] for m in models})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Cannot list models at {url}. Start Ollama (ollama serve) or check the server URL: {exc}") from exc

    def pull_model(self, model, progress=None):
        """Install a model, consuming Ollama's streamed download progress."""
        request = urllib.request.Request(self.config.ollama_url.rstrip("/") + "/api/pull",
            data=json.dumps({"model": model, "stream": True}).encode(),
            headers={"Content-Type": "application/json"})
        success = False
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                for line in response:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("Invalid download progress response")
                    if event.get("error"):
                        raise ValueError(event["error"])
                    if progress:
                        progress(event)
                    if event.get("status") == "success":
                        success = True
            if not success:
                raise ValueError("Download ended without confirmation of success; retry to resume")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Could not install {model}: {detail or str(exc)}") from exc
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"Could not install {model}: {exc}") from exc

    def preflight(self, models):
        """Fail before any slow or paid call: Ollama models installed, SDKs present, keys available."""
        local = [split_model(m)[1] for m in models if split_model(m)[0] == "ollama"]
        if local:
            available = self.list_models()
            missing = [m for m in local if m not in available and m + ":latest" not in available]
            if missing:
                raise RuntimeError("Models missing from Ollama; pull them first: " + ", ".join(missing))
        for provider in sorted({split_model(m)[0] for m in models} - {"ollama"}):
            if provider == "jev":
                api_key(provider)
            else:
                self.sdk_client(provider)

    def sdk_client(self, provider):
        if provider not in self.sdk:
            try:
                module = __import__(provider)
            except ImportError as exc:
                raise RuntimeError(f'{provider} models require: pip install "react-gpt[{provider}]"') from exc
            factory = module.Anthropic if provider == "anthropic" else module.OpenAI
            # ask() owns retries so every failed attempt is recorded under errors/.
            self.sdk[provider] = factory(api_key=api_key(provider), max_retries=0, timeout=self.config.timeout)
        return self.sdk[provider]

    def ask(self, model, prompt, kind, *, categories=None, choices=None):
        c = self.config
        provider, name = split_model(model)
        if provider == "jev" and kind in TEXT_KINDS:
            raise ValueError(f"{model} can only evaluate or judge; use an Ollama, Anthropic, or OpenAI model for {kind}")
        allowed = domain_categories(c.domain) if categories is None else categories
        schema = response_schema(kind, allowed, choices)
        request = {"contract": 2, "provider": provider, "model": name, "kind": kind, "prompt": prompt, "schema": schema}
        if provider == "ollama":
            request.update(endpoint=c.ollama_url, options={"temperature": c.temperature, "num_ctx": c.num_ctx,
                           "num_predict": c.num_predict, "think": c.think})
        elif provider == "jev":
            request["names"] = sorted(FEATURES) if kind == "features" else list(allowed) if kind == "categories" else None
            request["questions"] = jev_questions(kind, request["names"], choices)
        key = digest(request)
        path = Path(c.output) / "cache" / "requests" / f"{key}.json"
        if path.exists():
            result = parse_response(read_json(path)["response"], kind, allowed, choices)
            self.cache_hits += 1
            return result
        for attempt in range(c.retries + 1):
            raw = None
            try:
                self.calls += 1
                raw, metadata = getattr(self, "_" + provider)(request)
                result = parse_response(raw, kind, allowed, choices)
                write_json(path, {"model": model, "kind": kind, "prompt": prompt, "response": raw, "metadata": metadata})
                return result
            except Exception as exc:  # SDK, HTTP, and contract errors share one bounded retry policy.
                write_json(Path(c.output) / "errors" / f"{key}-{attempt}.json",
                           {"model": model, "kind": kind, "error": f"{type(exc).__name__}: {exc}", "response": raw})
                if attempt == c.retries:
                    raise RuntimeError(f"{kind} failed for {model} after {attempt+1} attempts: {exc}") from exc
                time.sleep(min(2 ** attempt, 10))

    def _ollama(self, request):
        c = self.config
        payload = {"model": request["model"], "prompt": request["prompt"], "stream": False,
                   "format": request["schema"] or "json", "think": c.think, "keep_alive": c.keep_alive,
                   "options": {"temperature": c.temperature, "num_ctx": c.num_ctx, "num_predict": c.num_predict}}
        http = urllib.request.Request(c.ollama_url.rstrip("/") + "/api/generate",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=c.timeout) as response:
            envelope = json.load(response)
        raw = envelope.get("response")
        if not isinstance(raw, str) or not envelope.get("done", False):
            raise ValueError("Incomplete Ollama response")
        if envelope.get("done_reason") == "length":
            raise ValueError("Output truncated; increase num_predict")
        return raw, {k: v for k, v in envelope.items() if k not in {"response", "thinking", "context"}}

    def _anthropic(self, request):
        kwargs = {"model": request["model"], "max_tokens": max(self.config.num_predict, 16000),
                  "messages": [{"role": "user", "content": request["prompt"]}]}
        if request["schema"]:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": portable(request["schema"])}}
        message = self.sdk_client("anthropic").messages.create(**kwargs)
        if message.stop_reason == "max_tokens":
            raise ValueError("Output truncated; increase num_predict")
        if message.stop_reason == "refusal":
            raise ValueError("The model declined this request")
        raw = "".join(block.text for block in message.content if block.type == "text")
        return raw, {"id": message.id, "model": message.model, "stop_reason": message.stop_reason,
                     "usage": message.usage.to_dict()}

    def _openai(self, request):
        schema = request["schema"]
        response_format = ({"type": "json_schema", "json_schema": {"name": request["kind"], "schema": portable(schema), "strict": True}}
                           if schema else {"type": "json_object"})
        completion = self.sdk_client("openai").chat.completions.create(
            model=request["model"], messages=[{"role": "user", "content": request["prompt"]}],
            response_format=response_format, max_completion_tokens=max(self.config.num_predict, 16000))
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("Output truncated; increase num_predict")
        if not choice.message.content:
            raise ValueError(getattr(choice.message, "refusal", None) or "Empty response")
        return choice.message.content, {"id": completion.id, "model": completion.model,
                                        "usage": completion.usage.to_dict() if completion.usage else None}

    def _jev(self, request):
        body = json.dumps({"model": request["model"], "state": request["prompt"], "questions": request["questions"]},
                          ensure_ascii=False).encode()
        http = urllib.request.Request(JEV_ENDPOINT, data=body,
            headers={"Authorization": "Bearer " + api_key("jev"), "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(http, timeout=self.config.timeout) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            # Never echo server error bodies; they could reflect credentials.
            raise RuntimeError(f"Typesafe HTTP {exc.code}") from None
        answer = jev_answer(reply, request["kind"], request["questions"], request["names"])
        return json.dumps(answer), {"model": reply.get("model"), "usage": reply.get("usage")}
