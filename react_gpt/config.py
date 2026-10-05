"""Portable, serializable run configuration."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
from urllib.parse import urlparse


@dataclass
class Config:
    corpus: str
    output: str
    models: list[str]
    evaluators: list[str] = field(default_factory=list)
    ollama_url: str = "http://localhost:11434"
    prompting: str = "CoT"
    temperature: float = 0.0
    timeout: float = 600.0
    retries: int = 2
    keep_alive: str = "5m"
    num_ctx: int = 32768
    num_predict: int = 4096
    chunk_chars: int = 24000
    similarity: str = "hybrid"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = 0.65
    benchmark: str | None = None
    intra_model_alpha: float = 0.15
    min_support: int = 1
    features: bool = False
    limit: int | None = None
    domain: str = "oss"
    category_mode: str = "fixed"
    judge: str | None = None
    think: bool = False

    def validate(self):
        if self.category_mode not in {"fixed", "dynamic"}:
            raise ValueError("category_mode must be fixed or dynamic")
        if self.domain not in {"oss", "medicine"}:
            raise ValueError("domain must be oss or medicine")
        if self.domain == "medicine" and self.features:
            raise ValueError("Socio-technical feature mapping is only available for the oss domain")
        for name in ("models", "evaluators"):
            values = getattr(self, name)
            if not isinstance(values, list) or any(not isinstance(m, str) or not m.strip() for m in values):
                raise ValueError(f"{name} must be a list of nonempty model names")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicate models")
        if not self.models:
            raise ValueError("At least one extraction model is required")
        if self.judge is not None and (not isinstance(self.judge, str) or not self.judge.strip()):
            raise ValueError("judge must be a model name")
        from .models import split_model
        for name in self.models + self.evaluators + ([self.judge] if self.judge else []):
            split_model(name)
        if any(split_model(m)[0] == "jev" for m in self.models):
            raise ValueError("Jev models can only evaluate or judge; choose Ollama, Anthropic, or OpenAI extraction models")
        if not self.evaluators:
            self.evaluators = list(self.models)
        if self.prompting not in {"IP", "CoT", "RA"} or self.similarity not in {"hybrid", "lexical"}:
            raise ValueError("Invalid prompting or similarity mode")
        for name in ("threshold", "intra_model_alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("num_ctx", "num_predict", "chunk_chars", "min_support"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_support > len(self.models):
            raise ValueError("min_support exceeds extraction panel size")
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError("retries must be a nonnegative integer")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise ValueError("limit must be a positive integer")
        for name in ("temperature", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.timeout == 0 or type(self.features) is not bool or type(self.think) is not bool:
            raise ValueError("timeout must be positive; features and think must be boolean")
        url = urlparse(self.ollama_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("ollama_url must be an http(s) URL")
        corpus, output = Path(self.corpus).resolve(), Path(self.output).resolve()
        if not corpus.exists():
            raise ValueError(f"Corpus does not exist: {corpus}")
        if output == corpus or (corpus.is_dir() and corpus in output.parents):
            raise ValueError("Output must be outside the corpus directory")
        self.corpus, self.output = str(corpus), str(output)
        if self.benchmark:
            self.benchmark = str(Path(self.benchmark).resolve())
            if not Path(self.benchmark).is_file():
                raise ValueError(f"Benchmark does not exist: {self.benchmark}")
        return self

    def to_dict(self):
        values = asdict(self)
        if self.category_mode == "fixed":
            values.pop("category_mode")
        if self.domain == "oss":
            values.pop("domain")  # Preserve existing OSS run fingerprints.
        return values
