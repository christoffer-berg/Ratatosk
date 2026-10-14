"""Configuration: config.yaml + env var OPENROUTER_API_KEY."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class OpenRouterConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""


class LadderStep(BaseModel):
    id: str


class ClassifierConfig(BaseModel):
    model: str
    tiers: dict[str, int] = Field(
        default={"trivial": 0, "easy": 0, "medium": 1, "hard": 2, "expert": 3}
    )


class JudgeConfig(BaseModel):
    model: str
    threshold: int = 7
    strict_threshold: int = 8
    cheap_threshold: int = 6
    max_escalations: int = 3
    cheap_max_escalations: int = 1


class LearningConfig(BaseModel):
    """Start-rung selection learned from logged attempts (see learning.py)."""

    enabled: bool = True
    # "expected_cost": minimise own cost + (1 - pass rate) * cost of escalating.
    # "pass_rate": cheapest rung clearing min_pass_rate.
    policy: Literal["expected_cost", "pass_rate"] = "expected_cost"
    min_samples: int = 20  # judged attempts needed before a cell overrides config
    min_pass_rate: float = 0.8  # only used by the "pass_rate" policy
    lookback_days: float = 30
    refresh_minutes: float = 60


class ExplorationConfig(BaseModel):
    """Epsilon-greedy: occasionally start one rung below the chosen one.

    Exploring downward is cheap and safe — the judge escalates back up, so the
    worst case is one extra attempt at a cheaper model. That buys continuously
    fresh pass-rate data for rungs the table would otherwise stop sampling.
    """

    enabled: bool = True
    # Probability of starting one rung lower. It is also the rate at which the
    # local model (predictor) gathers the evidence it needs to deploy: every
    # downward move it makes is validated against explored requests.
    epsilon: float = 0.10


class VerifierConfig(BaseModel):
    """Deterministic checks used as free, unfoolable judges."""

    enabled: bool = True
    # A hard failure (code that will not parse, arithmetic that is wrong)
    # forces escalation even when the LLM judge was satisfied.
    enforce: bool = True


class AuditConfig(BaseModel):
    """Shadow audits: re-judge a sample of accepted answers with a strong model.

    Runs after the client has its response, so it costs latency nothing. Measures
    how often the cheap judge accepts an answer the strong judge would reject.
    """

    enabled: bool = False
    model: str = ""  # defaults to the most expensive ladder rung
    sample_pct: float = 0.02


class FeedbackConfig(BaseModel):
    """Ground-truth labels from clients, which beat the judge as a signal."""

    enabled: bool = True
    # Infer a negative label when the next user turn reads as "that's wrong".
    implicit: bool = True
    max_regret_chars: int = 200  # only short follow-ups count as a reaction


class EmbeddingsConfig(BaseModel):
    """Nearest-neighbour routing over logged prompts, plus a semantic cache.

    Off by default: it adds an embedding call per request, and the matrix in
    learning.py already captures most of the win at low traffic.
    """

    enabled: bool = False
    base_url: str = ""  # defaults to openrouter.base_url
    api_key: str = ""  # defaults to openrouter.api_key
    model: str = "openai/text-embedding-3-small"
    dimensions: int | None = 256  # smaller vectors keep the pure-Python scan fast
    neighbors: int = 10
    min_similarity: float = 0.75  # neighbours below this are ignored
    min_neighbors: int = 5  # fewer than this ⇒ defer to the skill matrix
    max_vectors: int = 2000  # most recent N kept in memory for the scan
    refresh_minutes: float = 60
    cache_enabled: bool = False  # reuse the answer from a near-identical prompt
    cache_similarity: float = 0.97


class PredictorConfig(BaseModel):
    """A small local model trained on the log (`python -m app.train`).

    Predicts, per prompt, which ladder rung will pass — and routes on it once a
    training run has shown it beats the static tiers on held-out traffic.
    """

    enabled: bool = True  # route with the model whenever a deployable one exists
    path: str = "router_model.json"
    log_features: bool = True  # store hashed prompt features (not text) for training
    buckets: int = 8192  # hashed n-gram slots; changing this needs a retrain
    policy: Literal["expected_cost", "pass_rate"] = "expected_cost"
    min_pass_prob: float = 0.8  # only used by the "pass_rate" policy
    min_examples: int = 200  # training refuses to produce a model from fewer
    holdout: float = 0.2  # newest fraction of the log kept out of training
    epochs: int = 8
    learning_rate: float = 0.3
    l2: float = 1e-4
    impute: bool = True  # fill unobserved rungs using the ladder's monotonicity
    imputed_weight: float = 0.5  # how much an imputed label counts vs an observed one
    lookback_days: float = 90
    # Route even if the last training run failed its deployment check.
    allow_undeployable: bool = False


class Config(BaseModel):
    openrouter: OpenRouterConfig = OpenRouterConfig()
    api_keys: list[str] = Field(default_factory=list)
    ladder: list[LadderStep]
    classifier: ClassifierConfig
    judge: JudgeConfig
    learning: LearningConfig = LearningConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    verifiers: VerifierConfig = VerifierConfig()
    audit: AuditConfig = AuditConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    predictor: PredictorConfig = PredictorConfig()
    pricing_refresh_hours: float = 24
    request_timeout_seconds: float = 120
    db_path: str = "ratatosk.sqlite3"
    log_prompts: bool = False  # store prompt text in the log (off: length only)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("RATATOSK_CONFIG", "config.yaml"))
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    config = Config.model_validate(data)
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        config.openrouter.api_key = env_key
    return config
