"""Configuration: config.yaml + env var OPENROUTER_API_KEY."""

import os
from pathlib import Path

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


class Config(BaseModel):
    openrouter: OpenRouterConfig = OpenRouterConfig()
    api_keys: list[str] = Field(default_factory=list)
    ladder: list[LadderStep]
    classifier: ClassifierConfig
    judge: JudgeConfig
    pricing_refresh_hours: float = 24
    request_timeout_seconds: float = 120
    db_path: str = "ratatosk.sqlite3"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("RATATOSK_CONFIG", "config.yaml"))
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    config = Config.model_validate(data)
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        config.openrouter.api_key = env_key
    return config
