from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    id: str = "Horizon-Labs/multilingual-embedding-small"
    path: str | None = None
    max_length: int = 512
    inference_batch_size: int = 32


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    threads: int | None = None
    max_batch_size: int = 128


@dataclass
class Config:
    model: ModelConfig
    server: ServerConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        model=ModelConfig(**data.get("model", {})),
        server=ServerConfig(**data.get("server", {})),
    )
