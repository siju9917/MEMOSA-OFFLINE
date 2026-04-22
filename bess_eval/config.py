"""Typed site configuration loaded from YAML."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Thin dict wrapper with dotted access and YAML load."""

    def __getattr__(self, k: str) -> Any:
        if k in self:
            v = self[k]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(k)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(raw)
