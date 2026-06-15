"""YAML loading helpers for harness configuration."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from skulk_test_harness.models import HarnessConfig, ModelSetFile, TestSetFile

T = TypeVar("T", bound=BaseModel)


def load_yaml_model(path: Path, model_type: type[T]) -> T:
    """Load a YAML file and validate it as ``model_type``."""

    raw = yaml.safe_load(path.read_text()) or {}
    return model_type.model_validate(raw)


def load_config(path: Path | None = None) -> HarnessConfig:
    """Load the harness config, returning defaults when no file exists."""

    if path is None:
        path = Path("skulk-harness.yaml")
    if not path.exists():
        return HarnessConfig()
    return load_yaml_model(path, HarnessConfig)


def load_model_sets(path: Path) -> ModelSetFile:
    """Load named model sets."""

    return load_yaml_model(path, ModelSetFile)


def load_test_sets(path: Path) -> TestSetFile:
    """Load named test sets."""

    return load_yaml_model(path, TestSetFile)

