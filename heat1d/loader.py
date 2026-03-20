from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .schema import SimulationInput


def _resolve_paths(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        resolved = {key: _resolve_paths(item, base_dir) for key, item in value.items()}
        file_value = resolved.get("file")
        if isinstance(file_value, str):
            candidate = Path(file_value)
            if not candidate.is_absolute():
                resolved["file"] = base_dir / candidate
        return resolved
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir) for item in value]
    return value


def load(yaml_path: Path) -> SimulationInput:
    yaml_path = Path(yaml_path).expanduser().resolve()
    yaml = YAML(typ="safe")
    with yaml_path.open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle) or {}
    resolved = _resolve_paths(raw, yaml_path.parent)
    return SimulationInput.model_validate(resolved)
