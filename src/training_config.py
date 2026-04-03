"""
Load training / RL settings from ``training_config.json`` (or a path you pass in).

Edit the JSON to change runs without touching code. Unknown keys are ignored so you
can add notes in a copy of the file if you use a strict JSON parser elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_JSON = _CONFIG_DIR / "training_config.json"


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int
    headless: bool
    max_episode_steps: int
    starting_tiles_on_bench: int
    max_bench_tiles: int
    random_seed: Optional[int]
    ppo_verbose: int


def _defaults() -> dict[str, Any]:
    return {
        "total_timesteps": 40_000,
        "headless": False,
        "max_episode_steps": 100,
        "starting_tiles_on_bench": 10,
        "max_bench_tiles": 32,
        "random_seed": None,
        "ppo_verbose": 1,
    }


def load_training_config(path: Optional[str | Path] = None) -> TrainingConfig:
    """
    Merge JSON file over built-in defaults and return a ``TrainingConfig``.

    Parameters
    ----------
    path:
        JSON file path. Defaults to ``src/training_config.json`` next to this module.
    """
    data = dict(_defaults())
    json_path = Path(path) if path is not None else _DEFAULT_JSON
    if json_path.is_file():
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Config root must be a JSON object, got {type(raw)}")
        for key, value in raw.items():
            if key in data:
                data[key] = value

    return TrainingConfig(
        total_timesteps=int(data["total_timesteps"]),
        headless=bool(data["headless"]),
        max_episode_steps=int(data["max_episode_steps"]),
        starting_tiles_on_bench=int(data["starting_tiles_on_bench"]),
        max_bench_tiles=int(data["max_bench_tiles"]),
        random_seed=None if data["random_seed"] is None else int(data["random_seed"]),
        ppo_verbose=int(data["ppo_verbose"]),
    )
