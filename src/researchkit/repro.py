"""Reproducibility primitives.

Every experiment run should call `set_global_seed(...)` first and dump
`capture_env()` into its results, so a result is always traceable to an exact
(seed, code, environment) triple. Without this, a number is just a number.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from typing import Any


def set_global_seed(seed: int) -> int:
    """Seed Python, NumPy and (if present) PyTorch. Returns the seed used."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism costs speed; opt in per experiment if the hypothesis
        # depends on bitwise reproducibility.
    except ImportError:
        pass
    return seed


def git_commit() -> str:
    """Short commit hash of the repo, or 'nogit' / '<hash>-dirty'."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        )
        return f"{h}-dirty" if dirty else h
    except Exception:
        return "nogit"


def capture_env() -> dict[str, Any]:
    """A snapshot to paste into an experiment's results section."""
    return {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
    }


if __name__ == "__main__":
    print(json.dumps(capture_env(), indent=2))
