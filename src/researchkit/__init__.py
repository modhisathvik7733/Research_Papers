"""researchkit — shared, reused utilities across experiments.

Rule: code goes here only if >1 experiment uses it. Experiment-specific code
lives in the experiment folder. This package exists to make reproducibility the
default, not an afterthought.
"""

from researchkit.repro import capture_env, git_commit, set_global_seed

__version__ = "0.1.0"

__all__ = ["set_global_seed", "capture_env", "git_commit", "__version__"]
