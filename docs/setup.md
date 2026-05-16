# Setup — install everything

You have `uv` (0.11) and a Mac (M1 Max, Apple Silicon). All commands are run
from the repo root: `/Users/chintu/Research_Papers`.

## 1. One-time: create the environment

```bash
cd /Users/chintu/Research_Papers

# Base only (numpy/pyyaml/matplotlib + the local researchkit package).
# Fast; use this for reading/notes/design work.
uv sync

# Experiment environment (adds torch, torchvision, avalanche-lib, jupyterlab).
# Heavier (~a few GB, a few minutes). uv picks Python 3.12 automatically
# because pyproject caps <3.13 (avalanche-lib lags newest Python).
uv sync --extra experiments
```

`uv` creates `.venv/` in the repo and installs the local `researchkit`
package in editable mode, so `import researchkit` works inside the notebook.

## 2. Sanity-check the install

```bash
uv run python - <<'PY'
import torch, torchvision, avalanche, researchkit
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("avalanche   ", avalanche.__version__)
print("researchkit ", researchkit.__version__)
print("MPS available", torch.backends.mps.is_available())  # True on M1 Max
PY
```

Expected: all versions print, and **MPS available True** (Apple's GPU backend —
the notebook uses it automatically; MNIST is tiny so CPU is fine too).

## 3. Launch the notebook

```bash
uv run jupyter lab experiments/0001-reproduce-forgetting/run.ipynb
```

Or register the kernel for VS Code / any Jupyter client:

```bash
uv run python -m ipykernel install --user \
  --name research-papers --display-name "research-papers (.venv)"
```

then open `run.ipynb` and pick the **research-papers (.venv)** kernel.

## Apple-Silicon gotchas (read if install fails)

- **`avalanche-lib` won't resolve:** almost always a Python-version issue.
  Confirm uv used 3.12: `uv run python -V`. If it grabbed 3.13, run
  `uv python install 3.12 && uv venv --python 3.12` then re-sync.
- **torch wheel / architecture errors:** make sure you're on native arm64
  Python, not x86 under Rosetta: `uv run python -c "import platform;
  print(platform.machine())"` must print `arm64`.
- **MPS prints False:** fine for this experiment (MNIST is tiny) — it runs on
  CPU. Only matters once models get bigger.
- **`avalanche` import works but an API name differs:** Avalanche's API shifts
  between minor versions. Check `avalanche.__version__`, cross-check the
  [Avalanche docs](https://avalanche.continualai.org/) and the
  [continual-learning-baselines](https://github.com/ContinualAI/continual-learning-baselines)
  reference scripts. The notebook flags the version-sensitive cells.

## What got installed and why (no black boxes)

| Package | Why it's here |
|---|---|
| `torch`, `torchvision` | the model + MNIST data loaders |
| `avalanche-lib` | Split-MNIST benchmark, the `Naive` & `JointTraining` strategies, and forgetting/accuracy metrics — so experiment 0001 reproduces a *known* baseline instead of a hand-rolled one |
| `jupyterlab`, `ipykernel` | run the notebook |
| `researchkit` (local) | `set_global_seed` + `capture_env` so the run is reproducible |
