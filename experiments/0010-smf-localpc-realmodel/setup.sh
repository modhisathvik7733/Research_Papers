#!/usr/bin/env bash
# 0010 setup — run ONCE on the A100 box. Clones the official SMF repo at a
# PINNED commit, installs deps, drops in the Local-PC optimiser + wrapper.
# No fork: Local-PC is injected at runtime via monkeypatch (smf_localpc.py).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMF_DIR="${SMF_DIR:-$HERE/SMF-ICML}"
PIN="3237f35db38f0e04b221ea6b7fd126f36b9727b6"   # pinned for reproducibility

if [ ! -d "$SMF_DIR/.git" ]; then
  git clone https://github.com/prakharg55/SMF-ICML "$SMF_DIR"
fi
cd "$SMF_DIR"
git fetch --all --tags 2>/dev/null || true
git checkout "$PIN"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Local-PC lives at SMF repo root so `from local_pc import LocalPC` resolves
# when cwd = repo root (how run_0010.sh invokes everything).
cp "$HERE/local_pc.py"     "$SMF_DIR/local_pc.py"
cp "$HERE/smf_localpc.py"  "$SMF_DIR/smf_localpc.py"

python -c "import torch,transformers,accelerate,datasets; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available(), \
'transformers',transformers.__version__)"
python -m py_compile local_pc.py smf_localpc.py
echo "SETUP OK. Pinned SMF @ $PIN. Next: bash $HERE/run_0010.sh smoke"
