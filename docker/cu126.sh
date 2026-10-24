#!/bin/bash
# Swap torch / torchaudio / torchvision in a venv for the CUDA 12.6 builds of the same versions (grading driver 580 =
# CUDA <= 13.0; PyPI's default torch 2.14 wheels need CUDA 13.x libs), CUDA-13 nvidia-* wheels removed first
# (they share the nvidia/ lib dirs with the cu12 ones, so removing them afterwards deletes the cu12 libraries).
#   bash cu126.sh <venv dir>
set -euo pipefail
V=$1; PY=$V/bin/python
PK=$($PY -c "
import importlib.metadata as m
for p in ('torch', 'torchaudio', 'torchvision'):
    try: print(f'{p}=={m.version(p).split(\"+\")[0]}', end=' ')
    except m.PackageNotFoundError: pass")
CU13=$($PY -c "
import importlib.metadata as m
print(' '.join(sorted({d.metadata['Name'] for d in m.distributions() if d.metadata['Name'].lower().startswith('nvidia-') and not d.metadata['Name'].lower().endswith('-cu12')})))")
[ -n "$CU13" ] && uv pip uninstall -p $PY $CU13
uv pip install -p $PY $(for p in $PK; do echo --reinstall-package ${p%%==*}; done) $PK \
   --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
