#!/bin/bash
# Stage the build context and build witake:latest.
#   STAGE=<dir with hfsprint/ hfm/ data/vinv-tts-hf/ from setup_box.sh's hf downloads> bash docker/build.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); STAGE=${STAGE:-$HOME/vinvroot}; CTX=${CTX:-$HOME/witake_build}; TAG=${TAG:-witake:latest}
rm -rf "$CTX"; mkdir -p "$CTX/root"
cp -r "$REPO/vinvtts" "$CTX/code"; rm -rf "$CTX/code/.venv"; cp -r "$REPO/docker" "$CTX/docker"
# hard links: no copy of the ~15 GB of weights; drop HF download caches and unserved LoRA adapters
for d in hfsprint hfm data; do cp -al "$STAGE/$d" "$CTX/root/$d"; done
find "$CTX/root" -name .cache -type d -prune -exec rm -rf {} +
rm -rf "$CTX"/root/hfsprint/models/omnivoice/cdpo1b/adapter-* "$CTX"/root/hfm/omnivoice/omft_tdl4000   # omft_tdl4000: training base only
docker build -t "$TAG" -f "$CTX/docker/Dockerfile" "$CTX"
