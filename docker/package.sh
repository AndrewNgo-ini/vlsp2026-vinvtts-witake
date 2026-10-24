#!/bin/bash
# Export witake:latest into the submission zip (one file at the zip root: witake.tar) + its sha256.
#   bash docker/package.sh [out_dir]      -> <out_dir>/witake_submission.zip, <out_dir>/witake_submission.zip.sha256
set -euo pipefail
TEAM=${TEAM:-witake}; OUT=${1:-$HOME/witake_submission}
mkdir -p "$OUT"; cd "$OUT"; rm -f "$TEAM.tar" "${TEAM}_submission.zip"
docker image inspect "$TEAM:latest" > /dev/null
docker save "$TEAM:latest" -o "$TEAM.tar"
# zip64, deflate level 1; witake.tar at the zip root, nothing else (python zipfile: no zip binary needed)
python3 -c "
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z: z.write(sys.argv[2], arcname=sys.argv[2])
with zipfile.ZipFile(sys.argv[1]) as z: print([(i.filename, i.file_size, i.compress_size) for i in z.infolist()])" "${TEAM}_submission.zip" "$TEAM.tar"
rm -f "$TEAM.tar"
sha256sum "${TEAM}_submission.zip" | tee "${TEAM}_submission.zip.sha256"
