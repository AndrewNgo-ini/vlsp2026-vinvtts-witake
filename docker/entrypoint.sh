#!/bin/bash
# Container entrypoint: /input (ro) -> /output. Time budget counts from here (T0).
export T0=$(date +%s.%N)
mkdir -p /tmp/logs /output
cd /opt/vinvtts-submission/vinvtts
echo "ViNV-TTS (team witake) start $(date -u +%FT%TZ)"; nvidia-smi -L 2>&1 | head -2
.venv/bin/python run_batch.py --input /input --output /output
rc=$?
echo "run_batch exit $rc"
exit $rc
