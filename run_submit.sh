#!/bin/bash
# Run post-training's scripts/submit.py on the MN5 login node.
#
#   ./run_submit.sh --config configs/trl/sft-mn5.yaml [key=value overrides...]
#
# MUST be run over a proxy-carrying connection for prefetch to reach HF:
#   ssh -S none -o ControlMaster=no mn5 \
#     '/gpfs/scratch/ehpc532/eit234648/post-training/run_submit.sh --config configs/trl/sft-mn5.yaml --confirm'
#
# Notes:
# - Uses the py3.12 host venv (.submit-venv). The MN5 python/3.12.1 module only
#   sets LIBRARY_PATH (compile-time), so we add its lib dir to LD_LIBRARY_PATH
#   at runtime, otherwise libpython3.12.so.1.0 is not found.
# - The login shell already exports HTTP_PROXY/HTTPS_PROXY=localhost:8888, which
#   huggingface_hub uses for prefetch. (Do NOT set HF_HUB_ENABLE_HF_TRANSFER — the
#   rust downloader does not honor the proxy reliably.)
set -euo pipefail
REPO=/gpfs/scratch/ehpc532/eit234648/post-training
export LD_LIBRARY_PATH=/apps/GPP/PYTHON/3.12.1/INTEL/lib:${LD_LIBRARY_PATH:-}
# shellcheck disable=SC1091
source "$REPO/.submit-venv/bin/activate"
unset PYTHONPATH
export PYTHONPATH="$REPO/src"
cd "$REPO"
exec python scripts/submit.py "$@"
