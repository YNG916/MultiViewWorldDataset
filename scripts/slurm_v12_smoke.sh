#!/usr/bin/env bash
# One isolated final-robot v1.2 smoke episode on an allocated Slurm GPU.
set -euo pipefail
umask 077

base=/home/usqki/behavior_world
repo=/home/usqki/MultiViewWorldDataset
env_prefix=/tmp/mvwd-env-usqki-v392
assets=/tmp/mvwd-assets-usqki-v392

: "${SLURM_JOB_ID:?Run through sbatch}"
: "${CUDA_VISIBLE_DEVICES:?A GPU allocation is required}"
: "${MVWD_SCENE:?Set one scene}"
: "${MVWD_OUTPUT_ROOT:?Set a fresh output root}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Exactly one GPU is required" >&2; exit 2; }

if [[ ! -x "$env_prefix/bin/python" ]]; then
    tar -C /tmp -xf "$base/env_archives/behavior_env_v392_old_parity_20260924.tar"
fi
if [[ ! -f "$assets/behavior-1k-assets/VERSION" ]]; then
    mkdir -p -m 700 "$assets"
    tar -C "$assets" -xf "$base/assets_archives/behavior_datasets_v3.9.2.tar"
fi
if [[ ! -f "$assets/omnigibson.key" ]]; then
    cp -p "$base/software/BEHAVIOR-1K/datasets/omnigibson.key" "$assets/omnigibson.key"
fi

runtime="/tmp/mvwd-v12-smoke-${SLURM_JOB_ID}"
mkdir -p -m 700 "$runtime"/{tmp,xdg,og-appdata,cache}
export TMPDIR="$runtime/tmp"
export XDG_CACHE_HOME="$runtime/xdg"
export OMNIGIBSON_APPDATA_PATH="$runtime/og-appdata"
export OMNIGIBSON_DATA_PATH="$assets"
export BEHAVIOR_ROOT="$base/software/BEHAVIOR-1K"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="$env_prefix/lib/python3.11/site-packages/pymeshlab/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo/src"
export OMNIGIBSON_GPU_ID="$CUDA_VISIBLE_DEVICES"
unset CUDA_VISIBLE_DEVICES

source "$base/miniforge3/etc/profile.d/conda.sh"
conda activate "$env_prefix"
cd "$repo"
python -m multi_view_world_dataset.cli generate \
    --config configs/final_robot_v12_nonrigid_smoke.yaml \
    --scene "$MVWD_SCENE" --behavior-root "$BEHAVIOR_ROOT" \
    --output-root "$MVWD_OUTPUT_ROOT" --cache-root "$runtime/cache"
jq -e '.status == "pass" and .accepted_configurations == 1 and .accepted_episodes == 1' \
    "$MVWD_OUTPUT_ROOT/generation_result.json" >/dev/null
