#!/usr/bin/env bash
# Run one smoke or one independent scene-shard production coordinator.
set -euo pipefail
umask 077

base=/home/usqki/behavior_world
env_prefix=/tmp/mvwd-env-usqki-v392
assets=/tmp/mvwd-assets-usqki-v392
source_snapshot="$base/output/mvwd/dataset_v11_production_newcluster_20260924/global/frozen_producer"

: "${SLURM_JOB_ID:?Run through sbatch}"
: "${CUDA_VISIBLE_DEVICES:?A GPU allocation is required}"
: "${MVWD_MODE:?Set MVWD_MODE to smoke or production}"
: "${MVWD_SCENES:?Set MVWD_SCENES}"
: "${MVWD_OUTPUT_ROOT:?Set MVWD_OUTPUT_ROOT}"

# Some cluster nodes advertise a GPU to Slurm that the NVIDIA driver cannot
# enumerate. Fail before launching Isaac or touching dataset state.
allocated_gpus=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
visible_gpus=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
if (( visible_gpus != allocated_gpus )); then
    echo "GPU visibility mismatch: Slurm allocated $allocated_gpus, driver sees $visible_gpus; SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}" >&2
    exit 3
fi

# Slurm cleans this user's /tmp after the last job on a node exits.
# Reuse an already staged read-only environment; never unpack over a live one.
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
test -x "$env_prefix/bin/python"
test -f "$assets/omnigibson.key"

runtime="/tmp/mvwd-runtime-${SLURM_JOB_ID}"
mkdir -p -m 700 "$runtime"/{tmp,xdg,og-appdata,cache}
export TMPDIR="$runtime/tmp"
export XDG_CACHE_HOME="$runtime/xdg"
export OMNIGIBSON_APPDATA_PATH="$runtime/og-appdata"
export OMNIGIBSON_DATA_PATH="$assets"
export BEHAVIOR_ROOT="$base/software/BEHAVIOR-1K"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="$env_prefix/lib/python3.11/site-packages/pymeshlab/lib:${LD_LIBRARY_PATH:-}"
source "$base/miniforge3/etc/profile.d/conda.sh"
conda activate "$env_prefix"

if [[ "$MVWD_MODE" == smoke ]]; then
    [[ "$MVWD_SCENES" != *,* ]] || { echo 'Smoke accepts exactly one scene' >&2; exit 2; }
    export PYTHONPATH="$source_snapshot/src"
    export OMNIGIBSON_GPU_ID="$CUDA_VISIBLE_DEVICES"
    unset CUDA_VISIBLE_DEVICES
    cd "$source_snapshot"
    set +e
    python -m multi_view_world_dataset.cli generate \
        --config configs/final_robot_preview.yaml \
        --scene "$MVWD_SCENES" --behavior-root "$BEHAVIOR_ROOT" \
        --output-root "$MVWD_OUTPUT_ROOT" --cache-root "$runtime/cache"
    generator_exit=$?
    set -e
    if (( generator_exit != 0 )); then
        echo "Generator exited $generator_exit; checking persisted terminal result" >&2
    fi
    jq -e '.status == "pass" and .accepted_configurations == 1 and .accepted_episodes == 1' \
        "$MVWD_OUTPUT_ROOT/generation_result.json" >/dev/null
elif [[ "$MVWD_MODE" == production ]]; then
    producer="$MVWD_OUTPUT_ROOT/global/frozen_producer"
    test -f "$producer/pyproject.toml"
    diff -qr --exclude='__pycache__' --exclude='*.pyc' "$source_snapshot/src" "$producer/src" >/dev/null
    diff -qr "$source_snapshot/configs" "$producer/configs" >/dev/null
    diff -qr "$source_snapshot/assets" "$producer/assets" >/dev/null
    cmp "$source_snapshot/pyproject.toml" "$producer/pyproject.toml"
    export PYTHONPATH="$producer/src"
    cd "$producer"
    python -m multi_view_world_dataset.cli production-launch \
        --config configs/production_v1.yaml --scenes "$MVWD_SCENES" \
        --gpus "$CUDA_VISIBLE_DEVICES" \
        --max-workers "${MVWD_WORKERS:-1}" --allow-large --retry-failed \
        --output-root "$MVWD_OUTPUT_ROOT" --cache-root "$runtime/cache"
else
    echo "Unsupported MVWD_MODE=$MVWD_MODE" >&2
    exit 2
fi
