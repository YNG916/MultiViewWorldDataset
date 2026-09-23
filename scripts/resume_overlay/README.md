# Resuming the frozen Dataset-v1.1 production root

The `dataset_v11_production_full_20260923` root was created with source fingerprint
`7168c593cdec4f795f047aca388de9a678a0bbd86967d5811d75d3cf29f2e77a`.
The current `src/` fixes redundant resume work and therefore has a different fingerprint.
Do not point the current source directly at that root or rewrite its manifest.

The root contains `global/frozen_producer_469064fa/`, an exact copy of the
original producer, and `global/resume_overlay/`, a narrow operational overlay.
The overlay skips navigation-bank rebuilding only if a configuration has its
metadata, navigation metadata, and every requested atomically committed episode.
It does not change sampling for incomplete or new configurations. It records
its own SHA-256, the frozen source/configuration fingerprints, and skipped IDs
in each shard's `resume_overlay_audit.json`.

Use one coordinator per root, preferably under `tmux` or a batch scheduler.
On a new node, install the same OmniGibson/Isaac environment and assets first.
Put simulator scratch on node-local storage; only the final dataset root must
be persistent. For example, after the previous coordinator has exited:

```bash
DATASET_ROOT=/path/to/dataset_v11_production_full_20260923
LOCAL_SCRATCH=/local/scratch/mvwd-final-runtime
mkdir -p "$LOCAL_SCRATCH/tmp" "$LOCAL_SCRATCH/xdg" \
  "$LOCAL_SCRATCH/og-appdata" "$LOCAL_SCRATCH/cache"
export TMPDIR="$LOCAL_SCRATCH/tmp"
export XDG_CACHE_HOME="$LOCAL_SCRATCH/xdg"
export OMNIGIBSON_APPDATA_PATH="$LOCAL_SCRATCH/og-appdata"
export MVWD_RESUME_OVERLAY_ROOT="$DATASET_ROOT"
export PYTHONPATH="$DATASET_ROOT/global/resume_overlay:$DATASET_ROOT/global/frozen_producer_469064fa/src"
cd "$DATASET_ROOT/global/frozen_producer_469064fa"
python -m multi_view_world_dataset.cli production-launch \
  --config configs/production_v1.yaml --scenes Beechwood_0_int,Rs_int \
  --gpus 0,1 --max-workers 2 --allow-large --retry-failed \
  --output-root "$DATASET_ROOT" --cache-root "$LOCAL_SCRATCH/cache"
```

Check that both worker logs say `[mvwd-resume-overlay] active`, that
`generation_status.json` reports the prior accepted counts after scene loading,
and that the audit lists only already complete configurations. Existing
`generation_failure.json` may refer to an earlier retry; it is not proof the
current worker failed. New roots should use the current generator without this
overlay. Do not run both source versions into one root concurrently.
