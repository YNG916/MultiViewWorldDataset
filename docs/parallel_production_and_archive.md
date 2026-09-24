# Parallel Dataset-v1.1 production, archiving, and version collections

## Producer versions

The existing `dataset_v11_production_full_20260923` root is locked to its
original producer fingerprint. Never launch the revised route-feasibility
producer into that root. A new source/config fingerprint needs a new root.
Keep a frozen copy of `src/`, `configs/`, `assets/`, and `pyproject.toml`
under the new root's `global/` directory, and launch from that copy for
every batch and resume. Source edits during a batch otherwise block safe
retry.

The revised navigation preflight treats `route_bank_minimum_size: 24` as a
diagnostic, not a rejection gate. It requires a deterministic compatible
three-route set; the bounded episode search has an exact fallback. Robot
footprint, collision, kinematics, GT-depth, intervention, and paired-rollout
QA are unchanged. A functional triplet is necessary, but does not guarantee
that all three episodes in a configuration will pass later QA.

## One batch on the current node

Use GPU 0 and 1 for one scene worker each. GPU 2 is a GTX 1080 Ti whose
compute capability is unsupported by the installed PyTorch build. On this
node `/tmp` has very little free space; use node-local `/dev/shm` only for
temporary simulator data, not final episodes. It consumes RAM and disappears
on reboot. Check `df -h /dev/shm`, `free -h`, and `nvidia-smi` during the
first GPU smoke and the first two-worker batch.

After the new producer passes a final-robot GPU smoke, freeze it and start
one coordinator in `tmux`. Choose **scene IDs not already produced by the
old version** if the versions will later form one collection. For example:

```bash
MVWD_ROOT=/path/to/new-production-root
MVWD_SOURCE="$MVWD_ROOT/global/frozen_producer"
MVWD_SCRATCH=/dev/shm/mvwd-new-production
mkdir -p "$MVWD_SOURCE" "$MVWD_SCRATCH/tmp" "$MVWD_SCRATCH/xdg" \
  "$MVWD_SCRATCH/og-appdata" "$MVWD_SCRATCH/cache"
rsync -a --exclude='__pycache__' src configs assets pyproject.toml "$MVWD_SOURCE/"
source /path/to/behavior_world/activate.sh
export TMPDIR="$MVWD_SCRATCH/tmp"
export XDG_CACHE_HOME="$MVWD_SCRATCH/xdg"
export OMNIGIBSON_APPDATA_PATH="$MVWD_SCRATCH/og-appdata"
export PYTHONPATH="$MVWD_SOURCE/src"
cd "$MVWD_SOURCE"
tmux new-session -d -s mvwd-production -- \
  python -m multi_view_world_dataset.cli production-launch \
    --config configs/production_v1.yaml \
    --scenes Beechwood_1_int,Ihlen_0_int \
    --gpus 0,1 --max-workers 2 --allow-large --retry-failed \
    --output-root "$MVWD_ROOT" --cache-root "$MVWD_SCRATCH/cache"
```

Only one coordinator may own a root at a time. Further scene batches use
the **same frozen source, same root, and same config** with a new `--scenes`
list. A completed scene is skipped; an interrupted scene resumes committed
episodes. Do not start more workers than GPUs until actual per-GPU memory
and throughput are measured. Do not use this command for the old root:
its frozen producer and resume overlay are documented separately in
`scripts/resume_overlay/README.md`.

## Freeing space safely

Prefer a quota-backed, persistent HPC/project filesystem with a documented
backup policy as the primary archive. An external hard drive can be an
additional offline copy, but is a poor sole source for active production.
Copying to another directory on the **same full filesystem** does not free
space. Obtain the target's quota, free space, mount/access path, and checksum
capabilities before evicting anything.

Batch at **whole-scene** boundaries. After a scene reaches
`shard_status.json: {"status": "complete"}` and the coordinator for that
root has exited, use the archive utility. It refuses active coordinator
locks, partial scenes, failed QA, absent modalities, and fingerprint
mismatches. It copies the global manifest/frozen producer and full scene
shard, verifies the copy with rsync checksums, and writes per-file SHA-256
records. The default command never removes the source:

```bash
python scripts/archive_completed_shard.py archive \
  --dataset-root "$MVWD_ROOT" --archive-root /persistent/archive/new-version \
  --scene Beechwood_1_int
```

Only after independently checking the archive path and available backup,
an explicit second invocation may evict the verified source:

```bash
python scripts/archive_completed_shard.py archive \
  --dataset-root "$MVWD_ROOT" --archive-root /persistent/archive/new-version \
  --scene Beechwood_1_int --evict-after-verify \
  --confirm-scene Beechwood_1_int
```

Archived scenes are absent from the active root. Restore them before
`finalize-dataset`, pilot reports, or a consumer that expects one root:

```bash
python scripts/archive_completed_shard.py restore \
  --dataset-root "$MVWD_ROOT" --archive-root /persistent/archive/new-version \
  --scene Beechwood_1_int
```

If the active root runs out of space mid-scene, stop its coordinator,
free space by archiving another **completed** scene, then resume; never
move a live partial configuration/episode directory. If no complete scene
is available, migrate the entire stopped root to a larger filesystem
instead.

## Logical old/new collection

The stock `finalize-dataset` only merges shards with one configuration
fingerprint. To combine old and new producer versions, retain two separate
roots and build a provenance-preserving collection index:

```bash
python scripts/build_collection_index.py \
  --root /path/to/old-root --root "$MVWD_ROOT" \
  --output /path/to/collection_index.json --include-partial
```

The collection enforces the same schema, robot asset, and split mapping,
rejects duplicate scene IDs, and records each episode's source root and
fingerprint. It is a logical index, **not** a rewrite into one homogeneous
production root; consumers must resolve `root/relative_path`. Rebuild it
after roots are moved or new episodes are committed. Omit `--include-partial`
when only fully completed scenes should be indexed.
