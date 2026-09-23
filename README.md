# MultiViewWorldDataset

`MultiViewWorldDataset` is a portable research dataset generator for synchronized, calibrated views from three mobile
robots in BEHAVIOR-1K / OmniGibson scenes. It generates counterfactual before/after rollouts using exactly the same
physical trajectories.

The model input contains **no robot ego-view image**. Dataset v1 is defined around a robot-free environment BEV
(`B_env`), calibrated target camera poses, and—at Level 2—one structured intervention. Post-intervention BEVs and
world BEVs are GT/oracle data, not normal model inputs.

## Quick start

Create and activate a compatible external simulator environment, then install this repository without replacing its
simulator packages:

```bash
python -m pip install -e .
cp env.example.sh /path/outside/repository/mvwd-env.sh
source /path/outside/repository/mvwd-env.sh
mvwd validate-config --config configs/smoke.yaml
pytest
```

Inspect the installed stack and available scenes without generating data:

```bash
mvwd inspect-runtime --config configs/smoke.yaml
```

Run the small headless simulator probe only after accepting the Omniverse EULA and selecting an output directory:

```bash
scripts/run_smoke.sh /path/to/output
```

The wrapper verifies the persisted result marker because Kit fast shutdown may terminate before the Python CLI can
propagate its intended exit code.

Generate or resume an accepted development dataset with an explicit output root:

```bash
mvwd generate --config configs/smoke.yaml --scene Rs_int --output-root /path/to/smoke-output
mvwd generate --config configs/integration.yaml --scene Rs_int --output-root /path/to/integration-output
# Next clean final-robot integration step after a fresh 1x1x1 regression:
mvwd generate --config configs/integration_final_robot.yaml --scene Rs_int \
  --output-root /path/to/new-final-robot-integration-output
```

Generation writes `generation_status.json` while running, appends structured rejects to `rejects.jsonl`, and
atomically finalizes accepted configurations and episodes. Dataset-v1.1 also writes the resolved configuration,
a configuration fingerprint, a non-empty public semantic/instance taxonomy, per-frame observation calibration, and
full temporal overlap graph metadata. Resume is allowed only when the fingerprint is identical; a changed config is
refused instead of being mixed into an existing root. Pilot/default profiles are refused unless `--allow-large` is
supplied. Pilot and production profiles cannot loop through scenes in one Isaac process; use the scene-sharded
launcher below.

## Final production preparation

The canonical configs share identical Dataset-v1.1 research semantics and differ only in scale:

- `configs/integration_final.yaml`: Beechwood_0_int, 5 configurations × 3 episodes.
- `configs/pilot_production.yaml`: healthy Beechwood_0_int plus constrained Rs_int, each 20 × 3.
- `configs/production_v1.yaml`: all eligible scenes, each 150 × 3. Run explicit scene batches
  into one final root; the global manifest and dataset fingerprint stay fixed.

Run the clean integration on one GPU. The parent launches a fresh worker process for the scene and writes only global
status; the worker writes only `shards/Beechwood_0_int`:

```bash
mvwd production-launch --config configs/integration_final.yaml --gpus 7 --max-workers 1 \
  --output-root /path/to/integration-final --cache-root /path/to/cache
mvwd finalize-dataset --dataset-root /path/to/integration-final
mvwd dataset-diagnostics --dataset-root /path/to/integration-final
```

Only after integration passes, run the real two-scene pilot using available GPUs:

```bash
mvwd production-launch --config configs/pilot_production.yaml --gpus 0,1 --max-workers 2 \
  --allow-large --output-root /path/to/pilot-production --cache-root /path/to/cache
mvwd finalize-dataset --dataset-root /path/to/pilot-production
mvwd pilot-report --dataset-root /path/to/pilot-production
```

Each scene has `logs/<scene>.log`, `shards/<scene>/shard_status.json`, its own `rejects.jsonl`, and `timing.json`.
Rerunning skips matching completed shards. Use `--retry-failed` to retry only failed shards; a fingerprint mismatch is
refused. `finalize-dataset` merges metadata/indexes and taxonomy deterministically without copying dense episode data.
The readiness report is written under `global/pilot_report.{json,md}`, with plots and per-scene summaries. It always
stops after reporting; it never starts full production.

The launcher runs one scene worker per listed GPU in parallel; if a batch contains more scenes
than GPUs, each GPU processes its own queue serially. Full production requires an explicit
`--scenes` batch so it cannot accidentally start all 50 eligible scenes. For example, after
checking free disk space, start two complete 150 × 3 scene shards directly in the final root:

```bash
mvwd production-launch --config configs/production_v1.yaml --scenes Beechwood_0_int,Rs_int \
  --gpus 0,1 --max-workers 2 --allow-large --output-root /path/to/final-dataset \
  --cache-root /path/to/cache
mvwd finalize-dataset --dataset-root /path/to/final-dataset --allow-partial
```

Later batches use the same config and output root with other `--scenes` IDs. A partial
index includes completed scene shards and atomically committed, QA-passing episodes from
interrupted shards. Finalization without `--allow-partial` still requires all selected
scenes to be complete. A production root has one coordinator lock, and each
scene shard has its own worker lock. Run the long command in `tmux` or the cluster's
batch scheduler. To resume the same code and configuration after interruption:

```bash
mvwd production-launch --config configs/pilot_production.yaml --gpus 0,1 --max-workers 2 \
  --allow-large --retry-failed --output-root /path/to/pilot-production --cache-root /path/to/cache
```

The parent restarts sampling exhaustion and stalled workers with new deterministic retry epochs. Completed episodes
are committed by atomic directory rename and skipped on resume. Status JSON is also replaced atomically. A code or
configuration change changes the dataset fingerprint and requires a new output root; keep old roots as separate
provenance snapshots. After disk-full or an external interruption, restore free space and rerun the same
config, source, output root and scene batch with `--retry-failed`:

```bash
mvwd production-launch --config configs/production_v1.yaml --scenes Beechwood_0_int,Rs_int \
  --gpus 0,1 --max-workers 2 --allow-large --retry-failed \
  --output-root /path/to/final-dataset --cache-root /path/to/cache
```

Only incomplete staging directories are removed on resume; committed episodes remain. Regenerate the partial
index after each interruption or completed batch. Never copy data from a different fingerprinted root into this
one. Check free space before each batch: the full 50-scene projection can exceed the current filesystem.
Multi-node production uses one coordinator per dataset root, with GPU IDs local to that node; concurrent launchers
targeting one root are rejected.
If the same configuration repeatedly exhausts episode sampling before producing any complete episode, the parent
archives it under `shards/<scene>/quarantine/configurations/` and samples a replacement for that configuration slot.
Configurations with even one complete episode are retained.

`configs/scene_eligibility.yaml` records all 51 swept scenes: 50 eligible and the explicitly excluded
`Wainscott_0_garden`, whose final robot footprint has no valid navigable state. Installed-scene reconciliation fails
loudly if the catalog changes. Primary splits are generated only from eligible scenes and remain scene-family disjoint.

Before rendering a dataset, inspect stochastic sampling distributions without a dense RGB rollout:

```bash
mvwd sampling-diagnostics --config configs/final_robot_preview.yaml --scene Rs_int \
  --samples 100 --output-root /path/to/diagnostics

# Optional: also render sparse GT-depth keyframes and aggregate overlap topology.
mvwd sampling-diagnostics --config configs/final_robot_preview.yaml --scene Rs_int \
  --samples 20 --with-overlap-preflight --output-root /path/to/overlap-diagnostics
```

Before any new 60-frame smoke, run the orientation-aware navigation-only
feasibility gates. The first two commands are micro/regression scenes; omit
`--scene` only after they pass to sweep every installed scene. These commands
do not render dense RGB rollouts:

```bash
mvwd navigation-sweep --config configs/final_robot_preview.yaml --scene Rs_int \
  --output-root /path/to/nav-rs-int
mvwd navigation-sweep --config configs/final_robot_preview.yaml --scene Beechwood_0_int \
  --output-root /path/to/nav-beechwood
mvwd navigation-sweep --config configs/final_robot_preview.yaml \
  --output-root /path/to/nav-all-scenes
```

The full sweep runs scene discovery and every scene in separate spawned
processes. This is required by the installed Isaac Sim / OmniGibson runtime:
cross-scene `og.clear()` leaves stale renderer and physics graph nodes. Results
are checkpointed after every scene under `navigation_sweep_records/`; rerunning
the same command and output root resumes completed scenes after verifying the
scene list and navigation configuration manifest.

Each sweep writes `navigation_sweep.json`, `navigation_sweep.csv`, and
`navigation_sweep.html`, including route-bank/compatible-triplet metrics,
point-vs-any-yaw free space, exact footprint/caster evidence, PhysX false-safe
counts, timings, and healthy/constrained/marginal/infeasible classification.

After generation, aggregate every finalized configuration/episode (including
room coverage, trajectory distributions, overlap topology, intervention
visibility/effect, stable IDs, calibration evidence, rejects, and storage):

```bash
mvwd dataset-diagnostics --dataset-root /path/to/generated-dataset
```

The command writes `dataset_diagnostics.json` in the dataset root. Sampling
collapse thresholds and their minimum evaluation sample count are resolved
from `sampling_diagnostics` in the dataset's own `resolved_config.yaml`;
small smoke runs report that warning evaluation is deferred.

The generator samples OG room-instance (or explicitly named conceptual fallback) observation regions, three
independent traversability-aware geodesic paths, and a configurable `dense_shared` / `partial_chain` / `exploratory`
regime. It does not optimize for the most compact or most parallel three-robot formation. Each fixed placement owns a
nested pool of trajectory sets. Temporal acceptance always requires a connected union overlap graph, at least one
meaningful shared moment, participation by every robot, and no near-duplicate observation system. `dense_shared`
requires at least 60% connected GT keyframes; `partial_chain` requires at least three of seven shared keyframes, two
participating keyframes per robot, and at most five consecutive isolated samples; `exploratory` permits one temporal
anchor per robot and at most six. A candidate exactly on a sparse isolation boundary is confirmed on 13 keyframes
using normalized isolation duration. These rules do not recreate a compact-formation gate. When several candidates pass the
GT-depth hard checks, the generator softly re-ranks them by route quality plus
the running global/split deficit of the realized regime. This is never a hard
regime gate. Before and after branches still use the exact same accepted
trajectory bytes.

The official robot remains Nova Carter based. Its project overlay now uses an
opaque closed lower-chassis fairing aligned to the full Nova wheel/caster
frame, an expanded mounting deck, integrated fixed tower shroud, overlapping
sliding sleeve, clean sensor housing, broad color identity surfaces, and a
front lens/heading marker. These additions are visual-only and preserve the
validated collision footprint and camera frame. Canonical appearance identity is
`robot_00` orange, `robot_01` blue, and `robot_02` green. New RGB and normal
outputs are exactly three channels (RGB and camera-space XYZ respectively).

Visualize every stored image modality for one episode in a single command:

    source /path/to/behavior_world/activate.sh
    python visualize_dataset.py /path/to/dataset/episodes/Rs_int/config_000/episode_000

The default output is episode_000/inspection/all_modalities/index.md, which can be opened with VS Code Markdown
Preview. It contains browser-compatible H.264 MP4 videos and contact-sheet PNGs
for every temporal world / robot RGB, depth, height, semantic, instance, occupancy, and normal array, plus PNGs for
the static environment BEVs. Pass the dataset root instead to process every episode recursively. Use
--output-kind frames for PNG frames only, or --output-kind both for videos and every PNG frame. World-BEV occupancy
is validated by default so the known stale perspective-camera capture failure is not silently visualized.

The full profile is never started automatically. See [the environment guide](docs/environment_setup.md),
[dataset v1 specification](docs/dataset_v1_spec.md), [pipeline gates](docs/pipeline.md).

## Repository boundaries

- Dataset records and training-facing code contain no OmniGibson objects.
- All machine paths are resolved centrally from CLI overrides and environment variables.
- Dense arrays are derived products; structured state, snapshots, trajectories, and events are canonical.
- Each floor has one static canonical extent shared by all environment/world BEVs; resolution may differ.
- BEV occupancy, point traversability, any-yaw robot navigability, and yaw
  freedom are distinct stored modalities; `traversability` is a deprecated
  alias of `any_yaw_navigable`.
- Renderer IDs are remapped through native paths and stable ObjectState IDs into documented public integers.
- Scene-family-disjoint splits are assigned before configuration generation.
