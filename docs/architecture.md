# Architecture

`schema` owns simulator-independent records. `adapters.BaseSimulatorAdapter` is the only interface consumed by the
pipeline; `OmniGibsonAdapter` contains lazy imports and native-object translation. A future Isaac-based adapter can
implement the same boundary without changing stored records.

`utils.runtime` centrally resolves CLI/environment paths and versions. `utils.config` loads tracked YAML and applies
profile inheritance. `cameras` owns calibration, transforms, and depth overlap. `sampling` owns deterministic IDs,
configuration hashes, placement, smooth trajectories, interventions, and split assignment. `rendering` owns true
orthographic calibration and inspection products. `storage` writes canonical records and derived arrays atomically.
`qa` returns structured pass/fail results and reject reasons.

Production navigation is route-first. `sampling.se2` is a
simulator-independent orientation lattice with forward and swept stationary
rotation primitives. `adapters.navigation` calibrates map axes, builds
footprint masks and region topology, constructs the route bank, performs
bounded three-route selection, and runs sparse PhysX checks. The installed
OmniGibson XY shortest-path API may provide proposals or heuristics, but final
rectangular-robot validity comes from SE(2) masks and PhysX.

The older development placement and XY tangent-only trajectory methods remain
deprecated compatibility utilities for legacy probes and emit
`DeprecationWarning`; the dataset generator does not call them. The sole
production path calls `prepare_navigation_context` followed by
`sample_route_first_trajectory_sets`, then performs GT-depth validation and
soft realized-regime ranking.

The orchestration layer follows Base Scene → Dynamic Configuration → Episode and never merges configuration storage
into episodes. Simulator objects are transient and must not appear in JSON, Parquet, NPZ, or Zarr metadata.


## Production process and shard architecture

Production never changes scenes inside a long-lived Isaac Sim process. The parent launcher deterministically assigns
eligible scenes to GPUs and starts one fresh `scene-worker` process per scene. A worker owns exactly one
`shards/<scene_id>` directory; only the parent mutates `production_status.json`. Consequently taxonomy, reject logs,
resume markers, and counters are never concurrently appended by independent workers.

`configs/scene_eligibility.yaml` is reconciled against the full installed scene catalog before generation. Splits are
computed after eligibility filtering and checked for scene-family leakage. Each shard binds the resolved config,
generator source, Git commit, robot asset, schema, and fingerprint. Atomic configuration/episode staging directories
are recovered only after that fingerprint matches.

`finalize-dataset` verifies every selected shard, deterministically merges taxonomy and small metadata/indexes, and
leaves dense observations in place. `pilot-report` aggregates both global and per-scene acceptance, adaptive GT
shortlist, overlap, motion, intervention, DynamicConfiguration, runtime, and storage metrics; it produces an explicit
READY/NOT READY decision and never schedules the full run.
