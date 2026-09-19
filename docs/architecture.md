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
compatibility utilities for legacy probes; the dataset generator does not call
them. Production calls `prepare_navigation_context` followed by
`sample_route_first_trajectory_sets`.

The orchestration layer follows Base Scene → Dynamic Configuration → Episode and never merges configuration storage
into episodes. Simulator objects are transient and must not appear in JSON, Parquet, NPZ, or Zarr metadata.
