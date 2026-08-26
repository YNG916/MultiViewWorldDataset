# Architecture

`schema` owns simulator-independent records. `adapters.BaseSimulatorAdapter` is the only interface consumed by the
pipeline; `OmniGibsonAdapter` contains lazy imports and native-object translation. A future Isaac-based adapter can
implement the same boundary without changing stored records.

`utils.runtime` centrally resolves CLI/environment paths and versions. `utils.config` loads tracked YAML and applies
profile inheritance. `cameras` owns calibration, transforms, and depth overlap. `sampling` owns deterministic IDs,
configuration hashes, placement, smooth trajectories, interventions, and split assignment. `rendering` owns true
orthographic calibration and inspection products. `storage` writes canonical records and derived arrays atomically.
`qa` returns structured pass/fail results and reject reasons.

The orchestration layer follows Base Scene → Dynamic Configuration → Episode and never merges configuration storage
into episodes. Simulator objects are transient and must not appear in JSON, Parquet, NPZ, or Zarr metadata.

