# Installed API findings and gate status

This report records the stack inspected through 2026-09-20. It is evidence for this checkout, not a promise that every
OmniGibson release exposes identical APIs.

## Installed stack and APIs

- Python 3.11.16, OmniGibson 3.9.2, Isaac Sim 5.1.0.0, BDDL 3.7.0.
- Scene discovery uses `omnigibson.utils.asset_utils.get_available_behavior_1k_scenes()`; 51 scenes were found.
- Canonical simulator snapshots use `og.sim.dump_state/load_state(serialized=True)`.
- Robot observations use `omnigibson.sensors.vision_sensor.VisionSensor`.
- True orthographic projection is set on the USD camera with
  `UsdGeom.Camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)`.
- The Isaac asset-root API resolved Nova Carter to
  `.../Assets/Isaac/5.1/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd`. The project references that installed
  asset; it never edits NVIDIA or BEHAVIOR source assets.

Two installed-version behaviors are isolated inside the adapter. First,
`TraversableMap.get_random_point(floor=None)` calls `torch.randint` without a size under this Torch version, so the
adapter samples an explicit floor. Second, reference-point sampling is uniform over an entire connected component,
so clustered placement uses the robot-eroded traversability map and a seeded local candidate search. OmniGibson
cleanup can also race an asynchronous temporary USD writer; the adapter closes Kit even if that cleanup raises an
`OSError`.

## Verified smoke evidence

A headless `Rs_int` probe passed on the installed GPU stack:

- 83 catalog objects and stable instance IDs after restore.
- 1,318 serialized snapshot values; maximum restore error `2.384185791015625e-07`.
- Robot-free orthographic BEV at 0.02 m/px and robot-containing orthographic BEV at 0.04 m/px.
- Three 896x512 RGB, linear-depth, normal, semantic, and instance observations.
- Pairwise depth-reprojection overlaps `0.4064`, `0.5930`, and `0.5580`: connected and below the near-duplicate
  threshold.
- Total cold-run time about 187 seconds.

The output location is operator-selected and intentionally untracked. The probe writes `summary.json`, an HTML
inspection page, five PNGs, and `smoke_probe_last_result.json`.

## Gate status

Gates 0-4, 6, 7, and the gate-14 simulator probe have executable evidence. Simulator-independent schema,
configuration hashing, trajectories, rigid/articulation/state event proposals, paired QA, atomic episode writing, and
resume guards exist for gates 5, 8, 10, 12, and 13.

The 2026-09-18 redesign fixes NumPy-2 tensor detection, removes yaw-freedom,
route-bank-size, and waypoint-family hard gates, adds project-side SE(2)
forward/stop-turn planning, complete relation refresh, explicit BEV navigation
layers, six-category PhysX footprint calibration, finite-score proxy fallback,
and a navigation-only all-scene sweep command.

The isolated all-scene navigation sweep completed all 51 discovered scenes with
38 healthy, 12 constrained, one infeasible, and zero runtime classifications.
This establishes installed-stack coverage without dense RGB rollout cost.

The final Nova-Carter-based `mobile_sensor_robot_v1` completed a clean 1×1×1,
60-frame Beechwood smoke at
`final_robot_dataset_v11_visual_softregime_smoke_beechwood_gpu5_20260920`.
All eight episode QA checks passed. Before/after trajectory matrix and position
errors were exactly zero; all RGB and normal outputs were exactly three channel;
the temporal graph connected at 2/7 keyframes with all three union edges and a
maximum isolated run of five. One rigid relocation changed exactly one object
and produced 57,178 changed target pixels at mean RGB delta 81.72.

The official shrouded appearance was separately validated on all three robots
and all four camera heights by
`final_robot_asset_v8_topcap_gpu0_20260920`. Orange, blue, and green materials
bind to five visual-only prims per robot. The later BEV identity top-cap
refinement is 0.19×0.14×0.012 m and has no collision API; the focused runtime
validator again reported unchanged collision geometry and camera-frame errors
below 1e-6-scale numerical noise.

The final CPU suite passes 115 tests. No pilot or full generation was run, and
full generation has not been started.

OmniGibson 3.9.2 / Isaac Sim 5 does not reliably support loading a different
scene after `og.clear()` in the same process: stale SyntheticData nodes can
raise `Invalid NodeObj`, and PhysX/USD prim cleanup can fail. Full navigation
sweeps therefore isolate discovery and every scene in separate spawned
processes and persist each result before `SimulationApp.close()`.
