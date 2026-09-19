# Installed API findings and gate status

This report records the stack inspected through 2026-09-18. It is evidence for this checkout, not a promise that every
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

The final `mobile_sensor_robot_v1` completed a prior 1×1×1, 60-frame Beechwood
regression (`final_robot_routefirst_graph_v22_labelcache_gpu4_20260917`) with
all QA passing and byte-identical before/after trajectories. That run predates
the present SE(2) redesign and is a regression reference, not validation of the
new planner.

The 2026-09-18 redesign fixes NumPy-2 tensor detection, removes yaw-freedom,
route-bank-size, and waypoint-family hard gates, adds project-side SE(2)
forward/stop-turn planning, complete relation refresh, explicit BEV navigation
layers, six-category PhysX footprint calibration, finite-score proxy fallback,
and a navigation-only all-scene sweep command. CPU tests and microtests must
pass first. The required next simulator order is: `Rs_int` navigation sweep,
Beechwood regression sweep, full discovered-scene navigation sweep, evidence
review, then (only if healthy) a clean 1×1×1 smoke. No pilot or full generation
is authorized by these changes, and robot geometry has not been changed.

OmniGibson 3.9.2 / Isaac Sim 5 does not reliably support loading a different
scene after `og.clear()` in the same process: stale SyntheticData nodes can
raise `Invalid NodeObj`, and PhysX/USD prim cleanup can fail. Full navigation
sweeps therefore isolate discovery and every scene in separate spawned
processes and persist each result before `SimulationApp.close()`.
