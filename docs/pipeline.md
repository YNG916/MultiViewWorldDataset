# Generation pipeline and hard gates

The generator advances only after each gate's QA passes:

This list defines the intended sequence; it is not a claim that every gate has passed. The evidence-backed current
status is maintained in [runtime_findings.md](runtime_findings.md).

1. Inspect repository, installed APIs, scenes, assets, sensors, and snapshot support.
2. Validate portable configuration, schema, coordinate transforms, and geometry tests.
3. Start OmniGibson headlessly, load one discovered scene, catalog objects, and round-trip a snapshot.
4. For every floor, freeze one static canonical extent and render/calibrate robot-free true-orthographic `B_env`,
   storing occupancy and robot-eroded traversability separately.
5. Sample a bounded, stratified multi-object relation-preserving configuration, settle, validate, deduplicate, restore,
   and render it.
6. Build a route bank on exact orientation-dependent robot footprint masks.
   Plan forward and swept stop-and-turn actions in `(x,y,yaw_bin)`; yaw freedom
   is a soft rank only. Floor feasibility requires a valid three-route
   combination, not a universal route-count threshold.
7. Select three independent route-first paths; time-parameterize straight runs
   and stationary turns to exactly 60 physical frames. Enforce moving-frame
   tangent alignment, stationary-turn angular speed, linear speed,
   acceleration, lateral slip, collision safety, and separation. Path-family
   mix is recorded rather than hard-gated.
8. Run a seven-keyframe low-resolution temporal GT-depth overlap preflight. Keep union connectivity, participation,
   a meaningful shared moment, near-duplicate rejection, and the regime-specific dense/partial/exploratory rules hard.
   Confirm isolation-only or exact-boundary cases on 13 keyframes using normalized temporal duration; do not spend the
   dense confirmation on a disconnected union or a robot that never participates. Validate exact candidates adaptively in cumulative batches 12 → 24 → 48 without
   revalidating ranks; stop expansion after the first batch containing any hard-valid candidate. Then softly
   rank by route quality plus the running global/split deficit of its realized
   regime; this never becomes an acceptance gate. Persist requested/realized
   regimes, selection diagnostics, exact poses, and render `V0`.
9. Render mandatory `B_world_before[t]` and verify robot masks against projected robot poses.
10. Select a quota-controlled fixed intervention type and a V0-visible target; restore `W0` between attempts, render
    `B_env_after`, `V1`, and `B_world_after`, require post-render evidence, and run exact paired-trajectory QA.
11. Resolve the installed Nova Carter asset and build the official
   project-derived layer: unchanged functional mast/camera/colliders plus the
   visual-only shrouded tower, sliding sleeve, sensor housing, and three
   canonical color variants. Validate prims, materials, height/joint relation,
   camera frame, and unchanged footprint before smoke.
12. Add articulation and meaningful state events.
13. Persist through temporary directories, atomically finalize only accepted samples, resume safely, and log rejects.
14. Run CPU tests, SE(2) microtests, then `navigation-sweep` on `Rs_int`,
   Beechwood, and all discovered scenes without a dense RGB rollout. Classify
   scenes as healthy/constrained/marginal/infeasible and stop to review the
   evidence before any robot-geometry change. The installed simulator requires
   one spawned process per scene; the sweep writes a manifest plus a per-scene
   checkpoint so interrupted runs resume without cross-scene renderer/PhysX
   state contamination.
15. Run the clean final-robot 1×5×3 integration through one isolated scene shard using production semantics.
16. Only if integration passes, run the healthy + constrained two-scene 120-episode pilot through separate workers,
    finalize metadata, generate complete pilot diagnostics and an explicit READY/NOT READY report, then stop.
17. Full production remains prepared but is never started automatically.

The full profile is planning metadata only. A rejected intervention never changes or resamples `T_all`. Configuration
acceptance is based on accepted samples, not attempt count.
