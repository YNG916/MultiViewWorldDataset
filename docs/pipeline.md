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
8. Run low-resolution temporal GT-depth overlap preflight using union connectivity, participation, isolation, and
   near-duplicate hard checks; use per-frame connectivity only as a soft regime target, persist requested/realized
   regimes, store exact poses, and render `V0`.
9. Render mandatory `B_world_before[t]` and verify robot masks against projected robot poses.
10. Select a quota-controlled fixed intervention type and a V0-visible target; restore `W0` between attempts, render
    `B_env_after`, `V1`, and `B_world_after`, require post-render evidence, and run exact paired-trajectory QA.
11. Resolve the installed Nova Carter asset and build a project-derived robot layer with a physical mast; rerun smoke.
12. Add articulation and meaningful state events.
13. Persist through temporary directories, atomically finalize only accepted samples, resume safely, and log rejects.
14. Run CPU tests, SE(2) microtests, then `navigation-sweep` on `Rs_int`,
   Beechwood, and all discovered scenes without a dense RGB rollout. Classify
   scenes as healthy/constrained/marginal/infeasible and stop to review the
   evidence before any robot-geometry change. The installed simulator requires
   one spawned process per scene; the sweep writes a manifest plus a per-scene
   checkpoint so interrupted runs resume without cross-scene renderer/PhysX
   state contamination.
15. Only if navigation is healthy, run a clean final-robot 1×1×1 smoke and save
   expanded trajectory/overlap inspection; then run 1×5×3 integration and stop.

The full profile is planning metadata only. A rejected intervention never changes or resamples `T_all`. Configuration
acceptance is based on accepted samples, not attempt count.
