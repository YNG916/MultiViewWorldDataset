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
6. Select a room/region-balanced observation regime and place three separated calibrated robots without compactness or
   common-focus optimization.
7. Generate 8–16 nested independent geodesic trajectory sets; enforce kinematics, collision safety, separation, and
   formation-collapse controls.
8. Run low-resolution temporal GT-depth overlap preflight using union connectivity, participation, isolation, and
   near-duplicate hard checks; use per-frame connectivity only as a soft regime target, persist requested/realized
   regimes, store exact poses, and render `V0`.
9. Render mandatory `B_world_before[t]` and verify robot masks against projected robot poses.
10. Select a quota-controlled fixed intervention type and a V0-visible target; restore `W0` between attempts, render
    `B_env_after`, `V1`, and `B_world_after`, require post-render evidence, and run exact paired-trajectory QA.
11. Resolve the installed Nova Carter asset and build a project-derived robot layer with a physical mast; rerun smoke.
12. Add articulation and meaningful state events.
13. Persist through temporary directories, atomically finalize only accepted samples, resume safely, and log rejects.
14. Run 1×1×1 smoke and save a headless inspection report.
15. Run 1 scene × 5 configurations × 3 episodes integration, then stop.

The full profile is planning metadata only. A rejected intervention never changes or resamples `T_all`. Configuration
acceptance is based on accepted samples, not attempt count.

