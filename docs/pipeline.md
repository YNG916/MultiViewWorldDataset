# Generation pipeline and hard gates

The generator advances only after each gate's QA passes:

This list defines the intended sequence; it is not a claim that every gate has passed. The evidence-backed current
status is maintained in [runtime_findings.md](runtime_findings.md).

1. Inspect repository, installed APIs, scenes, assets, sensors, and snapshot support.
2. Validate portable configuration, schema, coordinate transforms, and geometry tests.
3. Start OmniGibson headlessly, load one discovered scene, catalog objects, and round-trip a snapshot.
4. For every floor, compute world bounds and render/calibrate a robot-free true-orthographic `B_env`.
5. Sample one relation-preserving movable-object configuration, settle, validate, deduplicate, restore, and render it.
6. Place three development robots and calibrated cameras with sampled physical heights.
7. Backproject depth, build the overlap graph, require connectivity, and reject near duplicates.
8. Produce one smooth shared `T_all`, validate it, store exact poses, and render `V0`.
9. Render mandatory `B_world_before[t]` and verify robot masks against projected robot poses.
10. Restore `W0`, apply one valid rigid event, revalidate the unchanged `T_all`, and render `B_env_after`, `V1`, and
    `B_world_after`; run paired QA.
11. Resolve the installed Nova Carter asset and build a project-derived robot layer with a physical mast; rerun smoke.
12. Add articulation and meaningful state events.
13. Persist through temporary directories, atomically finalize only accepted samples, resume safely, and log rejects.
14. Run 1×1×1 smoke and save a headless inspection report.
15. Run 1 scene × 5 configurations × 3 episodes integration, then stop.

The full profile is planning metadata only. A rejected intervention never changes or resamples `T_all`. Configuration
acceptance is based on accepted samples, not attempt count.

