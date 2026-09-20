# Robots

The official Dataset-v1.1 robot is `mobile_sensor_robot_v1`. It keeps the
installed NVIDIA Nova Carter chassis, collision links, wheel/caster geometry,
and non-holonomic base. The project never shrinks the base or edits the NVIDIA
asset. Turtlebot support remains only for early compatibility probes and is not
the production morphology.

The project-local overlay under
`assets/robots/mobile_sensor_robot_v1` adds a calibrated physical mast camera
and a lightweight product-style visual superstructure:

- a mounting plate and lower/upper collars integrate the tower with the chassis;
- a fixed graphite outer tower encloses the lower mechanism;
- a long sliding inner sleeve remains overlapped with the outer tower at every
  allowed height, eliminating the former visible floating gap;
- a compact sensor housing, front lens, and front marker make heading obvious;
- a broad color-matched sensor-head top cap makes identity legible from above;
- broad tower/head accent surfaces remain visible in ego images and world BEV.

The new shell and color parts are render-only and intentionally have no
`PhysicsCollisionAPI`. The previously validated Nova Carter footprint and the
hidden functional mast/head colliders remain authoritative. Therefore the
redesign changes appearance, not footprint, SE(2) masks, contact semantics, or
navigation behavior.

The three canonical identities share exactly the same geometry and differ only
by stable material binding:

- `robot_00`: warm orange
- `robot_01`: cool blue
- `robot_02`: signal green

This mapping is centralized in `assets.py`, checked at runtime against all
required prim/material paths, and persisted in dataset and inspection metadata.

The mast has four physical positions corresponding to camera heights 0.8, 1.0,
1.2, and 1.4 m. An episode samples one height per robot and keeps it fixed for
all 60 frames. `mast_joint_value_m = camera_height_m - 0.8`; both values are
stored in robot/camera state. The camera optical frame remains a perspective
pinhole camera with HFOV 70 degrees, pitch -5 degrees, roll 0 degrees, near
0.1 m, and far 15 m. The camera prim, mount translation, and orientation were
not changed by the appearance redesign.

The base asset is resolved at runtime through the Isaac asset-root API. The
portable template is materialized into an ignored runtime directory only after
the standard Nova Carter identifier is verified. No absolute Isaac path is
tracked.

Production navigation is route-first on orientation-dependent footprint masks.
The SE(2) lattice supports forward motion and swept stationary stop-and-turn
actions. The collision footprint is the convex union of enabled collision links,
including casters, expanded only by the configured 0.03 m safety margin.
Footprint evidence, polygons, dimensions, source links, and PhysX probes are
persisted for QA.
