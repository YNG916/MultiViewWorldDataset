# mobile_sensor_robot_v1 templates

These are portable project templates, not a vendored NVIDIA asset. At runtime
the adapter resolves the installed Isaac asset root, verifies Nova Carter,
materializes the ignored USDA/RobotDefinition files, and validates every
project-owned mast, camera, material, and appearance prim. NVIDIA and BEHAVIOR
source assets are never edited.

The official overlay keeps the Nova Carter chassis and functional collision
geometry. It adds a render-only integrated base plate, fixed outer tower,
overlapping sliding sleeve, collars, compact sensor head, front lens, and
heading marker. A broad color-matched sensor-head top cap keeps canonical robot
identity legible in world BEV. The long sleeve overlaps the fixed tower at all
four camera
heights, so no telescoping section appears disconnected. Decorative shell
geometry has no `PhysicsCollisionAPI`; hidden functional mast/head colliders
and the original footprint remain unchanged.

Canonical accent materials are bound per robot at load time:

- `robot_00` → `accent_orange`
- `robot_01` → `accent_blue`
- `robot_02` → `accent_green`

All variants use identical geometry, joint limits, mass, camera prim, and
collision semantics. Allowed camera heights are 0.8, 1.0, 1.2, and 1.4 m, with
`mast_joint_value_m = camera_height_m - 0.8`.
