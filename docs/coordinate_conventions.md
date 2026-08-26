# Coordinate and camera conventions

Dataset world coordinates are right-handed, measured in meters, with +Z up. `T_A_B` is a 4×4 homogeneous transform
that maps a point expressed in frame B into frame A. Composition is therefore `T_A_C = T_A_B @ T_B_C`, and
`T_B_A = inverse(T_A_B)`.

OmniGibson / Isaac / USD world coordinates are converted explicitly at the adapter boundary. The currently tested
stack also uses right-handed +Z-up meters for BEHAVIOR scenes, so the default native-to-dataset transform is identity;
the identity is recorded and validated, not assumed throughout the code.

OpenCV camera coordinates use +X right, +Y down, +Z forward. Pixel centers follow `(u,v)` with `u` increasing right
and `v` down. Pixel intrinsics are:

```text
[fx  0 cx]
[ 0 fy cy]
[ 0  0  1]
```

Normalized intrinsics divide the first row by image width and the second row by image height. Linear depth is distance
along OpenCV +Z. Camera records store both `T_world_camera` (`camera_to_world`) and its inverse, plus
`T_world_robot_base` and `T_robot_base_camera`.

BEV row 0 represents maximum world Y and column 0 minimum world X. For bounds `(xmin,ymin,xmax,ymax)` and resolution
`m`, pixel center `(u,v)` maps to `(xmin+(u+0.5)m, ymax-(v+0.5)m, floor_z)`. Bounds are expanded to an integer number
of pixels; no scene is resized to a fixed pixel shape. Each floor has independent calibration.

Generation must abort if inverse/composition, camera round-trip, depth backprojection, or BEV round-trip checks fail.

