# Robots

Early gates use the installed data-driven Turtlebot to avoid blocking the pipeline. It renders its body and uses its
non-holonomic base for placement and collision checks. Its built-in camera is not treated as the frozen dataset camera.

The final `mobile_sensor_robot_v1` uses the installed NVIDIA Nova Carter base, a project-derived USD layer, a vertical
telescopic mast, a simple head, and a dedicated calibrated perspective VisionSensor. It has no manipulator. All three
robots share morphology; optional small visual identity accents may differ.

The base asset is resolved at runtime through the Isaac asset root API. No absolute Isaac asset path is tracked and no
NVIDIA source asset is edited. The template under `assets/robots/mobile_sensor_robot_v1` is materialized into an ignored
runtime directory after the resolver confirms that the standard Nova Carter identifier exists.

The mast has four physical positions corresponding to camera heights 0.8, 1.0, 1.2, and 1.4 m. An episode samples one
height per robot and keeps its mast joint fixed for all frames. Pitch is −5°, roll 0°, and changing height cannot be
implemented by moving an invisible camera independently of the rendered mast.

The final robot is the production morphology. Its size is frozen during the
current feasibility study: code must first distinguish real geometric
infeasibility from the former XY-only planner's false rejections. The collision
footprint is the convex union of every enabled collision link, including caster
links, at the authored configuration, expanded only by the configured 0.03 m
safety margin. Metadata records source links, raw/expanded polygons, caster
treatment, width, length, area, radius, and the old reset-AABB dimensions.

Navigation uses orientation masks and an SE(2) lattice. A pose is valid exactly
when its `(x,y,yaw_bin)` footprint is safe; the fraction of safe yaws is only a
ranking diagnostic. PhysX calibration probes open, wall, furniture, corridor,
door, and corner categories and reports raster false-safe and conservative
predictions. Robot dimensions may be reconsidered only after the full
navigation-only scene sweep, never as an automatic rejection workaround.
