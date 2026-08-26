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

This document is the final-robot design contract. The current project-owned USDA template adds visible mast and
camera geometry over Nova Carter, but it does not yet contain a runtime-verified prismatic mast joint. Consequently
`mobile_sensor_robot_v1` must not be selected for accepted generation until gate-11 articulation and collision QA
passes; development probes use Turtlebot.
