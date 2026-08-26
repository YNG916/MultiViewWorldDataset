# mobile_sensor_robot_v1 templates

These files are portable templates, not a vendored NVIDIA asset. At runtime the adapter resolves Isaac's asset root,
checks the Nova Carter URI, replaces the placeholder into an ignored generated USDA, and writes the matching
RobotDefinition YAML. The project layer adds visible mast/head geometry and the calibrated camera; it never edits the
source asset.

The four mast positions still require runtime articulation QA against the resolved Nova Carter prim hierarchy before
this robot may be marked production-ready.

