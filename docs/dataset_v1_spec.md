# Dataset v1.1 specification

This file freezes dataset semantics. Machine paths and GPU selection are not dataset semantics and stay outside YAML.

## Task and notation

BEV means Bird's-Eye View, GT means Ground Truth, and 3DGS means 3D Gaussian Splatting. `N` is the robot count
(`N=3`), `T` is the number of physical frames (`T=60` in the full profile), `t` is physical/video time, and `tau` is
reserved for diffusion-model time.

No robot ego image is ever a model input. Level 1 maps robot-free `B_env` plus calibrated target camera poses to
synchronized target views. Level 2 additionally supplies one structured intervention but still uses the original
robot-free `B_env`; post-intervention BEV is GT/oracle data. Level 3 supplies continuous target trajectories. `V0` is
not required model input.

## Hierarchy and scale

The hierarchy is Base Scene → Dynamic Configuration → Episode → paired Before/After rollout. A configuration is one
accepted randomized physical arrangement; an episode contains three identities, fixed sampled mast heights, initial
placements, a nested pool of independent three-robot trajectory sets, one accepted intervention, and the paired
rollouts. Every dynamically discovered scene
targets 150 accepted configurations and every configuration targets three accepted episodes. Attempt counts do not
count as accepted samples. Full generation is never automatic.

The primary split is scene/scene-family disjoint and is decided before configuration generation. No primary split may
occur at frame, episode, or configuration level.

## Cameras and BEVs

The pinhole robot camera has no distortion: RGB 896×512, geometry 448×256, HFOV 70°, pitch −5°, roll 0°, near 0.1 m,
far 15 m. RGB and geometry intrinsics are stored separately at their native resolutions. Every modality records the
actual producing sensor pose; mounted and shared-capture poses plus their alignment errors are both retained. Per-robot height is sampled once from 0.8, 1.0, 1.2, 1.4 m and physically changes the
mast.

`B_env` is a robot-free whole-floor **true orthographic** render. Each floor is separate, at 0.02 m/px with RGB,
linear depth, height above floor, normals, semantic, instance, occupancy, and a separate robot-eroded traversability
mask. Occupancy is observed geometry and is never used as a synonym for navigability. Only RGB is the default
input. A configuration stores `B_env_before` once and its episodes reference it.

`B_world_before[t]` and `B_world_after[t]` are mandatory GT at 0.04 m/px and include all three rendered robot bodies.
They use identical bounds/calibration and contain RGB, linear depth, height, semantic, instance, and occupancy; normals
are optional. Dense renders are derived, never canonical state.

## Placement, overlap, and motion

Initial bases are sampled from balanced OG room-instance regions (or explicitly named conceptual fallback regions),
with at least about 0.6 m pairwise separation and no environment, object, robot, or camera-geometry collision. There
is no hard compact cluster, common-focus heading, or most-parallel joint objective. Configurable regimes are
`dense_shared`, `partial_chain`, and `exploratory`.

View overlap comes from GT-depth backprojection. Sparse keyframe graphs are accepted through union-graph
connectivity, at least one meaningful shared moment, participation by every robot, bounded isolation runs, rejection
of near-duplicate views, and no requirement that any individual keyframe graph be connected. Connected-frame and
shared-keyframe fractions are soft targets that distinguish requested from realized observation regimes. Neither all
pairs nor all frames must overlap. Full overlap matrices and graph topology are persisted.

At 10 FPS, 60 frames span 6 seconds with smooth acceleration/cruise/deceleration, collision-free independent paths of
1–3 m geodesic/smoothed arc length. Height, pitch, roll, and
intrinsics stay fixed. Exact robot-base and camera poses are stored at every `t`.

## Counterfactual intervention

Each v1.1 episode has exactly one atomic event: accepted-sample quotas target 60% rigid relocation, 30% articulation,
and 10% meaningful visible state change. The event type is fixed before resampling and never silently falls back.
Rigid relocation targets about 0.3–1.5 m and 30–120°, normally on the same floor and in the same room while preserving
support/containment. Add/remove and special mid-video sequences are excluded.

The configuration is restored; `W0`, `V0`, and `B_world_before` are produced on trajectory `T_all`; the same base state
is restored, one event creates `W1`, and **that same** `T_all` is validated and rendered after. If invalid, resample the
event, never the trajectory. The target must be visible in stable public V0 instance masks; after V1, mask/RGB evidence
must prove a visible effect or W0 is restored and only the same event type is resampled. Before/after robot and camera poses must be numerically equal. The event uses
`application_mode=pre_rollout,time_index=null`; the schema permits a future `timed` integer event.

## Canonical record

Canonical truth is structured world state + simulator snapshots + trajectory + event log. Store scene/config/episode
IDs, a renderer→native path→ObjectState ID→public integer mapping, non-empty semantic/category taxonomy, stable object IDs, versions, seeds, generator commit, `W0/W1`, robot timelines, changed-object tracks, and snapshot
references. Do not repeat unchanged full object state per frame. Catalogs use Parquet where available, trajectories use
NPZ, dense arrays use Zarr, and small metadata uses JSON/YAML.

