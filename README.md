# MultiViewWorldDataset

`MultiViewWorldDataset` is a portable research dataset generator for synchronized, calibrated views from three mobile
robots in BEHAVIOR-1K / OmniGibson scenes. It generates counterfactual before/after rollouts using exactly the same
physical trajectories.

The model input contains **no robot ego-view image**. Dataset v1 is defined around a robot-free environment BEV
(`B_env`), calibrated target camera poses, and—at Level 2—one structured intervention. Post-intervention BEVs and
world BEVs are GT/oracle data, not normal model inputs.

## Quick start

Create and activate a compatible external simulator environment, then install this repository without replacing its
simulator packages:

```bash
python -m pip install -e .
cp env.example.sh /path/outside/repository/mvwd-env.sh
source /path/outside/repository/mvwd-env.sh
mvwd validate-config --config configs/smoke.yaml
pytest
```

Inspect the installed stack and available scenes without generating data:

```bash
mvwd inspect-runtime --config configs/smoke.yaml
```

Run the small headless simulator probe only after accepting the Omniverse EULA and selecting an output directory:

```bash
scripts/run_smoke.sh /path/to/output
```

The wrapper verifies the persisted result marker because Kit fast shutdown may terminate before the Python CLI can
propagate its intended exit code.

Generate or resume an accepted development dataset with an explicit output root:

```bash
mvwd generate --config configs/smoke.yaml --scene Rs_int --output-root /path/to/smoke-output
mvwd generate --config configs/integration.yaml --scene Rs_int --output-root /path/to/integration-output
```

Generation writes `generation_status.json` while running, appends structured rejects to `rejects.jsonl`, and
atomically finalizes accepted configurations and episodes. Re-running the same command resumes finalized work.
Pilot/default profiles are refused unless `--allow-large` is supplied; this workflow stops after integration.

The full profile is never started automatically. See [the environment guide](docs/environment_setup.md),
[dataset v1 specification](docs/dataset_v1_spec.md), [pipeline gates](docs/pipeline.md), and the

## Repository boundaries

- Dataset records and training-facing code contain no OmniGibson objects.
- All machine paths are resolved centrally from CLI overrides and environment variables.
- Dense arrays are derived products; structured state, snapshots, trajectories, and events are canonical.
- Each floor has its own calibrated true-orthographic BEV.
- Scene-family-disjoint splits are assigned before configuration generation.

