# Environment setup

The tested stack is BEHAVIOR-1K v3.9.2, OmniGibson 3.9.2, Isaac Sim 5.1.0, Python 3.11, and headless Linux with an
RTX GPU. Install the simulator outside this repository by following the matching BEHAVIOR-1K release instructions.
Do not install a second Isaac Sim into an already working environment and do not change core versions merely to silence
static Pillow or websockets dependency warnings.

## Portable machine configuration

Copy `env.example.sh` to an untracked location. `BEHAVIOR_ROOT` must point at the external BEHAVIOR-1K checkout;
`DATASET_DEV_OUTPUT` and `DATASET_CACHE_ROOT` must be writable. GPU selection is optional and should usually be
provided by the scheduler. The precedence used by this project is CLI override, environment variable, then a portable
default where one exists. Missing required external paths are errors.

The Omniverse license must be reviewed and accepted by the operator on first launch. A headless session may set
`OMNI_KIT_ACCEPT_EULA=yes` after that review. This repository never sets or accepts it automatically.

## Custom Conda environment roots

The official setup supports `CONDA_ENVS_PATH`, but one tested installation exposed a name-check edge case:
`--new-env` created the environment successfully, while the setup script later rejected it because Conda had activated
the environment by absolute path and the script compared an activation name. The safe continuation was:

1. Verify that the newly created environment contains Python and Isaac Sim.
2. Manually activate that exact created environment.
3. Rerun the official setup command **without** `--new-env`.

Do not create a second environment or rewrite package versions in response to that activation-name check.

## Verification

With the external environment active:

```bash
python --version
python -m pip show omnigibson isaacsim bddl
mvwd inspect-runtime --config configs/smoke.yaml
```

Simulator tests require an NVIDIA GPU, `OMNIGIBSON_HEADLESS=1`, and prior EULA acceptance. CPU tests do not import
OmniGibson and can run in a normal Python environment.

