#!/usr/bin/env bash

# Copy this file outside the repository, fill in machine-specific values, and source it.
export BEHAVIOR_ROOT=/path/to/BEHAVIOR-1K
export DATASET_DEV_OUTPUT=/path/to/output
export DATASET_CACHE_ROOT=/path/to/cache

# Optional. Leave unset to let the scheduler / OmniGibson select visible devices.
# export OMNIGIBSON_GPU_ID=<gpu-index>
export OMNIGIBSON_HEADLESS=1

# Accept the Omniverse EULA only after reviewing it. Do not set this implicitly.
# export OMNI_KIT_ACCEPT_EULA=yes

