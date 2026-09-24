# New-cluster Dataset-v1.1 production (2026-09-24)

The legacy root `/home/usqki/behavior_world/output/mvwd/dataset_v11_production_full_20260923`
is continuing with its own frozen producer. Its configuration fingerprint is
`9af704ea5018d868b75a4f2263cb08cafbaf5484c21ed79d777d6812515fb142`.
Do not merge its shards into a new-version root.

All new-version production roots use the frozen source under
`dataset_v11_production_newcluster_20260924/global/frozen_producer`, canonical
`configs/production_v1.yaml`, and configuration fingerprint
`0df9be94730c33ba14de0ed070e86a8e8b35e75382b72e0b9eee745e3d62ffbb`.
They cover the 50 eligible scenes exactly once:

- `dataset_v11_production_newcluster_20260924`: Beechwood_1_int, Ihlen_0_int, Merom_0_int (Ada, node-l-02).
- `dataset_v11_production_ada_l01_20260924`: Ihlen_1_int, Pomaria_2_int, house_single_floor, office_bike, grocery_store_asian (Ada, node-l-01).
- `dataset_v11_production_ada_l02_20260924`: Wainscott_1_int, hall_arch_wood, hotel_suite_large, restaurant_asian (Ada, node-l-02).
- `dataset_v11_production_4090_s01_20260924`: Beechwood_0_garden, gates_bedroom, office_cubicles_left.
- `dataset_v11_production_4090_s02_20260924`: Benevolence_0_int, grocery_store_cafe, hall_conference_large.
- `dataset_v11_production_4090_s03_20260924`: Benevolence_1_int, grocery_store_convenience, hall_glass_ceiling.
- `dataset_v11_production_4090_s04_20260924`: Benevolence_2_int, grocery_store_half_stocked, hall_train_station.
- `dataset_v11_production_4090_s05_20260924`: Merom_0_garden, hotel_gym_spa, house_double_floor_lower.
- `dataset_v11_production_4090_s06_20260924`: Merom_1_int, hotel_suite_small, house_double_floor_upper.
- `dataset_v11_production_4090_s07_20260924`: Pomaria_0_garden, office_cubicles_right, restaurant_brunch.
- `dataset_v11_production_4090_s08_20260924`: Pomaria_0_int, office_large, restaurant_cafeteria.
- `dataset_v11_production_4090_s09_20260924`: Pomaria_1_int, office_vendor_machine, restaurant_diner.
- `dataset_v11_production_4090_s10_20260924`: Rs_garden, restaurant_hotel, school_biology.
- `dataset_v11_production_4090_s11_20260924`: Wainscott_0_int, restaurant_urban, school_chemistry.
- `dataset_v11_production_4090_s12_20260924`: school_computer_lab_and_infirmary, school_geography, school_gym.
- `dataset_v11_production_4090_s13_20260924`: Beechwood_0_int, Rs_int.

The small-node batches depend on the successful 4090 1×1×1 smoke job 50074,
which finished with Slurm exit code 0 and eight passing episode QA checks.
Each new production root is independently locked and resumable; no two
coordinators may write the same root at once. Slurm jobs use `--time=0`,
confirmed as `TimeLimit=UNLIMITED` on this cluster. The launcher is
`scripts/slurm_newcluster_run.sh`. It stages the exact 3.9.2 environment and
assets on node-local `/tmp`, isolates per-job caches, and checks GPU visibility.

Node-l-03 advertises a GPU that the NVIDIA driver does not enumerate. A five-GPU
Slurm allocation showed only four driver-visible GPUs and Isaac failed with
`cudaErrorDevicesUnavailable`. Do not start further multi-GPU producers on
node-l-03 until the node is fixed or an allocation is proved fully visible.
The original two-GPU legacy continuation there is separate and still running.
The diagnostic `dataset_v11_production_ada_l03_20260924` has no committed
episode and is **not** part of the new-version 50-scene plan.

Monitor with `squeue -u usqki -o '%i %T %M %N %b %j'`, individual
`shards/<scene>/generation_status.json`, and `logs/<scene>.log`. Slurm
reports `Requeue=1`; each producer resumes from atomic committed episodes
after interruption. Do not interpret a stale `generation_failure.json` as
the status of a currently running retry.

After a node's coordinator exits and a scene shard is fully complete, use
`scripts/archive_completed_shard.py archive --dataset-root SOURCE --archive-root DEST --scene SCENE`
to checksum-copy it into a common new-version archive root. This tool refuses
live coordinators, incomplete shards, and fingerprint mismatches. Initially
leave both copies; use `--evict-after-verify --confirm-scene SCENE` only when
space must be reclaimed and the verified archive record has been inspected.
Never archive a legacy shard into the new-version archive root.
