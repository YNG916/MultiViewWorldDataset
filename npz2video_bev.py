import cv2
import numpy as np

npz_path = "/cvhci/temp/yyang/behavior_world/output/mvwd/integration_1x5x3_20260902/episodes/Rs_int/config_002/episode_002/bev/world_before.npz"
output_path = "world_before.avi"

with np.load(npz_path) as data:
    rgba = data["rgb"]

rgb = rgba[..., :3]

T, H, W, _ = rgb.shape

if rgb.dtype != np.uint8:
    if rgb.max() <= 1.0:
        rgb = rgb * 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"MJPG"),
    10,
    (W, H),
)

assert writer.isOpened()

for frame in rgb:
    frame_bgr = cv2.cvtColor(
        np.ascontiguousarray(frame),
        cv2.COLOR_RGB2BGR
    )
    writer.write(frame_bgr)

writer.release()

print("Saved:", output_path)