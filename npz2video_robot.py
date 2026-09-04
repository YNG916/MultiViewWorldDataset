import cv2
import numpy as np
from pathlib import Path

npz_path = "/cvhci/temp/yyang/behavior_world/output/mvwd/integration_1x5x3_20260902/episodes/Rs_int/config_002/episode_002/robot_views/before/robot_02.npz"
output_path = "robot_02.avi"

with np.load(npz_path) as data:
    rgb = data["rgb"]

print("shape:", rgb.shape)
print("dtype:", rgb.dtype)
print("range:", rgb.min(), rgb.max())

assert rgb.ndim == 4
assert rgb.shape[-1] == 3

T, H, W, C = rgb.shape

# 确保是 uint8 [0, 255]
if rgb.dtype != np.uint8:
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"MJPG"),
    10,      # FPS
    (W, H),
)

if not writer.isOpened():
    raise RuntimeError("Failed to open MJPG VideoWriter")

for frame in rgb:
    frame = np.ascontiguousarray(frame)

    # NPZ 是 RGB，OpenCV 写视频使用 BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    writer.write(frame_bgr)

writer.release()

path = Path(output_path)

print("Saved:", path.resolve())
print("Size:", path.stat().st_size / 1024 / 1024, "MB")