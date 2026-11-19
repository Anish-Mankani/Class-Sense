
"""Camera-free test: project the 3D model at known angles, then recover them."""
import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_modules.head_pose import MODEL_POINTS, camera_matrix, estimate_pose
 
W, H = 1280, 720
 
def rot(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz
 
def simulate(yaw, pitch, roll, distance=1500.0, center=(640, 360), noise_px=0.0, seed=0):
    K = camera_matrix(W, H)
    rvec, _ = cv2.Rodrigues(rot(yaw, pitch, roll))
    tx = (center[0] - W / 2) * distance / K[0, 0]
    ty = (center[1] - H / 2) * distance / K[1, 1]
    pts, _ = cv2.projectPoints(MODEL_POINTS, rvec, np.array([[tx], [ty], [distance]]), K, None)
    pts = pts.reshape(5, 2)
    if noise_px:
        pts = pts + np.random.default_rng(seed).normal(0, noise_px, pts.shape)
    return pts
 
def run():
    cases = [(0, 0, 0), (25, 0, 0), (-25, 0, 0), (0, 20, 0), (0, -15, 0),
             (30, 15, 5), (-20, 25, -8), (0, 35, 0)]
    print(f"{'true (y,p,r)':>20} {'est (y,p,r)':>24} {'err':>6}")
    worst = 0
    for y, p, r in cases:
        est = estimate_pose(simulate(y, p, r), W, H)
        e = max(abs(est.yaw - y), abs(est.pitch - p), abs(est.roll - r))
        worst = max(worst, e)
        print(f"{str((y, p, r)):>20} {est.yaw:8.1f}{est.pitch:8.1f}{est.roll:8.1f} {e:6.2f}")
    assert worst < 1.0, f"noise-free error too large: {worst}"
 
    # Off-center face, plus 1px landmark noise: expect a few degrees of error.
    errs = [abs(estimate_pose(simulate(20, 10, 0, center=(300, 200), noise_px=1.0, seed=s), W, H).pitch - 10)
            for s in range(50)]
    print(f"\n1px noise: mean pitch error {np.mean(errs):.1f} deg, max {np.max(errs):.1f} deg")
 
    # Tiny face (far away) must be rejected.
    assert estimate_pose(simulate(0, 0, 0, distance=12000), W, H) is None
    print("Small-face rejection: OK\nAll tests passed.")
 
if __name__ == "__main__":
    run()
 