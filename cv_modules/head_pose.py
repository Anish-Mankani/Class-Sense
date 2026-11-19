from dataclasses import dataclass
import math
import numpy as np
import cv2
 
# Generic face model in mm. Camera-style axes: x right, y down, z away from
# camera. Nose tip is the origin and sticks out toward the camera (smaller z).
# Order matches YuNet: right eye, left eye, nose, right mouth, left mouth
# (YuNet's "right" is the subject's right, i.e. image-left for a frontal face).
MODEL_POINTS = np.array([
    [-32.0, -35.0, 30.0],   # right eye (image-left)
    [32.0, -35.0, 30.0],    # left eye  (image-right)
    [0.0, 0.0, 0.0],        # nose tip
    [-25.0, 40.0, 25.0],    # right mouth corner
    [25.0, 40.0, 25.0],     # left mouth corner
], dtype=np.float64)
 
 
@dataclass
class Pose:
    yaw: float
    pitch: float
    roll: float
 
 
def camera_matrix(frame_w, frame_h, focal=None):
    f = focal if focal else float(frame_w)  # rough guess; calibrate if you can
    return np.array([[f, 0, frame_w / 2.0],
                     [0, f, frame_h / 2.0],
                     [0, 0, 1]], dtype=np.float64)
 
 
def rotation_to_euler(R):
    """R = Ry(yaw) @ Rx(pitch) @ Rz(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    yaw = math.atan2(R[0, 2], R[2, 2])
    roll = math.atan2(R[1, 0], R[1, 1])
    return Pose(*(math.degrees(a) for a in (yaw, pitch, roll)))
 
 
def estimate_pose(landmarks, frame_w, frame_h, min_face_px=40, focal=None):
    """landmarks: 5x2 array in YuNet order. Returns Pose, or None if the face
    is too small (eye distance in pixels < min_face_px) or the solve fails."""
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    eye_dist = np.linalg.norm(pts[0] - pts[1])
    if eye_dist < min_face_px:
        return None
 
    K = camera_matrix(frame_w, frame_h, focal)
    # Initial guess: frontal face at a distance implied by the eye spacing.
    dist = K[0, 0] * 64.0 / eye_dist
    rvec0 = np.zeros((3, 1))
    tvec0 = np.array([[0.0], [0.0], [dist]])
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, pts, K, None, rvec0, tvec0,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return rotation_to_euler(R)
 
 
def pose_from_yunet_row(row, frame_w, frame_h, **kw):
    """row: one YuNet detection [x,y,w,h, 10 landmark values, score]."""
    return estimate_pose(np.asarray(row[4:14]).reshape(5, 2), frame_w, frame_h, **kw)
 
 
def eye_distance(landmarks):
    """Pixel distance between the two eyes (first two YuNet landmarks)."""
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    return float(np.linalg.norm(pts[0] - pts[1]))
 
 
def head_pose(landmarks, frame_size):
    """Returns (yaw, pitch, roll) in degrees, or None. frame_size = (width, height).
    The minimum-size check is left to attention.py (min_eye_distance)."""
    w, h = frame_size
    pose = estimate_pose(landmarks, w, h, min_face_px=1)
    return None if pose is None else (pose.yaw, pose.pitch, pose.roll)
 