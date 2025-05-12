from dataclasses import dataclass
import numpy as np

@dataclass
class FaceInferenceResults:
    """
    Stores inference results for a detected face, including bounding box, confidence, and additional attributes.
    """
    face_detection_confidence: float = 0.0
    face_bounding_box: tuple = (0, 0, 0, 0)  # x, y, w, h
    left_eye_bounding_box: tuple | None = None
    right_eye_bounding_box: tuple | None = None
    head_pose_angles: np.ndarray | None = None  # [yaw, pitch, roll]
    face_landmarks: list | None = None  # list of (x, y) tuples
    gaze_vector: np.ndarray | None = None      # normalized 3D vector
    leftEyeState: bool = True
    rightEyeState: bool = True