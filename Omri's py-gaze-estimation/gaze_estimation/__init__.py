# extern the module classes
from .estimators.face import FaceDetector
from .estimators.head_pose  import HeadPoseEstimator
from .estimators.gaze       import GazeEstimator
from .FaceInferenceResults import FaceInferenceResults
from .estimators.face_landmarks import LandmarksEstimator
from .utils import SmoothingFilter
__all__ = ["FaceDetector", "HeadPoseEstimator", "GazeEstimator", "FaceInferenceResults", "LandmarksEstimator", "SmoothingFilter"]

# check if ML models are available
import os
REQUIRED_MODELS = [
    "intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml",
    "intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml",
    "intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml",
    "intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml",
]
def _check_models():
    missing = [m for m in REQUIRED_MODELS if not os.path.exists(m)]
    if missing:
        raise RuntimeError(
            f"Required model files not found: {missing}\n"
            "Please run: python download_models.py"
        )
_check_models()