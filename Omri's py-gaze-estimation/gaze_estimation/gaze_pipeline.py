from openvino.runtime import Core
import numpy as np
from gaze_estimation.estimators.face import FaceDetector
from gaze_estimation.estimators.head_pose import HeadPoseEstimator
from gaze_estimation.estimators.face_landmarks import LandmarksEstimator
from gaze_estimation.estimators.gaze import GazeEstimator
from gaze_estimation import FaceInferenceResults

class GazePipeline:
    """
    GazePipeline wraps the full gaze estimation pipeline:
    face detection, head pose estimation, facial landmarks, and gaze estimation.

    Usage:
        core = Core()
        pipeline = GazePipeline(
            core,
            face_detector_model_path,
            head_pose_model_path,
            landmarks_model_path,
            gaze_model_path,
            device_name='CPU'
        )
        results = pipeline.process(image)
    """
    def __init__(
        self,
        core: Core,
        face_detector_path: str,
        head_pose_path: str,
        landmarks_path: str,
        gaze_path: str,
        device_name: str = 'CPU',
        smoothing_alpha: float = 0.5
    ):
        """
        Initialize the GazePipeline with all required models.

        Args:
            core (Core): OpenVINO Core object.
            face_detector_path (str): Path to face detector model.
            head_pose_path (str): Path to head pose estimator model.
            landmarks_path (str): Path to facial landmarks model.
            gaze_path (str): Path to gaze estimator model.
            device_name (str, optional): Device for inference. Defaults to 'CPU'.
        """
        self.face_detector = FaceDetector(core, face_detector_path, device_name)
        self.head_pose_estimator = HeadPoseEstimator(core, head_pose_path, device_name)
        self.landmarks_estimator = LandmarksEstimator(core, landmarks_path, device_name)
        self.gaze_estimator = GazeEstimator(core, gaze_path, device_name, smoothing_alpha=smoothing_alpha)

    def process(self, image: np.ndarray) -> list:
        """
        Run the full gaze estimation pipeline on an input image.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            list: List of FaceInferenceResults, one per detected face.
        """
        faces = self.face_detector.detect(image)
        results_list = []
        for results in faces:
            # Head pose estimation
            self.head_pose_estimator.estimate(image, results)

            # Landmarks estimation
            self.landmarks_estimator.estimate(image, results)

            # Gaze estimation
            self.gaze_estimator.estimate(image, results)

            results_list.append(results)
        return results_list