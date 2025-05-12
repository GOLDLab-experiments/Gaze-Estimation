from openvino.runtime import Core
import cv2
import numpy as np
from gaze_estimation.FaceInferenceResults import FaceInferenceResults

class HeadPoseEstimator:
    """
    HeadPoseEstimator uses an OpenVINO model to estimate head pose angles (yaw, pitch, roll) from a face image.
    """
    def __init__(self, core: Core, model_path: str, device_name: str = 'CPU'):
        """
        Initialize the HeadPoseEstimator.

        Args:
            core (Core): OpenVINO Core object.
            model_path (str): Path to the head pose estimation model.
            device_name (str, optional): Device to run inference on. Defaults to 'CPU'.
        """
        # Load and compile the head pose estimation model for the specified device
        model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model=model, device_name=device_name)
        self.input_layer = self.compiled_model.input(0)
        # Map output names to indices for yaw, pitch, and roll
        self.output_map = {
            'angle_y_fc': 0,  # yaw
            'angle_p_fc': 1,  # pitch
            'angle_r_fc': 2,  # roll
        }
        # Ensure outputs are available by name
        for name in self.output_map:
            self.compiled_model.output(name)

    def estimate(self, image: np.ndarray, results: FaceInferenceResults):
        """
        Estimate head pose angles from a face image.

        Args:
            face_image (np.ndarray): Cropped face image.

        Returns:
            np.ndarray: Head pose angles [yaw, pitch, roll].
        """
        # Crop the face region from the image using the bounding box
        x, y, w, h = results.face_bounding_box
        face = image[y:y+h, x:x+w]
        # Resize and preprocess the face image for the model
        _, _, H, W = self.input_layer.shape
        resized = cv2.resize(face, (W, H))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        # Run inference to get head pose angles
        outputs = self.compiled_model([blob])
        angles = np.zeros(3, dtype=np.float32)
        for name, idx in self.output_map.items():
            angles[idx] = float(outputs[name][0])
        # Store the estimated angles in the results object
        results.head_pose_angles = angles