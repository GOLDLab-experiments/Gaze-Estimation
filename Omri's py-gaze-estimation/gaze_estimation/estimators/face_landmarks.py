from openvino.runtime import Core
import cv2
import numpy as np
from gaze_estimation import FaceInferenceResults

# The indices of the landmarks for the right and left eye (for the 35-point model)
LANDMARK_IDX = {
    'right_eye': [2,3,15,16,17],
    'left_eye': [0,1,12,13,14],
}

class LandmarksEstimator:
    """
    LandmarksEstimator uses an OpenVINO model to estimate facial landmarks from a face image.
    """
    def __init__(self, core: Core, model_path: str, device_name: str = 'CPU'):
        """
        Initialize the LandmarksEstimator.

        Args:
            core (Core): OpenVINO Core object.
            model_path (str): Path to the facial landmarks model.
            device_name (str, optional): Device to run inference on. Defaults to 'CPU'.
        """
        # Load and compile the facial landmarks model for the specified device
        model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model=model, device_name=device_name)
        self.input_layer = self.compiled_model.input(0)
        outputs = self.compiled_model.outputs
        # Ensure the model has a single output
        if len(outputs) != 1:
            raise ValueError("Landmarks model must have a single output")
        self.output_layer = outputs[0]
        self.output_shape = tuple(self.output_layer.shape)

    def estimate(self, image: np.ndarray, results: FaceInferenceResults):
        """
        Estimate facial landmarks from a face image.

        Args:
            face_image (np.ndarray): Cropped face image.

        Returns:
            list: List of (x, y) tuples representing landmark coordinates.
        """
        # Crop the face region from the image using the bounding box
        x, y, w, h = results.face_bounding_box
        face_crop = image[y:y+h, x:x+w]
        # Resize and preprocess the face crop for the model
        _, _, H, W = self.input_layer.shape
        resized = cv2.resize(face_crop, (W, H))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        # Run inference to get landmark coordinates
        outputs = self.compiled_model([blob])[self.output_layer]
        out = outputs[0]
        # Postprocess output to get landmark points in image coordinates
        results.face_landmarks = self._simple_postprocess(out, x, y, w, h)
        # store the eye bounding boxes in the results object
        results.right_eye_bounding_box = self._compute_eye_bbox(results.face_landmarks, 'right_eye')
        results.left_eye_bounding_box = self._compute_eye_bbox(results.face_landmarks, 'left_eye')

    def _simple_postprocess(self, raw, x, y, w, h):
        # Convert raw model output to a list of (x, y) landmark points in image coordinates
        num = raw.size // 2
        pts = []
        flat = raw.flatten()
        for i in range(num):
            px = int(flat[2*i] * w + x)
            py = int(flat[2*i+1] * h + y)
            pts.append((px, py))
        return pts
    
    def _compute_eye_bbox(self, landmarks: list, eye: str, padding: int = 5) -> tuple:
        # Compute a bounding box around the specified eye using landmark indices
        xs = [landmarks[i][0] for i in LANDMARK_IDX[eye]]
        ys = [landmarks[i][1] for i in LANDMARK_IDX[eye]]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # Add padding to the bounding box
        return (
            x_min - padding,
            y_min - padding,
            (x_max - x_min) + 2 * padding,
            (y_max - y_min) + 2 * padding
        )