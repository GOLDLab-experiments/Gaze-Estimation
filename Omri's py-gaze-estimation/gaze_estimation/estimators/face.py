from openvino.runtime import Core
import cv2
import numpy as np
from gaze_estimation.FaceInferenceResults import FaceInferenceResults

class FaceDetector:
    """
    FaceDetector uses an OpenVINO model to detect faces in images.
    It provides bounding box adjustment and detection confidence thresholding.
    """
    def __init__(self, core: Core, model_path: str, device_name: str = 'CPU',
                 detection_confidence_threshold: float = 0.5, enable_reshape: bool = True):
        """
        Initialize the FaceDetector.

        Args:
            core (Core): OpenVINO Core object.
            model_path (str): Path to the face detection model.
            device_name (str, optional): Device to run inference on. Defaults to 'CPU'.
            detection_confidence_threshold (float, optional): Minimum confidence for detections. Defaults to 0.5.
            enable_reshape (bool, optional): Whether to enable model reshaping. Defaults to True.
        """
        # Load the OpenVINO model and compile it for the specified device
        # Set detection threshold and enable/disable model reshaping
        model = core.read_model(model=model_path)
        self.enable_reshape = enable_reshape
        self.detection_threshold = detection_confidence_threshold
        self.compiled_model = core.compile_model(model=model, device_name=device_name)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        out_shape = list(self.output_layer.shape)
        # Validate output shape to ensure model compatibility
        if len(out_shape) != 4 or out_shape[0] != 1 or out_shape[1] != 1 or out_shape[3] != 7:
            raise ValueError(f"Expected output shape [1,1,N,7], got {out_shape}")
        self.num_detections = out_shape[2]

    def adjust_bounding_box(self, bbox):
        """
        Adjust the bounding box to better fit the face region.

        Args:
            bbox (tuple): Bounding box (x, y, w, h).

        Returns:
            tuple: Adjusted bounding box (x, y, w, h).
        """
        # Adjust the bounding box to better fit the face region
        # Expands and centers the box based on heuristics for better cropping
        x, y, w, h = bbox
        x -= int(0.067 * w)
        y -= int(0.028 * h)
        w += int(0.15 * w)
        h += int(0.13 * h)
        if w < h:
            dx = (h - w)
            x -= dx // 2
            w += dx
        else:
            dy = (w - h)
            y -= dy // 2
            h += dy
        return (x, y, w, h)

    def detect(self, image: np.ndarray):
        """
        Detect faces in the input image.

        Args:
            image (np.ndarray): Input image (BGR format).

        Returns:
            list[FaceInferenceResults]: List of detected faces with bounding boxes and confidence.
        """
        # Run face detection on the input image
        # Preprocess image to model input size and format
        _, _, h, w = self.input_layer.shape
        resized = cv2.resize(image, (w, h))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        # Run inference
        outputs = self.compiled_model([blob])[self.output_layer]
        outputs = outputs[0][0]
        results = []
        ih, iw = image.shape[:2]
        for det in outputs:
            conf = float(det[2])
            if conf < self.detection_threshold:
                continue
            # Convert normalized coordinates to image coordinates
            x_min = int(det[3] * iw)
            y_min = int(det[4] * ih)
            x_max = int(det[5] * iw)
            y_max = int(det[6] * ih)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            adj = self.adjust_bounding_box(bbox)
            ax, ay, aw, ah = adj
            # Skip boxes that are out of image bounds
            if ax < 0 or ay < 0 or ax+aw > iw or ay+ah > ih:
                continue
            # Store detection result
            fim = FaceInferenceResults()
            fim.face_detection_confidence = conf
            fim.face_bounding_box = adj
            results.append(fim)
        return results