from openvino.runtime import Core
import cv2
import numpy as np
import gaze_estimation.utils as utils
from gaze_estimation.FaceInferenceResults import FaceInferenceResults

class GazeEstimator:
    """
    GazeEstimator uses an OpenVINO model to estimate gaze direction from eye images and head pose angles.
    """
    def __init__(self, core: Core, model_path: str, device_name: str = 'CPU',
                 do_roll_align: bool = False, smoothing_alpha: float = 0.5):
        """
        Initialize the GazeEstimator.

        Args:
            core (Core): OpenVINO Core object.
            model_path (str): Path to the gaze estimation model.
            device_name (str, optional): Device to run inference on. Defaults to 'CPU'.
            do_roll_align (bool, optional): Whether to align roll angle. Defaults to False.
            smoothing_alpha (float, optional): Smoothing factor for gaze vector. Defaults to 0.5.
        """
        # Load and compile the gaze estimation model for the specified device
        model = core.read_model(model=model_path)
        self.compiled_model = core.compile_model(model=model, device_name=device_name)
        # Prepare input and output layers for inference
        self.inputs = {
            'head_pose_angles': self.compiled_model.input('head_pose_angles'),
            'left_eye_image': self.compiled_model.input('left_eye_image'),
            'right_eye_image': self.compiled_model.input('right_eye_image'),
        }
        self.output_layer = self.compiled_model.output(0)
        self.roll_align = do_roll_align
        # Optionally enable smoothing of the gaze vector
        self.smoothing = utils.SmoothingFilter(alpha=smoothing_alpha) if smoothing_alpha < 1 else None

    @staticmethod
    def rotate_image_around_center(src, angle):
        """
        Rotate an image around its center.

        Args:
            src (np.ndarray): Source image.
            angle (float): Angle to rotate the image.

        Returns:
            np.ndarray: Rotated image.
        """
        # Rotate an image around its center by the given angle (in degrees)
        h, w = src.shape[:2]
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def estimate(self, image: np.ndarray, results: FaceInferenceResults):
        """
        Estimate gaze direction from eye images and head pose angles.

        Args:
            image (np.ndarray): Input image.
            results (FaceInferenceResults): Inference results containing eye states and head pose angles.

        Returns:
            None
        """
        # Estimate gaze direction from the input image and face inference results
        # Only proceed if both eyes are open
        if not results.leftEyeState or not results.rightEyeState:
            return
        assert results.head_pose_angles is not None, "Head pose angles are required for gaze estimation"
        yaw, pitch, roll = results.head_pose_angles
        angles = np.array([[yaw, pitch, roll]], dtype=np.float32)
        # Crop left and right eye images using bounding boxes
        assert results.left_eye_bounding_box is not None, "Left eye bounding box is required"
        assert results.right_eye_bounding_box is not None, "Right eye bounding box is required"
        lx, ly, lw, lh = results.left_eye_bounding_box
        rx, ry, rw, rh = results.right_eye_bounding_box
        left_eye = image[ly:ly+lh, lx:lx+lw]
        right_eye = image[ry:ry+rh, rx:rx+rw]
        # Optionally align eyes by removing roll angle
        if self.roll_align:
            angles[0,2] = 0.0
            left_eye = self.rotate_image_around_center(left_eye, roll)
            right_eye = self.rotate_image_around_center(right_eye, roll)
        # Preprocess eye images for model input
        _, _, H, W = self.inputs['left_eye_image'].shape
        def preproc(img):
            im = cv2.resize(img, (W, H))
            b = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return b.transpose(2,0,1)[np.newaxis, ...]
        left_blob = preproc(left_eye)
        right_blob = preproc(right_eye)
        infer_inputs = {
            'head_pose_angles': angles,
            'left_eye_image': left_blob,
            'right_eye_image': right_blob,
        }
        # Run inference to get the gaze vector
        outputs = self.compiled_model(infer_inputs)[self.output_layer]
        gaze = outputs[0]
        gaze = gaze / np.linalg.norm(gaze)

        # Optionally smooth the gaze vector
        if self.smoothing is not None:
            gaze = self.smoothing.update(gaze)
        # Optionally rotate the gaze vector back by the roll angle
        if self.roll_align:
            cs = np.cos(np.deg2rad(roll))
            sn = np.sin(np.deg2rad(roll))
            xg, yg = gaze[0], gaze[1]
            gaze[0] = xg*cs + yg*sn
            gaze[1] = -xg*sn + yg*cs
        # Store the normalized gaze vector in the results object
        results.gaze_vector = gaze
