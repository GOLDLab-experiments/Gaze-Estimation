import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

class HelmetDetector:
    def __init__(self, model_name="yolo_helmet.pt"):
        """
        Initializes the HelmetDetector with the YOLO model.
        """
        self.model_name = model_name
        self.model = YOLO(self.model_name)

    def contains_helmet(self, results):
        """
        Checks if the results contain a helmet detection.
        Returns the bounding box [x1, y1, x2, y2] if a helmet is detected, otherwise an empty list.
        """
        detections = results[0].boxes.data.tolist()
        for detection in detections:
            if detection[5] == 0:  # Assuming class 0 corresponds to a helmet
                return [detection[0], detection[1], detection[2], detection[3]]
        return []

    def run_inference(self, image: cv2.Mat):
        """
        Runs inference on the given image to detect helmets.
        Returns the bounding box of the detected helmet or an empty list if no helmet is detected.
        """
        results = self.model.predict(image, conf=0.5, save=False, show=False, verbose=False)
        return self.contains_helmet(results)
    
    """
    # Example usage
    detector = HelmetDetector()

    # Load an image
    image = cv2.imread("path/to/image.jpg")

    # Run inference
    helmet_box = detector.run_inference(image)
    if helmet_box:
        print("Helmet detected at:", helmet_box)
    else:
        print("No helmet detected.")
    """