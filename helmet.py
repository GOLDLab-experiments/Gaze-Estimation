import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

model_name = "yolo_helmet.pt"

def contains_helmet(results):
    detections = results[0].boxes.data.tolist()
    for detection in detections:
        if detection[5] == 0:
            return [detection[0], detection[1], detection[2], detection[3]]
    return []

def run_inference_helmet(image:cv2.Mat):
    model = YOLO(model_name)
    results = model.predict(image, conf=0.5, save=False, show=False, verbose=False)
    # results = model.predict(image_path, conf=0.5, save=True, show=False, project=f"{model_name}.py_helmet_detections", verbose=False)
    return contains_helmet(results)