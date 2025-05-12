from ultralytics import YOLO

def initialize_model(model_name):
    # Load a pretrained YOLO model
    model = YOLO(f"trained_models/{model_name}.pt")
    return model

def predict(model, image_path, model_name):
    # Perform inference on a single image
    results = model.predict(image_path, conf=0.5, save=True, show=False, project=f"{model_name}.py_helmet_detections", verbose=False)
    return results

def contains_helmet(results):
    detections = results[0].boxes.data.tolist()
    for detection in detections:
        if detection[5] == 0:
            return '1'
    return '0'

def run_inference_helmet(image_path, model_name):
    model = initialize_model(model_name)
    results = predict(model, image_path, model_name)
    return contains_helmet(results)