import cv2
import os
from openvino.runtime import Core
from gaze_estimation.gaze_pipeline import GazePipeline

def main():
    # Parse .env for model paths (expects environment variables to be set)
    fd_path = os.environ["FD_MODEL_PATH"]
    hp_path = os.environ["HP_MODEL_PATH"]
    lm_path = os.environ["LM_MODEL_PATH"]
    gz_path = os.environ["GZ_MODEL_PATH"]

    # Set up OpenVINO Core and the full gaze estimation pipeline
    core = Core()
    pipeline = GazePipeline(core, fd_path, hp_path, lm_path, gz_path, smoothing_alpha=0.2)

    # Parse .env for camera source (default: 0 = first webcam)
    cam_source = os.environ.get("CAMERA_SOURCE", "0")
    if cam_source.isdigit():
        cap = cv2.VideoCapture(int(cam_source))
    else:
        raise ValueError(f"Invalid camera source: {cam_source}")

    while True:
        frame = None
        if cap:
            # Capture a frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

        if frame is None:
            print("No frame captured. breaking...")
            break

        # Run the full pipeline on the current frame
        results = pipeline.process(frame)
        for r in results:
            # Draw face bounding box on the frame
            x, y, w, h = r.face_bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw gaze vector as an arrow from the center of the face
            cx, cy = x + w // 2, y + h // 2
            vx, vy = r.gaze_vector[0], -r.gaze_vector[1]  # Flip y for display
            cv2.arrowedLine(frame, (cx, cy),
                            (int(cx + vx * w / 2), int(cy + vy * h / 2)),
                            (0, 255, 0), 2, tipLength=0.2)

        # Show the annotated frame in a window
        cv2.imshow("Gaze", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if not cap:
            cv2.waitKey(0)
            break

    # Release camera and close windows on exit
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()