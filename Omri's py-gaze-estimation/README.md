# py-gaze-estimation

A Python module for real-time gaze vector estimation using [OpenVINO™](https://docs.openvino.ai/2025/index.html) and pre-trained Intel models.

---

## Features

- **Face Detection**: Detects faces in images or video streams.
- **Head Pose Estimation**: Estimates yaw, pitch, and roll of detected faces.
- **Facial Landmarks**: Locates key facial points, including eyes.
- **Gaze Estimation**: Computes 3D gaze vectors for each detected face.
- **Filtering**: Smooths and stabilizes gaze vectors using configurable filters.
- **Pipeline API**: Compose and run the full gaze estimation pipeline with a single class.
- **Easy Integration**: Modular design for use in your own projects.
- **OpenVINO-powered**: Fast inference on CPU and other supported devices.

---

## Requirements

- Python 3.8–3.12
- [OpenVINO™](https://pypi.org/project/openvino-dev/)
- [OpenCV](https://pypi.org/project/opencv-python/)

---

## Installation

1. **Clone the repository**
    ```sh
    git clone https://github.com/OmriPer/py-gaze-estimation.git
    cd py-gaze-estimation
    ```

2. **Create a virtual environment**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install dependencies**
    ```sh
    pip install -e .
    ```

4. **Download the required models**
    ```sh
    python download_models.py
    ```
    This will use [models.lst](models.lst) and the Open Model Zoo downloader to fetch all required models into the `intel/` directory.

---

## Quick Start

1. **Configure environment variables**

    Copy `.env.example` to `.env` and adjust paths if needed:
    ```sh
    cp .env.example .env
    ```

    The `.env` file should look like:
    ```
    FD_MODEL_PATH=intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml
    HP_MODEL_PATH=intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
    GZ_MODEL_PATH=intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml
    LM_MODEL_PATH=intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002.xml
    CAMERA_SOURCE=0
    ```

2. **Run the demo**
    ```sh
    python test/test_gaze.py
    ```
    This will open your default camera and display real-time gaze estimation.

---

## Usage in Your Code

You can use the core classes in your own scripts:

```python
from openvino.runtime import Core
from gaze_estimation import FaceDetector, HeadPoseEstimator, LandmarksEstimator, GazeEstimator, Filter, Pipeline

core = Core()
fd = FaceDetector(core, "path/to/face-detection-retail-0004.xml")
hp = HeadPoseEstimator(core, "path/to/head-pose-estimation-adas-0001.xml")
lm = LandmarksEstimator(core, "path/to/facial-landmarks-35-adas-0002.xml")
gz = GazeEstimator(core, "path/to/gaze-estimation-adas-0002.xml")

# Use OpenCV to capture frames, then:
results = fd.detect(frame)
for r in results:
    hp.estimate(frame, r)
    lm.estimate(frame, r)
    r.right_eye_bounding_box = lm.compute_eye_bbox(r.face_landmarks, "right_eye")
    r.left_eye_bounding_box  = lm.compute_eye_bbox(r.face_landmarks, "left_eye")
    gz.estimate(frame, r)
    r.gaze_vector = gaze_filter(r.gaze_vector)
    print("Gaze vector:", r.gaze_vector)
```

Or use the Pipeline class for end-to-end processing:
```python
from openvino.runtime import Core
from gaze_estimation import Pipeline
core = Core()
pipeline = Pipeline(core,
    fd_model="path/to/face-detection-retail-0004.xml",
    hp_model="path/to/head-pose-estimation-adas-0001.xml",
    lm_model="path/to/facial-landmarks-35-adas-0002.xml",
    gz_model="path/to/gaze-estimation-adas-0002.xml"
)
results = pipeline.process(frame)
for r in results:
    print("Pipeline gaze vector:", r.gaze_vector)
```

---

## Model Download & Management

- Models are listed in [models.lst](models.lst).
- Use `python download_models.py` to fetch all required models.
- Models are stored in the `intel/` directory (ignored by git).
- **Note:** The package will check for required model files on import.  
  If any models are missing, you will see an error like:
  ```
  RuntimeError: Required model files not found: [...]
  Please run: python download_models.py
  ```
  Make sure to download the models before using the library.

---

## Project Structure

```
gaze_estimation/      # Core Python package
    ├── estimators/   # Face, head pose, landmarks, and gaze estimators (submodule)
    │     ├── face.py
    │     ├── head_pose.py
    │     ├── face_landmarks.py
    │     └── gaze.py
    ├── pipeline.py   # Pipeline class for end-to-end processing
    ├── filter.py     # Filtering utilities for smoothing outputs
    ├── utils.py      # Utility functions and classes
    └── __init__.py   # Package exports
test/                 # Example/demo scripts
intel/                # Downloaded OpenVINO models (not included in repo)
.env                  # Environment variables for model paths and camera
models.lst            # List of required models for download
download_models.py    # Script to download models
pyproject.toml        # Build and dependency info
README.md             # This file
```

---

## License

TBD

---

## Acknowledgements

- [Intel Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
- [OpenVINO™ Toolkit](https://github.com/openvinotoolkit/openvino)