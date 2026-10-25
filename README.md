# FaceView — Multi-Angle Face Orientation Detection

FaceView is a real-time computer-vision project that uses a webcam to estimate whether a face is oriented approximately forward, left, or right.

The project includes a simple baseline approach and an updated pose-estimation approach.

## Features

- Live webcam video
- Facial landmark detection with MediaPipe Face Mesh
- Face orientation labels: `Front Face`, `Left Side Face`, and `Right Side Face`
- Estimated yaw, pitch, and roll values in the updated implementation
- On-screen face landmarks, pose axes, and approximate FPS display
- Press `q` to exit the webcam window

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy

## Project files

Place these files in the repository root:

```text
faceview-mediapipe/
├── main_updated.py
├── main_baseline.py
├── README.md
├── requirements.txt
├── FaceView_Learning_Guide.pdf
└── .gitignore
```

- `main_updated.py` — updated head-pose estimation implementation.
- `main_baseline.py` — baseline 2D facial-symmetry approach.
- `FaceView_Learning_Guide.pdf` — learning guide, basic-vs-updated comparison, pipeline explanation, and interview revision notes.

## Setup (Windows PowerShell)

Run these commands from the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the updated version

```powershell
.\.venv\Scripts\python.exe -u main_updated.py
```

## Run the baseline version

```powershell
.\.venv\Scripts\python.exe main_baseline.py
```

Run only one version at a time because both use the webcam. Press `q` in the OpenCV video window to close the application.

## How it works

1. Capture a frame from the webcam.
2. Convert the frame to the color format expected by MediaPipe.
3. Detect facial landmarks.
4. Match selected 2D facial landmarks with a generic 3D reference-face model.
5. Use OpenCV `solvePnP` to estimate pose, then derive yaw, pitch, and roll.
6. Use the yaw estimate and a threshold to classify the face as front-facing, left-facing, or right-facing.
7. Display the annotated webcam feed and estimated values.

The baseline version instead compares horizontal eye-to-nose distances as a simple 2D symmetry heuristic. It distinguishes front/side, but does not estimate a 3D pose angle or separate the two side directions.

## Limitations

- The 3D reference face is generic rather than individually calibrated.
- The camera matrix uses approximate intrinsic parameters; the camera has not been calibrated as part of this project.
- Angle estimates can be affected by landmark quality, lighting, camera position, and model assumptions.
- The yaw threshold is a chosen rule, not a trained orientation classifier.
- Pitch and roll have not been comprehensively validated for accuracy. Treat all displayed angles as estimates, not ground-truth measurements.

## Validation performed

Manual webcam checks were performed for:
- Looking approximately straight ahead
- Turning the head left and right
- Keeping the head approximately forward during an eyes-only movement check
- Exiting with `q`

These are basic functional checks, not a formal accuracy benchmark or exhaustive failure-path test.

## Learning guide

See [`FaceView_Learning_Guide.pdf`](FaceView_Learning_Guide.pdf) for the detailed learning and interview guide.

## Future improvements

- Calibrate the camera and validate angle estimates against a reference.
- Test under different lighting, distances, and head poses.
- Improve handling of camera errors and pose-estimation failures.
- Measure accuracy and performance systematically.

## License

Add a license only when you have chosen the terms under which others may use or modify this project.
