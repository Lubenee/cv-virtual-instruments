# cv-virtual-instruments

A computer vision project using OpenCV, MediaPipe, and NumPy to detect piano keys and track hand movements through a webcam.

## Prerequisites

- Python 3.11 or later (3.11 recommended)
- A working webcam
- pip

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lubenee/cv-virtual-instruments.git
cd cv-virtual-instruments
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

Alternatively, if a `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

### 4. Download the model

Download the MediaPipe Hand Landmarker `.task` model from [MediaPipe's documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

Place the model at the location specified by `LANDMARKER_MODEL_PATH` in the source code.

## Running the Application

With the virtual environment activated, run:

```bash
python main.py
```

Replace `main.py` with the actual entry-point filename if different.

Allow camera access when prompted.

Press **Q** to exit the application.