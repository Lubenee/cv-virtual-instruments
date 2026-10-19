import mediapipe as mp
import cv2
import time
from pathlib import Path
from constants import BGR_BLUE, BGR_RED, BGR_GREEN, BGR_BLACK

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FINGERTIP_LANDMARKS = [4, 8, 12, 16, 20]
LANDMARKER_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
DEFAULT_SETTINGS = (.5, .5, .5)

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class Landmarker:
    def __init__(self, settings=DEFAULT_SETTINGS):
        if not LANDMARKER_MODEL_PATH.exists():
            raise FileNotFoundError(f"Hand Landmarker model not found: {LANDMARKER_MODEL_PATH}")
        self._settings = tuple(settings)
        self._landmarker = self._create_landmarker(settings)
        self._start_time = time.monotonic()

    @property
    def landmarker(self):
        return self._landmarker

    def update_settings(self, new_settings):
        new_settings = tuple(new_settings)

        if new_settings != self._settings:
            self._landmarker.close()
            self._landmarker = self._create_landmarker(new_settings)
            self._settings = new_settings
            
    def close(self):
        self._landmarker.close()

    def detect_for_video(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def _create_landmarker(settings):
        detection, presence, tracking = settings

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(LANDMARKER_MODEL_PATH)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=detection,
            min_hand_presence_confidence=presence,
            min_tracking_confidence=tracking
        )

        return HandLandmarker.create_from_options(options)
    
############################################
## C++ class equivalents cuz im retarded
# lassName() constructor	__init__(self) initializer
# this->variable	        self.variable
# private:	                _variable convention
# public:	                Normal methods/attributes
# static void foo()	        @staticmethod
# const member	            No direct equivalent
# Getter method	            @property
# Destructor ~ClassName()	Explicit close(), often with with
# nullptr	                None
# obj.method()	            obj.method()