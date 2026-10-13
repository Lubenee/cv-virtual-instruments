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
    def __init__(self, settings):
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
            
    def draw_landmarks(self, frame, result, draw_hand_skeleton=False, draw_fingertips_only=True, draw_fingertip_coords=True):
        height, width = frame.shape[:2]

        for hand in result.hand_landmarks:
            # Convert normalized coordinates to pixels
            points = [(int(lm.x * width), int(lm.y * height)) for lm in hand]

            # Draw hand skeleton
            if draw_hand_skeleton:
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], BGR_GREEN, 1)

            # Draw all landmarks
            for i, point in enumerate(points):
                color = BGR_BLACK
                fingertip_landmark = i in FINGERTIP_LANDMARKS
                if fingertip_landmark:
                    color = BGR_RED
                else: 
                    color = BGR_BLUE
                
                if fingertip_landmark or not draw_fingertips_only:
                    cv2.circle(frame, point, 5, color, -1)

            # Print index fingertip coordinates on screen
            if draw_fingertip_coords:
                index_x, index_y = points[8]
                cv2.putText(
                    frame,
                    f"Index: ({index_x}, {index_y})",
                    (index_x + 10, index_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

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