
import cv2
import time
import mediapipe as mp
from pathlib import Path

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Connections between the 21 hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGERTIPS = [4, 8, 12, 16, 20]


def draw_landmarks(frame, result):
    height, width = frame.shape[:2]

    for hand in result.hand_landmarks:
        # Convert normalized coordinates to pixels
        points = [
            (int(lm.x * width), int(lm.y * height))
            for lm in hand
        ]

        # Draw hand skeleton
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        # Draw all landmarks
        for i, point in enumerate(points):
            color = (0, 255, 255) if i in FINGERTIPS else (0, 0, 255)

            cv2.circle(frame, point, 5, color, -1)

        # Print index fingertip coordinates on screen
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


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open the camera")

    start_time = time.monotonic()

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Mirror the camera image
                frame = cv2.flip(frame, 1)

                # OpenCV BGR -> MediaPipe RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # VIDEO mode requires a timestamp in milliseconds
                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                # Detect hands
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # Draw landmarks
                draw_landmarks(frame, result)

                cv2.imshow("Virtual Piano", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
