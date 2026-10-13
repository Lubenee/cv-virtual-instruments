
import cv2
from constants import WINDOW
from hand_detection import Landmarker
from settings_panel import SettingsPanel

def main():
    cv2.namedWindow(WINDOW)
    lm = Landmarker()
    settings_panel = SettingsPanel(lm)
    cv2.setMouseCallback(WINDOW, settings_panel.on_mouse)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            result = lm.detect_for_video(frame)
            lm.draw_landmarks(frame, result, True, True, True)
            settings_panel.draw(frame)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
            
    finally:
        cap.release()
        lm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
