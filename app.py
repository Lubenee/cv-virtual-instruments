import time
from pathlib import Path
from hand_detection import Landmarker
import overlay
from settings_panel import SettingsPanel
from constants import HELP, HELP, WINDOW
import cv2

from settings_panel import SettingsPanel

ALL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")

class App:
    def __init__(self):
        self.keyboard = None
        self.calibration = None
        self.status, self.status_until = "", 0.0
        self.show_help = True
        self.flip_frame = False
        self.fps = 0.0
        self.frame_size = (640, 480)
        
        self.landmarker = Landmarker()
        self.settings_panel = SettingsPanel(self.landmarker)

    # ------------------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.settings_panel.on_mouse)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open the camera")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                self._update()
                self._draw(frame)
                if not self._poll_events(): break
            
        finally:
            cap.release()
            self.landmarker.close()
            cv2.destroyAllWindows()
    
    def _update(self): pass
    
    def _draw(self, frame):
        if self.flip_frame: frame = cv2.flip(frame, 1)
        result = self.landmarker.detect_for_video(frame)
        overlay.draw_landmarks(frame, result, True, True, True)
        self.settings_panel.draw(frame)
        
        if self.show_help:
            h = frame.shape[0]
            overlay.draw_text_lines(frame, HELP, origin=(10, h - 30 - 22 * len(HELP)), scale=0.5, color=(200, 255, 200))
        
        cv2.imshow(WINDOW, frame)
        
    def _poll_events(self):
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27: # q or esc
            return False
        elif key == ord('f'):
            self.flip_frame = not self.flip_frame
        elif key == ord('h'):
            self.show_help = not self.show_help
        return True