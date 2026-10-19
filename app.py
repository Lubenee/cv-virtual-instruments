import time
from pathlib import Path
from hand_detection import Landmarker
import overlay
from settings_panel import SettingsPanel
from constants import *
import cv2

from settings_panel import SettingsPanel

ALL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")

class App:
    def __init__(self):
        self.show_help = True
        self.flip_frame = False
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        
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
            self._release()
    
    def _update(self):
        self.frame_count += 1
        current_time = time.perf_counter()
        elapsed = current_time - self.last_fps_time
    
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
    
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _draw(self, frame):
        if self.flip_frame: frame = cv2.flip(frame, 1)
        result = self.landmarker.detect_for_video(frame)
        overlay.draw_landmarks(frame, result, True, True, True)
        self.settings_panel.draw(frame)
        
        if self.show_help:
            h = frame.shape[0]
            overlay.draw_text_lines(frame, HELP, origin=(10, h - 30 - 22 * len(HELP)), scale=0.5, color=BGR_WILLOW)
        overlay.draw_text_lines(frame, [f"FPS: {self.fps:.1f}"], origin=(20, 20), scale=0.5, color=BGR_WILLOW)
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
    
    def _release(self):
        self.landmarker.close()
        cv2.destroyAllWindows()