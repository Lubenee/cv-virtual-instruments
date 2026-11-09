import time
from pathlib import Path
from hand_detection import Landmarker
import overlay
from settings_panel import SettingsPanel
from constants import *
import cv2

from settings_panel import SettingsPanel
from keyboard_detector import KeyboardDetector
from finger_detector import FingerKeyDetector

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
        self.keyboard_detector = KeyboardDetector()
        self.finger_detector = FingerKeyDetector()

    # ------------------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        
        # this opens the camera
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
        # Find the hands, then the fingertips inside the (locked) keys
        result = self.landmarker.detect_for_video(frame)
        touches = self.finger_detector.detect(result, frame.shape, self.keyboard_detector.keyboard)

        # Detect & draw the keyboard first, while the frame is still clean
        self.keyboard_detector.draw_keyboard(frame, hovered_notes={t.key.note for t in touches})
        overlay.draw_landmarks(frame, result, True, True, True)
        # self.settings_panel.draw(frame)
        overlay.draw_calibration_points(frame, self.keyboard_detector.get_selected_quad())
        overlay.draw_text_lines(frame, self._status_lines(touches), origin=(20, 60), scale=0.6)
        if self.show_help:
            h = frame.shape[0]
            overlay.draw_text_lines(frame, HELP, origin=(10, h - 40 - 22 * len(HELP)))
        if self.keyboard_detector.manual_calibration:
            overlay.draw_text_lines(frame, ["Input a quad!"], origin=(20, 80), scale=1.2, color=BGR_PINK)
        overlay.draw_text_lines(frame, [f"FPS: {self.fps:.1f}"], origin=(20, 30))
        cv2.imshow(WINDOW, frame)

    def _status_lines(self, touches):
        detector = self.keyboard_detector
        keyboard = detector.keyboard
        if keyboard is None:
            if len(detector.selected_corners) != 4:
                return []
            return [f"Detecting: {len(detector.white_keys)} white, {len(detector.black_keys)} black keys"
                    "   l: lock them in"]

        fingers = "index finger only" if self.finger_detector.index_only else "all fingers"
        lines = [
            f"LOCKED: {keyboard.white_count} white, {keyboard.black_count} black keys "
            f"({keyboard.keys[0].name} - {keyboard.keys[-1].name})   l: unlock",
            f"Fingers: {fingers}   i: toggle",
        ]
        if not keyboard.pattern_ok:
            lines.append("Keys don't follow the piano pattern - some were missed, notes may be off")
        if touches:
            lines.append("On keys: " + ", ".join(f"{t.key.name} ({t.hand} {t.finger})" for t in touches))
        return lines

    def _poll_events(self):
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q') or key == 27: # q or esc
            return False
        elif key == ord('f'):
            self.flip_frame = not self.flip_frame
        elif key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord('c'):
            self.keyboard_detector.set_manual_calibration(not self.keyboard_detector.manual_calibration)
        elif key == ord('l'):
            if self.keyboard_detector.locked:
                self.keyboard_detector.unlock()
            else:
                self.keyboard_detector.lock()
        elif key == ord('i'):
            self.finger_detector.toggle_index_only()
        return True
    
    def _on_mouse(self, event, x, y, flags, param):
        # self.settings_panel.on_mouse(event,x,y,flags,param)
        self.keyboard_detector.on_mouse(event,x,y,flags,param)
        
    def _release(self):
        self.landmarker.close()
        cv2.destroyAllWindows()