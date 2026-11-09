"""A locked-in keyboard: the keys KeyboardDetector found, frozen in place.

Once locked, the keys stop moving even when hands cover them, and every key knows its note.
"""
from dataclasses import dataclass

import cv2
import numpy as np

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Which notes of an octave (starting at C) are black keys
OCTAVE_IS_BLACK = tuple("#" in name for name in NOTE_NAMES)
# The leftmost key is placed in the octave of middle C (MIDI 60 = C4)
BASE_OCTAVE_NOTE = 60


def note_name(note: int) -> str:
    """MIDI note number -> name, e.g. 60 -> 'C4'."""
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def roi_box_to_frame(box, M_inv) -> np.ndarray:
    """Map an (x, y, w, h) box from ROI coordinates to a 4-point polygon in the frame."""
    x, y, w, h = box
    rect = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    return np.int32(np.round(cv2.perspectiveTransform(rect, M_inv).reshape(-1, 2)))


@dataclass
class Key:
    note: int                        # MIDI note number
    is_black: bool
    box: tuple[int, int, int, int]   # (x, y, w, h) in ROI coordinates
    poly: np.ndarray                 # the same box as 4 points in frame coordinates

    @property
    def name(self) -> str:
        return note_name(self.note)


class Keyboard:
    def __init__(self, corners, M, M_inv, white_boxes, black_boxes):
        self.corners = np.int32(np.round(corners))  # the quad, in frame coordinates
        self._M = M                                  # frame -> ROI

        # Sorting all keys by their center, left to right, gives the piano order:
        # a black key's center lies between the centers of its two white neighbours (C < C# < D)
        boxes = [(box, False) for box in white_boxes] + [(box, True) for box in black_boxes]
        boxes.sort(key=lambda item: item[0][0] + item[0][2] / 2)

        # Which note does the leftmost key have? Slide the octave's white/black pattern
        # along the detected keys until it fits.
        offset = self._first_pitch_class([is_black for _, is_black in boxes])
        # If nothing fits, some keys were missed: the notes will be off
        self.pattern_ok = offset is not None
        first_note = BASE_OCTAVE_NOTE + (offset or 0)

        self.keys = [
            Key(first_note + i, is_black, box, roi_box_to_frame(box, M_inv))
            for i, (box, is_black) in enumerate(boxes)
        ]
        # Black keys lie on top of the white ones (the white boxes run under them) -> test them first
        self._hit_order = sorted(self.keys, key=lambda key: not key.is_black)

    @property
    def white_count(self) -> int:
        return sum(not key.is_black for key in self.keys)

    @property
    def black_count(self) -> int:
        return sum(key.is_black for key in self.keys)

    def image_polygons(self) -> dict[int, np.ndarray]:
        """note -> polygon in frame coordinates (what overlay.draw_keyboard expects)."""
        return {key.note: key.poly for key in self.keys}

    def key_at(self, x: float, y: float) -> Key | None:
        """The key under a point in the frame, or None."""
        # Go back to the flat ROI, where every key is a plain rectangle
        px, py = cv2.perspectiveTransform(np.float32([[[x, y]]]), self._M)[0, 0]
        for key in self._hit_order:
            bx, by, bw, bh = key.box
            if bx <= px < bx + bw and by <= py < by + bh:
                return key
        return None

    @staticmethod
    def _first_pitch_class(is_black: list[bool]) -> int | None:
        """Pitch class (0 = C ... 11 = B) of the leftmost key, or None if the pattern fits nowhere."""
        if not is_black:
            return None
        for offset in range(12):
            if all(OCTAVE_IS_BLACK[(offset + i) % 12] == black for i, black in enumerate(is_black)):
                return offset
        return None
