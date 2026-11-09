"""Which keys of the locked keyboard have a fingertip inside them."""
from dataclasses import dataclass

from hand_detection import FINGERTIPS
from keyboard import Key, Keyboard


@dataclass
class FingerOnKey:
    hand: str # "Left" / "Right", as MediaPipe reports it
    finger: str # "thumb", "index", ...
    point: tuple[int, int] # fingertip, in frame coordinates
    key: Key


class FingerKeyDetector:
    def __init__(self, index_only: bool = False):
        # True gives only the index fingertips count, falsegives every fingertip
        self.index_only = index_only

    def toggle_index_only(self):
        self.index_only = not self.index_only

    @property
    def fingers(self) -> tuple[str, ...]:
        return ("index",) if self.index_only else tuple(FINGERTIPS)

    def detect(self, result, frame_shape, keyboard: Keyboard | None) -> list[FingerOnKey]:
        # Every (hand, finger) whose tip is inside key
        if keyboard is None or result is None:
            return []
        height, width = frame_shape[:2]

        touches = []
        for i, hand in enumerate(result.hand_landmarks):
            handedness = result.handedness[i] if i < len(result.handedness) else None
            hand_label = handedness[0].category_name if handedness else f"hand {i + 1}"
            for finger in self.fingers:
                tip = hand[FINGERTIPS[finger]]
                x, y = tip.x * width, tip.y * height
                key = keyboard.key_at(x, y)
                if key is not None:
                    touches.append(FingerOnKey(hand_label, finger, (int(x), int(y)), key))
        return touches
