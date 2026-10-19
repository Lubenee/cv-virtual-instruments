"""Everything that draws on the video frame."""

import cv2
import numpy as np

from constants import *
from hand_detection import FINGERTIP_LANDMARKS, HAND_CONNECTIONS

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_landmarks(frame, result, draw_hand_skeleton=False, draw_fingertips_only=True, draw_fingertip_coords=True):
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
            cv2.putText(frame, f"({index_x}, {index_y})",
                (index_x + 10, index_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )
        # TODO: 
        # Print the note under each fingertip on screen

def draw_keyboard(frame, keyboard, hovered_notes=(), sounding_notes=()):
    """Outline every key; tint hovered keys, fill sounding keys."""
    if keyboard is None:
        return
    polys = keyboard.image_polygons()
    hovered_notes, sounding_notes = set(hovered_notes), set(sounding_notes)

    overlay = frame.copy()
    untouched_black = np.zeros(frame.shape[:2], np.uint8)
    # White keys first, then black keys on top (same order as on a real piano).
    for key in sorted(keyboard.keys, key=lambda k: k.is_black):
        poly = polys[key.note]
        if key.note in sounding_notes:
            cv2.fillPoly(overlay, [poly], (0, 140, 255))
        elif key.note in hovered_notes:
            cv2.fillPoly(overlay, [poly], (255, 200, 0))
        elif key.is_black:
            cv2.fillPoly(untouched_black, [poly], 255)
    # A tinted white key must not paint over the black keys lying on top of it.
    overlay[untouched_black > 0] = frame[untouched_black > 0]
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)

    for key in keyboard.keys:
        color = (60, 60, 60) if key.is_black else (200, 200, 200)
        cv2.polylines(frame, [polys[key.note]], True, color, 1, cv2.LINE_AA)
    cv2.polylines(frame, [np.int32(keyboard.corners)], True, (0, 255, 0), 2, cv2.LINE_AA)

def draw_text_lines(frame, lines, origin=(10, 28), scale=.7, color=BGR_WILLOW):
    x, y = origin
    for line in lines:
        if not line:
            y += int(28 * scale / 0.6)
            continue
        (w, h), _ = cv2.getTextSize(line, FONT, scale, 2)
        cv2.rectangle(frame, (x - 4, y - h - 6), (x + w + 4, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, line, (x, y), FONT, scale, color, 2, cv2.LINE_AA)
        y += int(28 * scale / 0.6)

def draw_big_label(frame, text, color=(0, 200, 255)):
    """Large label in the top-right corner (current chord)."""
    if not text:
        return
    scale, thick = 1.6, 3
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = frame.shape[1] - w - 20, h + 20
    cv2.rectangle(frame, (x - 10, y - h - 10), (x + w + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def draw_rec_indicator(frame, elapsed):
    h = frame.shape[0]
    cv2.circle(frame, (24, h - 26), 10, (0, 0, 255), -1)
    cv2.putText(frame, f"REC {elapsed:5.1f}s", (42, h - 19), FONT, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA)

def draw_calibration_points(frame, points):
    for i, (x, y) in enumerate(points):
        cv2.circle(frame, (int(x), int(y)), 7, (0, 0, 255), -1)
        cv2.putText(frame, str(i + 1), (int(x) + 10, int(y) - 10), FONT, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
    if len(points) > 1:
        cv2.polylines(frame, [np.int32(points)], len(points) == 4, (0, 0, 255), 2)
