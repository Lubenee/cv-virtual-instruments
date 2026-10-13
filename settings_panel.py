import cv2

class SettingsPanel:
    LABELS = ("Detection confidence", "Presence confidence", "Tracking confidence")
    PANEL_COLOR = (25, 32, 36)
    ACCENT_COLOR = (88, 196, 165)
    MUTED_COLOR = (145, 161, 163)
    TEXT_COLOR = (236, 241, 237)

    def __init__(self, landmarker):
        self.landmarker = landmarker
        self.values = [50, 50, 50]
        self.applied_values = tuple(self.values)
        self.visible = True
        self.dragging_slider = None
        self.panel_bounds = (0, 0, 0, 0)
        self.header_bounds = (0, 0, 0, 0)
        self.apply_bounds = (0, 0, 0, 0)
        self.slider_bounds = []

    def draw(self, frame):
        _, frame_width = frame.shape[:2]
        panel_width = min(304, frame_width - 24)
        panel_height = 244 if self.visible else 44
        left = max(12, frame_width - panel_width - 12)
        top = 14
        right = left + panel_width
        bottom = top + panel_height
        self.panel_bounds = (left, top, right, bottom)
        self.header_bounds = (left, top, right, top + 44)

        region = frame[top:bottom, left:right]
        overlay = region.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), self.PANEL_COLOR, -1)
        cv2.addWeighted(overlay, 0.9, region, 0.1, 0, region)
        cv2.rectangle(frame, (left, top), (right, top + 3), self.ACCENT_COLOR, -1)
        cv2.putText(frame, "HAND TRACKING", (left + 16, top + 27),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, self.TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, "-" if self.visible else "+", (right - 30, top + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, self.ACCENT_COLOR, 2, cv2.LINE_AA)

        self.slider_bounds = []
        self.apply_bounds = (0, 0, 0, 0)
        if not self.visible:
            return

        for index, label in enumerate(self.LABELS):
            row_top = top + 57 + index * 43
            slider_left = left + 18
            slider_right = right - 52
            slider_y = row_top + 28
            self.slider_bounds.append((slider_left, slider_y, slider_right, index))
            cv2.putText(frame, label, (left + 18, row_top + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, self.TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{self.values[index]}%", (right - 46, row_top + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, self.MUTED_COLOR, 1, cv2.LINE_AA)
            cv2.line(frame, (slider_left, slider_y), (slider_right, slider_y),
                     (75, 87, 88), 4, cv2.LINE_AA)
            value_x = slider_left + round((slider_right - slider_left) * self.values[index] / 100)
            cv2.line(frame, (slider_left, slider_y), (value_x, slider_y),
                     self.ACCENT_COLOR, 4, cv2.LINE_AA)
            cv2.circle(frame, (value_x, slider_y), 7, self.TEXT_COLOR, -1, cv2.LINE_AA)

        button_left = left + 18
        button_top = top + 198
        button_right = left + 142
        button_bottom = top + 230
        self.apply_bounds = (button_left, button_top, button_right, button_bottom)
        cv2.rectangle(frame, (button_left, button_top), (button_right, button_bottom),
                      self.ACCENT_COLOR, -1)
        cv2.putText(frame, "APPLY SETTINGS", (button_left + 10, button_top + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.PANEL_COLOR, 2, cv2.LINE_AA)
        status = "APPLIED" if tuple(self.values) == self.applied_values else "PENDING"
        status_color = self.MUTED_COLOR if status == "APPLIED" else self.ACCENT_COLOR
        cv2.putText(frame, status, (button_right + 14, button_top + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

    def on_mouse(self, event, x, y, flags, _parameter):
        if event == cv2.EVENT_LBUTTONUP:
            self.dragging_slider = None
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._contains(self.header_bounds, x, y):
                self.visible = not self.visible
                return
            if self.visible and self._contains(self.apply_bounds, x, y):
                self.landmarker.update_settings(value / 100 for value in self.values)
                self.applied_values = tuple(self.values)
                return
            for slider_left, slider_y, slider_right, index in self.slider_bounds:
                if slider_y - 12 <= y <= slider_y + 12 and slider_left <= x <= slider_right:
                    self.dragging_slider = index
                    self._set_slider_value(index, x, slider_left, slider_right)
                    return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging_slider is not None:
            slider_left, _, slider_right, _ = self.slider_bounds[self.dragging_slider]
            self._set_slider_value(self.dragging_slider, x, slider_left, slider_right)

    def _set_slider_value(self, index, x, slider_left, slider_right):
        position = max(0, min(slider_right - slider_left, x - slider_left))
        self.values[index] = round(position * 100 / (slider_right - slider_left))

    @staticmethod
    def _contains(bounds, x, y):
        left, top, right, bottom = bounds
        return left <= x <= right and top <= y <= bottom
