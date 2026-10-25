import cv2
from constants import *
import numpy as np
from cv2.typing import MatLike

#! TODO: Use 4 points from user input as a starting quad
#! TODO: Also add automatic detection, try finding a few quad candidates
def detector():
    img = cv2.imread("synth.jpeg")

    if img is None:
        raise FileNotFoundError("Could not load synth.jpeg")
    
    ################################
    ####### Image processing #######
    ################################

    # turn to grayscale
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # cv.blur() applies a mean filter. It replaces each pixel 
    # with the average value of the pixels surrounding it, including itself
    # using a Kernel. https://docs.opencv.org/4.13.0/d4/d13/tutorial_py_filtering.html
    
    # Gaussian blur means that pixels closer to the center contribute more than pixels farther away.
    # Effective for removing gaussian noise
    Kernel = (5,5)
    gaussian = cv2.GaussianBlur(imgray, Kernel, 0)
    
    # Thresholding
    # For every pixel, the same threshold value is applied. 
    # If the pixel value is smaller than the threshold, it is set to 0, 
    # otherwise it is set to a maximum value.
    max = 255
    threshold = 240
    ret, thresh = cv2.threshold(gaussian, threshold,max,cv2.THRESH_BINARY)
    ret, thresh_inv = cv2.threshold(gaussian, threshold, max, cv2.THRESH_BINARY_INV)
    
    # otsu algorithm applies it's own threshold, so this 0 is actually ignored.
    otsu_thresh_val, thresh_otsu = cv2.threshold(gaussian, 0, max, cv2.THRESH_OTSU)
    otsu_thresh_val, thresh_otsu_inv = cv2.threshold(gaussian, 0, max, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    ######################################
    ### Get a ROI - region of interest ###
    ######################################
    
    # Select a horizontal strip containing the black keys
    H, W = thresh_otsu_inv.shape # image width and height
    y1 = int(H * 0.40)
    y2 = int(H * 0.55)
    x1 = int (W * 0.15)
    x2 = int (W * 0.9)
    roi = thresh_otsu_inv[y1:y2, x1:x2].copy()
    
    
    ##############################
    ## Morphological transforms ##
    ##############################
    # https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html?utm_source=chatgpt.com
    kernel = np.ones((3,3),np.uint8)
    opening_inv = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opening_inv', opening_inv)
    
    ############################
    ##### Image Contours #######
    ###########################
    # https://docs.opencv.org/4.13.0/d4/d73/tutorial_py_contours_begin.html?utm_source=chatgpt.com
    
    # RETR_EXTERNAL	Retrieves only outermost contours
    # RETR_LIST	Retrieves all contours without preserving their hierarchy
    # RETR_TREE	Retrieves all contours and builds a parent-child hierarchy
    contours, _ = cv2.findContours(opening_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Contours found: ", len(contours))
    # cv2.drawContours(img, contours2, index, BGR_GREEN, thickness)
    
    result = img.copy()
    detected = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Reject short objects
        if h < 0.35 * roi.shape[0]:
            continue
        # Reject objects that aren't tall and narrow
        if h < 1.3 * w:
            continue
        # Reject objects that don't fill their bounding rectangle
        if area < 0.5 * w * h:
            continue
        # Convert ROI coordinates back to original image coordinates
        cv2.rectangle(
            result,
            (x + x1, y + y1),
            (x + x1 + w, y + y1 + h),
            (0, 255, 0),
            2
        )

        detected += 1
    
    print("Potential black keys:", detected)
    # cv2.imshow("thresh", thresh_otsu)
    cv2.imshow("thresh_inv", thresh_otsu_inv)
    
    # cv2.imshow("img_thresh", img1)
    cv2.imshow("ROI Mask", opening_inv)
    cv2.imshow("Detected Black Keys", result)
    
    # cv2.imshow("Original", img)
    cv2.imshow
    cv2.waitKey(0)
    cv2.destroyAllWindows()
class KeyboardDetector:
    def __init__(self):
        self._manual_calibration = False
        self.selected_corners: list[tuple[float, float]] = []
        # Black keys found in the last frame, as (x, y, w, h) boxes in ROI coordinates
        self.black_keys: list[tuple[int, int, int, int]] = []
        # The perspective transform only depends on the corners, so we cache it
        self._roi_cache_key = None
        self._roi_corners = None     # ordered corners: TL, TR, BR, BL
        self._roi_size = None        # (width, height) of the flattened ROI
        self._roi_M = None           # frame -> ROI
        self._roi_M_inv = None       # ROI -> frame
        self._roi_window_open = False
        
    @property
    def manual_calibration(self):
        return self._manual_calibration

    def set_manual_calibration(self, value):
        if value:
            self.selected_corners = []
            self._manual_calibration = True
        else:
            self._manual_calibration = False
        
    def get_selected_quad(self):
        return self.selected_corners
        
    def on_mouse(self, event, x, y, flags, param):
        if not self.manual_calibration:
            return
        if (event == cv2.EVENT_LBUTTONDOWN):
            self.selected_corners.append((x,y))
            
            if (len(self.selected_corners) >= 4):
                self._manual_calibration = False
                
    def _ordered_corners(self) -> np.ndarray:
        """Return the 4 clicked corners as TL, TR, BR, BL - whatever order they were clicked in."""
        pts = np.array(self.selected_corners, dtype=np.float32)
        # Sort the points by their angle around the center -> clockwise order on screen
        # (clockwise because the image y axis points down)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        # Rotate the list so it starts at the top-left corner (smallest x + y)
        start = int(np.argmin(pts.sum(axis=1)))
        return np.roll(pts, -start, axis=0)

    def extract_roi(self, frame: MatLike):
        """
            Cut the selected quad out of the frame and flatten it ("bird's-eye view").
            https://stackoverflow.com/questions/75336140/generating-bird-eye-view-of-image-in-opencvpython-without-knowing-exact-positi
        """
        if len(self.selected_corners) != 4:
            raise ValueError("Exactly 4 corners must be selected.")

        # The transform only changes when the corners change -> compute it once
        cache_key = tuple(self.selected_corners)
        if cache_key != self._roi_cache_key:
            corners = self._ordered_corners()
            if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
                raise ValueError("The 4 corners don't form a convex shape.")
            tl, tr, br, bl = corners
            # Output size: the longest of each pair of opposite edges keeps the most detail
            width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
            height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
            if width < 20 or height < 20:
                raise ValueError("The selected area is too small.")
            dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            self._roi_M = cv2.getPerspectiveTransform(corners, dst)
            self._roi_M_inv = np.linalg.inv(self._roi_M)
            self._roi_corners = corners
            self._roi_size = (width, height)
            self._roi_cache_key = cache_key

        roi = cv2.warpPerspective(frame, self._roi_M, self._roi_size)
        return roi, self._roi_M_inv

    def _roi_box_to_frame(self, box, M_inv) -> np.ndarray:
        """Map an (x, y, w, h) box from ROI coordinates to a 4-point polygon in the frame."""
        x, y, w, h = box
        rect = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
        return np.int32(np.round(cv2.perspectiveTransform(rect, M_inv).reshape(-1, 2)))

    def _close_roi_window(self):
        if self._roi_window_open:
            cv2.destroyWindow(ROI_WINDOW)
            self._roi_window_open = False

    def draw_keyboard(self, frame: MatLike):
        """Detect the black keys inside the selected quad and draw them on the frame.

        Call this on a clean frame, before other overlays are drawn on it,
        otherwise text/markers inside the quad get detected as keys.
        """
        if len(self.selected_corners) != 4:
            self.black_keys = []
            self._close_roi_window()
            return
        # Cut out a flattened ROI from the user markings
        try:
            roi, M_inv = self.extract_roi(frame)
        except ValueError as err:
            self.black_keys = []
            self._close_roi_window()
            cv2.putText(frame, f"{err} Press 'c' to try again.", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR_RED, 2, cv2.LINE_AA)
            return
        H, W = roi.shape[:2]

        # turn it to grayscale
        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Apply a gaussian blur to remove some noise
        blur = cv2.GaussianBlur(imgray, (5, 5), 1)
        # Apply an otsu threshold & invert the image (black keys -> white).
        _, thresh_otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Clear a thin band along the ROI border. Clicked corners are never pixel-perfect,
        # and a sliver of dark case/table at the edge glues all black keys into one blob.
        mx, my = max(2, int(0.01 * W)), max(2, int(0.03 * H))
        thresh_otsu_inv[:my, :] = 0
        thresh_otsu_inv[H - my:, :] = 0
        thresh_otsu_inv[:, :mx] = 0
        thresh_otsu_inv[:, W - mx:] = 0

        # Opening removes thin lines (gaps between white keys, labels).
        # Scale the kernel with the rois's width so it works close to and far from the camera.
        k = max(3, W // 185)
        opening_inv = cv2.morphologyEx(thresh_otsu_inv, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

        # Find the contours, which should be exactly the black keys
        contours, _ = cv2.findContours(opening_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        black_keys = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # reject short objects
            if h < 0.35 * H:
                continue
            # reject wide objects
            if w > 0.15 * W:
                continue
            # Reject objects that aren't tall and narrow
            if h < 1.3 * w:
                continue
            # Reject objects that don't fill their bounding rectangle
            if area < 0.5 * w * h:
                continue
            # Black keys start at the back edge: give back the strip we cleared above
            if y <= my + 1:
                h += y
                y = 0
            black_keys.append((x, y, w, h))
        black_keys.sort(key=lambda key: key[0])  # left to right
        self.black_keys = black_keys

        ########################
        ####### Drawing ########
        ########################
        # ROI boxes -> polygons in the frame.
        key_polys = [self._roi_box_to_frame(b, M_inv) for b in black_keys]
        quad = np.int32(np.round(self._roi_corners))

        # draw outlines on top
        cv2.polylines(frame, [quad], True, BGR_WILLOW, 2, cv2.LINE_AA)
        cv2.polylines(frame, key_polys, True, BGR_GREEN, 2, cv2.LINE_AA)

        
#####################################################
# # Read video from file system/camera
# video = cv2.VideoCapture(0)
# ret = True
# while ret:
#     # Ret: Was frame read successfully?
#     # frame: The actual frame
#     ret, frame = video.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# video.release()
# cv2.destroyAllWindows()

#####################################################
# Drawing 
# img = cv2.imread('opencv-testing/board.jpg')

# cv2.line(img, (100, 150), (300, 450), (0, 255, 0), 3)
# cv2.rectangle(img, (200, 350), (450, 600), (0, 0, 255), -1)
# cv2.circle(img, (500, 550), 20, (210, 0 ,0), 2)
# cv2.putText(img, 'hi', (700, 450), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255,155,0), 2, )

# cv2.imshow('img', img)
# cv2.waitKey(0)



