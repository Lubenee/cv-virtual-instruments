import cv2
from constants import *
import numpy as np

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
        self.selected_corners = []
        
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



