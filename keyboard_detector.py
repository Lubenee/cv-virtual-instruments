import cv2
from constants import *
import numpy as np

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
    # cv2.imshow("thresh", thresh)
    
    ret, thresh_inv = cv2.threshold(gaussian, threshold, max, cv2.THRESH_BINARY_INV)
    # cv2.imshow("thresh_inv", thresh_inv)
    
    # otsu algorithm applies it's own threshold, so this 0 is actually ignored.
    otsu_thresh_val, thresh_otsu = cv2.threshold(gaussian, 0, max, cv2.THRESH_OTSU)
    otsu_thresh_val, thresh_otsu_inv = cv2.threshold(gaussian, 0, max, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    print("Otsu thresh val: ", otsu_thresh_val)
    # cv2.imshow("thresh_otsu", thresh_otsu)
    
    ##############################
    ## Morphological transforms ##
    ##############################
    # https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html?utm_source=chatgpt.com
    kernel = np.ones((6,6),np.uint8)
    opening_org = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel)
    opening_inv = cv2.morphologyEx(thresh_otsu_inv, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening_org', opening1)
    cv2.imshow('opening_inv', opening_inv)
    
    ############################
    ##### Image Contours #######
    ###########################
    # https://docs.opencv.org/4.13.0/d4/d73/tutorial_py_contours_begin.html?utm_source=chatgpt.com
    
    # RETR_EXTERNAL	Retrieves only outermost contours
    # RETR_LIST	Retrieves all contours without preserving their hierarchy
    # RETR_TREE	Retrieves all contours and builds a parent-child hierarchy
    contours1, _ = cv2.findContours(opening_org, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(opening_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print("Contours found: ", len(contours1))
    index = -1
    thickness = 1
    # cv2.drawContours(img, contours1, index, BGR_GREEN, thickness)
    # cv2.drawContours(img, contours2, index, BGR_GREEN, thickness)
    
    img1 = img.copy()
    img2 = img.copy()
    for cnt in contours1:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img1,(x,y),(x+w,y+h),BGR_RED,thickness)
        # cv2.rectangle(thresh_inv,(x,y),(x+w,y+h),BGR_RED,thickness)
        
    for cnt in contours2:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img2,(x,y),(x+w,y+h),BGR_BLUE,thickness)
        # cv2.rectangle(thresh,(x,y),(x+w,y+h),BGR_RED,thickness)
    
    # cv2.imshow("thresh", thresh_otsu)
    cv2.imshow("thresh_inv", thresh_otsu_inv)
    
    # cv2.imshow("img_thresh", img1)
    cv2.imshow("img_thresh_inv", img2)
    
    # cv2.imshow("Original", img)
    cv2.imshow
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

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



