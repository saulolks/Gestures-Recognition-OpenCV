import cv2
import imutils
import numpy as np

import camera_module as cm

# video_capture = cv2.VideoCapture(0)
# # Check success
# if not video_capture.isOpened():
#     raise Exception("Could not open video device")
# # Read picture. ret === True on success
# while True:
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("", gray)
#
# # Close device
# video_capture.release()

handcascade = cv2.CascadeClassifier('./haarcascades/palm_v4.xml')
headcascade = cv2.CascadeClassifier('./haarcascades/face.xml')
backSub = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((1, 1), np.uint8)

def camera_module(image):
    # ----------------------- remove face -------------------------------
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = headcascade.detectMultiScale(image_gray, 1.3, 5)

    # -------------------- background substraction -----------------------

    # mask = backSub.apply(img_ycrcb)

    # ------------------- splitting into 3 channels ----------------------

    y, cr, cb = cv2.split(image)

    # ------------------------- binarization -----------------------------
    y = cv2.adaptiveThreshold(y, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    cr = cv2.adaptiveThreshold(cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    cb = cv2.adaptiveThreshold(cb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)

    # ------------------------- morphology -------------------------------
    y = cv2.morphologyEx(y, cv2.MORPH_OPEN, kernel)
    cr = cv2.morphologyEx(cr, cv2.MORPH_OPEN, kernel)
    cb = cv2.morphologyEx(cb, cv2.MORPH_OPEN, kernel)

    # ---------------------------  merge ---------------------------------
    img = y + cr + cb
    mask = backSub.apply(img)                                       # background substraction
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)           # opening operator
    img = cv2.bitwise_and(image, image, mask=mask)                  # and operator

    imutils.resize(img, 600)

    # ------------------------- removing face ---------------------------
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # -------------------------- detect hands ---------------------------
    hands = handcascade.detectMultiScale(img, 1.1, 1)
    for (x, y, w, h) in hands:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("", img)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()


cap = cv2.VideoCapture('./videos/libras.mp4')
ret = True

while ret:

    ret, frame = cap.read()

    frame = imutils.resize(frame, 600)
    camera_module(frame)