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
kernel = np.ones((3, 3), np.uint8)

def take_a_pic():
    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()

        frame = imutils.resize(frame, 600)
        cv2.imwrite("./image/background.jpeg", frame)
        break

def camera_module(original, background):
    # ----------------------- remove face -------------------------------
    image = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2YCR_CB)

    # image = cv2.subtract(image, background)

    image_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    faces = headcascade.detectMultiScale(image_gray, 1.3, 5)

    # -------------------- background substraction -----------------------

    # mask = backSub.apply(img_ycrcb)

    # ------------------- splitting into 3 channels ----------------------
    y, cr, cb = cv2.split(image)
    y_original, cr_original, cb_original = cv2.split(image)

    # ------------------------- binarization -----------------------------
    _, y = cv2.threshold(y, 54, 163, cv2.THRESH_BINARY)
    _, cr = cv2.threshold(cr, 131, 157, cv2.THRESH_BINARY)
    _, cb = cv2.threshold(cb, 110, 135, cv2.THRESH_BINARY)

    # ------------------------- morphology -------------------------------
    y = cv2.morphologyEx(y, cv2.MORPH_OPEN, kernel)
    cr = cv2.morphologyEx(cr, cv2.MORPH_OPEN, kernel)
    cb = cv2.morphologyEx(cb, cv2.MORPH_OPEN, kernel)

    # ---------------------------  merge ---------------------------------
    y = cv2.bitwise_and(y, y_original)
    cr = cv2.bitwise_and(cr, cr_original)
    cb = cv2.bitwise_and(cb, cb_original)
    img = cv2.merge((y, cr, cb))
    img = cv2.Canny(img, 100, 200)

    # ------------------------- removing face ---------------------------
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # -------------------------- detect hands ---------------------------
    hands = handcascade.detectMultiScale(img, 1.1, 1)
    for (x, y, w, h) in hands:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("", img)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

take_a_pic()

# cap = cv2.VideoCapture('./videos/libras.mp4')
cap = cv2.VideoCapture(0)
ret = True

background = cv2.imread("./images/background.jpeg", cv2.IMREAD_COLOR)


while ret:
    ret, frame = cap.read()

    frame = imutils.resize(frame, 600)
    camera_module(frame, background)
