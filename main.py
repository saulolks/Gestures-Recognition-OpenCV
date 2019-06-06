import cv2
import imutils
import numpy as np

ymin = 0
ymax = 0
crmin = 0
crmax = 0
cbmin = 0
cbmax = 0


def nothing(pos):
    pass


def change_ymin(pos):
    ymin = pos


def change_ymax(pos):
    ymax = pos


def change_cbmin(pos):
    cbmin = pos


def change_cbmax(pos):
    cbmax = pos


def change_crmin(pos):
    crmin = pos


def change_crmax(pos):
    crmax = pos


def take_a_pic():
    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()

        frame = imutils.resize(frame, 600)
        cv2.imwrite("./image/background.jpeg", frame)
        break


handcascade = cv2.CascadeClassifier('./haarcascades/palm_v4.xml')
headcascade = cv2.CascadeClassifier('./haarcascades/face.xml')
backSub = cv2.createBackgroundSubtractorMOG2()


def camera_module(original, background):
    ymin = cv2.getTrackbarPos('Ymin', 'YRB_calib')
    ymax = cv2.getTrackbarPos('Ymax', 'YRB_calib')
    crmin = cv2.getTrackbarPos('CRmin', 'YRB_calib')
    crmax = cv2.getTrackbarPos('CRmax', 'YRB_calib')
    cbmin = cv2.getTrackbarPos('CBmin', 'YRB_calib')
    cbmax = cv2.getTrackbarPos('CBmax', 'YRB_calib')
    wsize_open = cv2.getTrackbarPos('OpenSize', 'Windows sizes')
    wsize_gaussian = cv2.getTrackbarPos('Gaussian', 'Windows sizes')

    kernel = np.ones((wsize_open, wsize_open), np.uint8)
    print(ymin, ymax, crmin, crmax, cbmin, cbmax)

    # ----------------------- remove face -------------------------------
    image = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)
    # background = cv2.cvtColor(background, cv2.COLOR_BGR2YCR_CB)

    image_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    faces = headcascade.detectMultiScale(image_gray, 1.3, 5)

    no_back = image  # original - background
    mask = backSub.apply(original)
    # ------------------- splitting into 3 channels ----------------------
    y, cr, cb = cv2.split(no_back)

    # ------------------------- binarization -----------------------------
    _, y = cv2.threshold(y, ymin, ymax, cv2.THRESH_BINARY)
    _, cr = cv2.threshold(cr, crmin, crmax, cv2.THRESH_BINARY)
    _, cb = cv2.threshold(cb, cbmin, cbmax, cv2.THRESH_BINARY)

    # ------------------------- morphology -------------------------------
    y = cv2.morphologyEx(y, cv2.MORPH_OPEN, kernel)
    cr = cv2.morphologyEx(cr, cv2.MORPH_OPEN, kernel)
    cb = cv2.morphologyEx(cb, cv2.MORPH_OPEN, kernel)

    # ---------------------------  merge ---------------------------------
    img = y + cr + cb
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # y, cr, cb = cv2.split(original)
    # y *= img
    # cr *= img
    # cb *= img
    # img = cv2.merge([y, cr, cb])
    # img = cv2.bitwise_and(mask, img)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    kernel_gaussian = np.ones((wsize_gaussian, wsize_gaussian), np.float32) / (wsize_gaussian * wsize_gaussian)
    img = cv2.filter2D(img, -1, kernel_gaussian)
    # ------------------------- removing face ---------------------------
    for (x, y, w, h) in faces:
        h += 20
        cv2.rectangle(img, (x, y - h), (x + w, y + h), (0, 0, 0), -1)

    # -------------------------- detect hands ---------------------------
    hands = handcascade.detectMultiScale(img, 1.1, 1)
    for (x, y, w, h) in hands:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("", img)
    cv2.waitKey(1)

    return img


take_a_pic()
background = cv2.imread("./images/background.jpeg", cv2.IMREAD_COLOR)

cap = cv2.VideoCapture(0)
ret = True

cv2.namedWindow('YRB_calib')
cv2.createTrackbar('Ymin', 'YRB_calib', 54, 255, nothing)
cv2.createTrackbar('Ymax', 'YRB_calib', 137, 255, nothing)
cv2.createTrackbar('CRmin', 'YRB_calib', 135, 255, nothing)
cv2.createTrackbar('CRmax', 'YRB_calib', 174, 255, nothing)
cv2.createTrackbar('CBmin', 'YRB_calib', 0, 255, nothing)
cv2.createTrackbar('CBmax', 'YRB_calib', 125, 255, nothing)

cv2.namedWindow('Windows sizes')
cv2.createTrackbar('OpenSize', 'Windows sizes', 1, 10, nothing)
cv2.createTrackbar('Gaussian', 'Windows sizes', 5, 15, nothing)

while ret:
    ret, frame = cap.read()

    frame = imutils.resize(frame, 600)
    bin_image = camera_module(frame, background)
