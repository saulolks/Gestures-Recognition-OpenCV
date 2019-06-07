import cv2
import imutils
import numpy as np

shape = None
back_photos = []
background = None
handcascade = cv2.CascadeClassifier('./haarcascades/palm_v4.xml')
headcascade = cv2.CascadeClassifier('./haarcascades/face.xml')
backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=12)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
lineType = 4


def nothing(pos):
    pass


def get_background(img, num_frames=20):
    global background
    if len(back_photos) < num_frames:
        back_photos.append(img)
        return backSub.apply(img)
    elif len(back_photos) == num_frames:
        for item in back_photos:
            background = backSub.apply(item)
        return background
    else:
        return background


def remove_concomponent(img, min_value):
    img2 = np.zeros(shape)

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    for i in range(0, nb_components):
        if sizes[i] >= min_value:
            img2[output == i + 1] = 255

    return img2.astype(np.uint8)


def define_mask(img):
    pass


def take_a_pic():
    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()

        frame = imutils.resize(frame, 600)
        cv2.imwrite("./image/background.jpeg", frame)
        break


def camera_module(original, background):
    ymin = cv2.getTrackbarPos('Ymin', 'YRB_calib')
    ymax = cv2.getTrackbarPos('Ymax', 'YRB_calib')
    crmin = cv2.getTrackbarPos('CRmin', 'YRB_calib')
    crmax = cv2.getTrackbarPos('CRmax', 'YRB_calib')
    cbmin = cv2.getTrackbarPos('CBmin', 'YRB_calib')
    cbmax = cv2.getTrackbarPos('CBmax', 'YRB_calib')
    wsize_open = cv2.getTrackbarPos('OpenSize', 'Windows sizes')
    wsize_gaussian = cv2.getTrackbarPos('Gaussian', 'Windows sizes') + 1
    min_value = cv2.getTrackbarPos('Connected', 'Windows sizes')

    kernel = np.ones((wsize_open, wsize_open), np.uint8)
    print(ymin, ymax, crmin, crmax, cbmin, cbmax)

    # ----------------------- rgb to ycrcb -------------------------------
    image = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)

    # ------------------------ remove face -------------------------------
    image_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    faces = headcascade.detectMultiScale(image_gray, 1.3, 5)

    # -------------------------- background ------------------------------
    # mask = get_background(image)
    # print(mask.shape)

    # ------------------- splitting into 3 channels ----------------------
    y, cr, cb = cv2.split(image)

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

    mask = backSub.apply(image)
    img = cv2.bitwise_and(mask, img)

    kernel_gaussian = np.ones((wsize_gaussian, wsize_gaussian), np.float32) / (wsize_gaussian * wsize_gaussian)
    # img = cv2.filter2D(img, -1, kernel_gaussian)
    img = cv2.medianBlur(img, 7)
    img = remove_concomponent(img=img, min_value=min_value)


    # ------------------------- removing face ---------------------------
    marge = 10
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x- marge, y - h + marge), (x + marge + w, y + marge + h), (0, 0, 0), -1)

    # -------------------------- detect hands ---------------------------
    hands = handcascade.detectMultiScale(img, 1.1, 1)
    for (x, y, w, h) in hands:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img


def detection_module(original, binary):
    binary_aux = binary
    aux = np.array(shape)

    contours, _ = cv2.findContours(binary, 1, 2)

    if len(contours) > 0:
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.putText(original, 'mao detectada', (approx[0][0][0], approx[0][0][1] + 10), font, fontScale, fontColor, lineType)

        cv2.drawContours(original, [approx], -1, (0, 255, 0), 2)



    return binary_aux


take_a_pic()

# cap = cv2.VideoCapture('./videos/libras.mp4')
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
cv2.createTrackbar('OpenSize', 'Windows sizes', 3, 10, nothing)
cv2.createTrackbar('Gaussian', 'Windows sizes', 15, 30, nothing)
cv2.createTrackbar('Connected', 'Windows sizes', 150, 500, nothing)

while ret:
    ret, frame = cap.read()

    frame = imutils.resize(frame, 600)
    shape = (len(frame), len(frame[0]))

    bin_image = camera_module(frame, background)
    detect = detection_module(frame, bin_image)

    cv2.imshow("binary", bin_image)
    cv2.imshow("original", frame)
    cv2.imshow("detection", detect)
    cv2.waitKey(1)
