import pickle
from glob import glob
from skimage.feature import hog
import cv2
import imutils
import numpy as np
import os.path
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier


shape = None
back_photos = []
background = None
handcascade = cv2.CascadeClassifier('./haarcascades/palm_v4.xml')
headcascade = cv2.CascadeClassifier('./haarcascades/face.xml')
backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=30)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)
lineType = 4
hand_image = np.zeros((80, 80), np.uint8)
data = None


def preprocessing():
    data = []
    for file in glob('./images/**'):
        file_string = str(file)
        label = ""
        print(file_string)

        for i in range(9, len(file_string)):
            if file_string[i].isnumeric():
                label = file_string[9:i]
                break
        image = cv2.imread(file_string)
        image = imutils.resize(image, 300)
        binary = camera_module(image)
        _, hand, _ = detection_module(image, binary)
        back = np.zeros((80, 80), np.uint8)
        hand = cv2.resize(hand, (80, 80), interpolation=cv2.INTER_NEAREST)
        back[:hand.shape[0], :hand.shape[1]] = hand
        # cv2.imshow("", back)
        # cv2.waitKey()
        features = get_feature_vector(back)
        data.append((label, features))
        print(1)

    with open('images/test_features', 'wb') as fp:
        pickle.dump(data, fp)


def get_feature_vector(img):
    result = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(10, 10),
                 feature_vector=True)
    return result


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
    shape = (len(img), len(img[0]))
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


def camera_module(original):
    ymin = cv2.getTrackbarPos('Ymin', 'YRB_calib')
    ymax = cv2.getTrackbarPos('Ymax', 'YRB_calib')
    crmin = cv2.getTrackbarPos('CRmin', 'YRB_calib')
    crmax = cv2.getTrackbarPos('CRmax', 'YRB_calib')
    cbmin = cv2.getTrackbarPos('CBmin', 'YRB_calib')
    cbmax = cv2.getTrackbarPos('CBmax', 'YRB_calib')
    wsize_open = cv2.getTrackbarPos('OpenSize', 'Windows sizes')
    wsize_gaussian = cv2.getTrackbarPos('Gaussian', 'Windows sizes') + 1
    wsize_gaussian = wsize_gaussian if wsize_gaussian % 2 == 1 else wsize_gaussian + 1
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
    img = cv2.medianBlur(img, wsize_gaussian)
    img = remove_concomponent(img=img, min_value=min_value)

    # ------------------------- removing face ---------------------------
    marge = 30
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x - marge, y - h + marge), (x + marge + w, y + marge + h), (0, 0, 0), -1)

    # -------------------------- detect hands ---------------------------
    # hands = handcascade.detectMultiScale(img, 1.1, 1)
    # for (x, y, w, h) in hands:
    #     # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img


def detection_module(original, binary):
    hand_only = None
    binary_aux = binary
    position = None

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
        hull = cv2.convexHull(approx)
        M = cv2.moments(approx)
        x, y, w, h = cv2.boundingRect(hull)
        position = (x, y, w, h)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(original, (cX, cY), 5, (255, 255, 255), -1)
            # cv2.putText(original, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)
        hand_only = binary_aux[y:y + h, x:x + w]
        # cv2.putText(original, 'mao detectada', (approx[0][0][0], approx[0][0][1] + 10), font, fontScale, fontColor,
        #             lineType)
        cv2.drawContours(original, [approx], -1, (0, 255, 0), 2)
        cv2.drawContours(original, [hull], -1, (0, 0, 255), 2)

    return binary_aux, hand_only, position


def classification_module(frame, hand_image, position):
    feature = get_feature_vector(hand_image)

    label = ""
    minim = np.inf

    for item in data:
        dist = distance.euclidean(feature, item[1])
        if dist < minim:
            minim = dist
            label = item[0]
    if position is not None and minim < 1.1:
        print("distance: ", minim)
        cv2.putText(frame, str(label), (position[0], position[1] + 10), font, fontScale, fontColor, lineType)


def main():
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
    cv2.createTrackbar('Gaussian', 'Windows sizes', 5, 30, nothing)
    cv2.createTrackbar('Connected', 'Windows sizes', 150, 500, nothing)

    if not os.path.exists('images/test_features'):
        preprocessing()

    with open('images/test_features', 'rb') as fp:
        global data
        data = pickle.load(fp)

    while ret:
        ret, frame = cap.read()

        frame = imutils.resize(frame, 600)
        shape = (len(frame), len(frame[0]))

        bin_image = camera_module(frame)
        detect, hand, position = detection_module(frame, bin_image)

        if hand is not None:
            hand = cv2.resize(hand, (80, 80), interpolation=cv2.INTER_NEAREST)
            hand_image[:hand.shape[0], :hand.shape[1]] = hand

        classification_module(frame, hand_image, position)

        cv2.imshow("detection", detect)
        cv2.imshow("original", frame)
        cv2.imshow("hand", hand_image)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
