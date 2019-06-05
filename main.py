import cv2
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
def camera_module(image):
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    shape = (len(image), len(image[0]))

    print(len(image), len(image[0]))
    # -------------------- background substraction -----------------------

    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbmask = fgbg.apply(img_ycrcb)

    # ------------------- splitting into 3 channels ----------------------

    y, cr, cb = cv2.split(img_ycrcb)

    # ------------------------- binarization -----------------------------
    y = cv2.adaptiveThreshold(y, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    cr = cv2.adaptiveThreshold(cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    cb = cv2.adaptiveThreshold(cb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)

    # ---------------------------  merge ---------------------------------
    img = cv2.merge([y, cr, cb])

    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


image = cv2.imread("lenna.png", cv2.IMREAD_COLOR)
camera_module(image)