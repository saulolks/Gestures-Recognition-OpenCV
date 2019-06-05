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

image = cv2.imread("lenna.png", cv2.IMREAD_COLOR)
img_result = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
shape = (len(image), len(image[0]))

print(len(image), len(image[0]))

y, cr, cb = cm.split_channels(img_result)


print(len(y), len(y[0]))
print(len(cr), len(cr[0]))
print(len(cb), len(cb[0]))

y = cm.binarization(y)
cr = cm.binarization(cr)
cb = cm.binarization(cb)

img = cm.add(y, cr, cb)

cv2.imshow("", img)
cv2.waitKey()
cv2.destroyAllWindows()