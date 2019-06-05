import cv2
import numpy as np


def background_substraction(img):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbmask = fgbg.apply(img)
    return fgbmask


def split_channels(image):
    return cv2.split(image)

def binarization(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def morphology(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def add(channel1, channel2, channel3):
    return cv2.merge([channel1, channel2, channel3])
