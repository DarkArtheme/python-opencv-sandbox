import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)


def erosion(img):

    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # invert the image
    invert = cv2.bitwise_not(binr)

    # erode the image
    erosion_img = cv2.erode(invert, kernel, iterations=1)

    return erosion_img


def dilation(img):

    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # invert the image
    invert = cv2.bitwise_not(binr)

    # dilate the image
    dilation_img = cv2.dilate(invert, kernel, iterations=1)

    return dilation_img


def opening(img):
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # opening the image
    opening_img = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1)

    return opening_img


def closing(img):
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # opening the image
    closing_img = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing_img
