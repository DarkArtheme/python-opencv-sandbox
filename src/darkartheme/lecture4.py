import threading

import cv2

import numpy as np
from matplotlib import pyplot as plt
from tools import sign_image, Histogram


def binarize_image(image, thresh, maxval, type):
    ret, dst = cv2.threshold(image, thresh, maxval, type)
    return dst


def closing(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(cv2.dilate(image, kernel), kernel)


def opening(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(cv2.erode(image, kernel), kernel)


def main():
    # src_img = cv2.imread("./data/house.jpg", cv2.IMREAD_GRAYSCALE)
    src_img = cv2.imread("../../data/cross_0256x0256.png", cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print('Could not read image')
        exit(1)

    dst_bin = [sign_image(src_img, "Original")]
    bin_images = list()
    bin_images.append(binarize_image(src_img, 127, 255, cv2.THRESH_BINARY))
    bin_images.append(binarize_image(src_img, 127, 255, cv2.THRESH_BINARY_INV))
    bin_images.append(binarize_image(src_img, 127, 255, cv2.THRESH_TRUNC))
    bin_images.append(binarize_image(src_img, 127, 255, cv2.THRESH_TOZERO))
    dst_bin.append(sign_image(bin_images[0], "Thresh_Binary"))
    dst_bin.append(sign_image(bin_images[1], "Thresh_Binary_Inv"))
    dst_bin.append(sign_image(bin_images[2], "Thresh_Trunc"))
    dst_bin.append(sign_image(bin_images[3], "Thresh_ToZero"))

    bin_images1 = list()
    bin_images1.append(sign_image(binarize_image(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU), "Otsu"))

    dst = [sign_image(bin_images[0], "Original bin image")]
    dst.append(sign_image(closing(bin_images[0]), "Closing"))
    dst.append(sign_image(opening(bin_images[0]), "Opening"))
    dst.append(sign_image(opening(closing(bin_images[0])), "Closing-->Opening"))
    dst.append(sign_image(closing(opening(bin_images[0])), "Opening-->Closing"))
    for i in range(len(bin_images1), len(dst_bin)):
        bin_images1.append(np.full((320, 256), 255, dtype=src_img.dtype))
    for i in range(len(dst), len(dst_bin)):
        dst.append(np.full((320, 256), 255, dtype=src_img.dtype))
    res1 = cv2.hconcat(dst_bin)
    res2 = cv2.hconcat(bin_images1)
    res3 = cv2.hconcat(dst)
    # job_hist = Histogram(src_img)
    # job_hist.start()
    res = cv2.vconcat([res1, res2, res3])
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./output/result_L4.png", res)


if __name__ == "__main__":
    main()
