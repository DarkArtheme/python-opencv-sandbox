import cv2
import os

import numpy as np


def main():
    src_img = cv2.imread("./data/cross_0256x0256.png", cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print('Could not read image')
        exit(1)
    # images = read_dataset("./data/other", cv2.IMREAD_GRAYSCALE)
    # write_dataset("./output/other", images, "grayscale")
    kernel = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    dst = cv2.filter2D(src=src_img, ddepth=-1, kernel=kernel)
    cv2.imwrite("./output/result.png", cv2.hconcat([src_img, dst]))


if __name__ == "__main__":
    main()
