import os
import threading

import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_dataset(directory: str, flag: int):
    images = []
    for file in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, file), flag)
        images.append(image)
    return np.array(images)


def concatenate_dataset(*args):
    images = []
    for i in range(0, len(args), 4):
        for j in range(len(args[i])):
            image = args[i][j]
            for k in range(1, min(len(args) - i + 1, 4)):
                image += args[i + k][j]
            images.append(image)
    return np.array(images)


def write_dataset(directory: str, images, prefix: str):
    i = 1
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))
    for image in images:
        file = f"{i}_{prefix}.jpg"
        cv2.imwrite(os.path.join(directory, file), image)
        i += 1


def sign_image(image, text: str, add_border=False, thickness=0):
    try:
        height, width, channels = image.shape
    except ValueError:
        height, width = image.shape
        channels = 1
    text_image = np.full((64, width, channels), 255, dtype=image.dtype)

    # text_image = cv2.resize(text_image, (256, 128))
    font = cv2.FONT_HERSHEY_DUPLEX
    bottom_left_corner_of_text = (2, 32)
    font_scale = 0.75
    font_color = 0
    thickness = 1
    line_type = -1

    cv2.putText(text_image, text, bottom_left_corner_of_text, font,
                font_scale, font_color, thickness, line_type)
    res = cv2.vconcat([image, text_image])
    if add_border:
        r = thickness
        res = cv2.copyMakeBorder(res, top=r, bottom=r, left=r, right=r, borderType=cv2.BORDER_CONSTANT)
    return res


def read_image(path, flag=cv2.IMREAD_UNCHANGED):
    src_img = cv2.imread(path, flag)
    if src_img is None:
        print(f"Could not read image '{path}'")
        exit(1)
    return src_img


class Histogram(threading.Thread):
    def __init__(self, img):
        super(Histogram, self).__init__()
        self.img = img

    def run(self):
        plt.figure()
        plt.xlim([0, 256])
        histogram, bin_edges = np.histogram(self.img, bins=256, range=(0, 256))
        plt.plot(bin_edges[0:-1], histogram, color="black")
        plt.title("Histogram")
        plt.xlabel("Color value")
        plt.ylabel("Pixel count")
        plt.show()
