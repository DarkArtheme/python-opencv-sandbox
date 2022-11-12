import os

import cv2
import numpy as np


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


def sign_image(image, text: str):
    text_image = np.full((64, 256), 255, dtype=image.dtype)

    # text_image = cv2.resize(text_image, (256, 128))
    font = cv2.FONT_HERSHEY_DUPLEX
    bottom_left_corner_of_text = (2, 32)
    font_scale = 0.75
    font_color = 0
    thickness = 1
    line_type = -1

    cv2.putText(text_image, text, bottom_left_corner_of_text, font,
                font_scale, font_color, thickness, line_type)

    return cv2.vconcat([image, text_image])
