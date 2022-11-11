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
