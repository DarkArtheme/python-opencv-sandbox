import cv2
import numpy as np


def render_circle():
    image = np.full((256, 256), 120, np.uint8)
    return cv2.circle(image, (128, 128), 64, 255, -1)