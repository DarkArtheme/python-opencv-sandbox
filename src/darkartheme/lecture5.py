import cv2

import numpy as np
from scipy import ndimage
from render import render_circle
from tools import sign_image


def roberts_wrong(image):
    kernel_x = np.array([[1, 0],
                         [0, -1]])
    kernel_y = np.array([[0, 1],
                         [-1, 0]])
    gx = cv2.filter2D(image, cv2.CV_16S, kernel_x)
    gy = cv2.filter2D(image, cv2.CV_16S, kernel_y)
    gx, gy = (np.asarray(np.clip(el, 0, 255), dtype=np.uint8) for el in (gx, gy))
    g = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    g = (np.asarray(np.clip(g, 0, 255), dtype=np.uint8))
    return gx, gy, g


def roberts(image):
    kernel_x = np.array([[1, 0],
                         [0, -1]])
    kernel_y = np.array([[0, 1],
                         [-1, 0]])
    image = np.asarray(image, dtype=np.int32)

    gx = ndimage.convolve(image, kernel_x)
    gy = ndimage.convolve(image, kernel_y)

    g = np.sqrt(np.square(gx) + np.square(gy))
    gx, gy, g = (np.asarray(np.clip(el, 0, 255), dtype=np.uint8) for el in (gx, gy, g))
    return gx, gy, g


def _generate_row(src_img, method_name, func):
    gx, gy, g = func(src_img)
    dst = [sign_image(src_img, "Original")]
    dst.append(sign_image(gx, f"{method_name} Hx"))
    dst.append(sign_image(gy, f"{method_name} Hy"))
    dst.append(sign_image(g, f"{method_name} Gradient"))
    return cv2.hconcat(dst)


def main():
    src_img2 = cv2.imread("./data/cross_0256x0256.png", cv2.IMREAD_GRAYSCALE)
    if src_img2 is None:
        print('Could not read image')
        exit(1)
    src_img1 = render_circle()

    res1 = cv2.vconcat([_generate_row(src_img1, "", roberts_wrong),
                        _generate_row(src_img2, "", roberts_wrong)])
    cv2.namedWindow("Wrong Roberts", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Wrong Roberts", res1)

    res2 = cv2.vconcat([_generate_row(src_img1, "Roberts", roberts),
                        _generate_row(src_img2, "Roberts", roberts)])
    cv2.namedWindow("Right Roberts", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Right Roberts", res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./output/result_L5-1.png", res2)


if __name__ == "__main__":
    main()
