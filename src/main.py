import cv2

import numpy as np
from tools import sign_image
from morphological_operations import *

# images
images = ['./data/chess.png', './data/cross_0256x0256.png', './data/letters.jpg', './data/noise.png']


def sharpening(image):
    """Пример увеличения резкости на изображении"""
    kernel = np.array([[0, 0, 0],
                       [0, 2, 0],
                       [0, 0, 0]]) - \
             np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


def box_filter(image):
    """Бокс фильтр в общем виде"""
    w = 5
    kernel = np.ones((w, w)) / ((w + 1) * (w + 1))
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


def gaussian_filter(image):
    """
    Пример Гауссова фильтра.

    Здесь:
    ksize - размер kernel (окна фильтрации)

    The final two arguments are sigmaX and sigmaY, which are both set to 0.
    These are the Gaussian kernel standard deviations, in the X (horizontal) and Y (vertical) direction.
    The default setting of sigmaY is zero. If you simply  set sigmaX to zero, then the standard deviations are computed
    from the kernel size (width and height respectively). You can also explicitly set the size of each argument to
    positive values greater than zero.
    """
    return cv2.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0, sigmaY=0)


def custom_linear_filter(image):
    """Пример кастомного линейного фильтра из лекции (по факту это лапласиан гауссиана)"""
    kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_DEFAULT)


def median_filter(image):
    """Медианный фильтр. ksize - размер стороны квадрата окна"""
    return cv2.medianBlur(src=image, ksize=5)


def bilateral_filter(image):
    """
    Билатеральный фильтр.

    d - Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
    it is computed from sigmaSpace

    sigmaColor - Filter sigma in the color space. A larger value of the parameter means that farther colors within
    the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.

    sigmaSpace - Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels
    will influence each other as long as their colors are close enough. When d > 0, it specifies the neighborhood size
    regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
    """
    return cv2.bilateralFilter(image, d=15, sigmaColor=80, sigmaSpace=80)


def main():
    src_images = list(map(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE), images))

    for i in src_images:
        if i is None:
            print('Could not read image')
            exit(1)

    # images = read_dataset("./data/other", cv2.IMREAD_GRAYSCALE)
    # write_dataset("./output/other", images, "grayscale")
    # dst = [sign_image(src_img, "Original"), sign_image(sharpening(src_img), "Sharpening"),
    #        sign_image(box_filter(src_img), "Box filter"), sign_image(gaussian_filter(src_img), "Gaussian filter"),
    #        sign_image(custom_linear_filter(src_img), "Custom (LoG) filter"),
    #        sign_image(median_filter(src_img), "Median filter"),
    #        sign_image(bilateral_filter(src_img), "Bilateral filter")]

    dst = [sign_image(erosion(src_images[2]), "Erosion filter"), sign_image(dilation(src_images[2]), "Dilation filter"),
           sign_image(opening(src_images[2]), "Opening filter"), sign_image(closing(src_images[2]), "Closing filter")]
    res = cv2.hconcat(dst)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./output/result.png", res)


if __name__ == "__main__":
    main()
