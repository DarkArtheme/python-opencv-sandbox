import cv2

import numpy as np
from tools import sign_image, read_image


def harris_corner_detection(img):
    res = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    harris_res = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    harris_res = cv2.dilate(harris_res, None)
    res[harris_res > 0.01 * harris_res.max()] = [0, 0, 255]
    return res


# NOTE: Это не чистый Ши-Томаси, тут оберточная функция, которая упрощает детектирование углов
def shi_tomasi_detection(img):
    res = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(res, (x, y), 3, 255, -1)

    return res


def _generate_row(src_img, method_name, func):
    result = func(src_img)
    dst = [sign_image(src_img, "Original")]
    dst.append(sign_image(result, f"Result of {func.__name__}"))
    return cv2.hconcat(dst)


def _create_window(images, name, func):
    res = cv2.vconcat([_generate_row(image, name, func) for image in images])
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, res)
    cv2.imwrite(f"./output/result_L6-{name}.png", res)


def main():
    files = ("cross_0256x0256.png", "letters.jpg")
    paths = (f"./data/{file}" for file in files)
    images = [read_image(path) for path in paths]

    for i in range(len(files)):
        _create_window([images[i]], "Harris " + files[i], harris_corner_detection)
        _create_window([images[i]], "Shi-Tomasi " + files[i], shi_tomasi_detection)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
