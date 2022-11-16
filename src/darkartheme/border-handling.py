import cv2

from tools import sign_image


def different_borders(images):
    r = 128
    res = []
    for brd_type in [(cv2.BORDER_REFLECT, "Reflect"), (cv2.BORDER_CONSTANT, "Constant"),
                     (cv2.BORDER_REFLECT_101, "Reflect 101"), (cv2.BORDER_WRAP, "Wrap")]:
        res.append([sign_image(cv2.copyMakeBorder(image, top=r, bottom=r, left=r, right=r, borderType=brd_type[0]),
                                   brd_type[1], add_border=True, thickness=1) for image in images])
    res = [cv2.vconcat(col) for col in res]
    cv2.namedWindow("Original images", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Original images", cv2.vconcat(images))
    cv2.namedWindow("Borders", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Borders", cv2.hconcat(res))


def main():
    src_img1 = cv2.imread("./data/cross_0256x0256.png")
    if src_img1 is None:
        print('Could not read image')
        exit(1)
    src_img2 = cv2.imread("./data/house.jpg")
    if src_img2 is None:
        print('Could not read image')
        exit(1)
    different_borders([src_img1, src_img2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
