import cv2


def main():
    src_img = cv2.imread("./data/cross_0256x0256.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("./output/result.png", src_img)



if __name__ == "__main__":
    main()
