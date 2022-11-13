from tools import sign_image
from morphological_operations import *

# images
images = ['./data/chess.png', './data/cross_0256x0256.png', './data/letters.jpg',
          './data/noise.png', './data/house.jpg']


def create_result(image_num, dst):
    dst = [] + dst
    res = cv2.hconcat(dst)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"./output/result_{image_num}.png", res)


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

    image_num = 2
    dst = [sign_image(src_images[image_num], "Original"),
           sign_image(erosion(src_images[image_num]), "Erosion filter"),
           sign_image(dilation(src_images[image_num]), "Dilation filter")]
    create_result(image_num, dst)
    image_num = 3
    dst = [sign_image(src_images[image_num], "Original"),
           sign_image(opening(src_images[image_num]), "Opening filter"),
           sign_image(closing(src_images[image_num]), "Closing filter")]
    create_result(image_num, dst)


if __name__ == "__main__":
    main()
