import cv2
import numpy as np

kernel = np.ones((3, 3), np.uint8)


def erosion(img):
    """
    Erosion primarily involves eroding the outer surface (the foreground) of the image. As binary images only contain
    two pixels 0 and 255, it primarily involves eroding the foreground of the image and it is suggested to have the
    foreground as white. The thickness of erosion depends on the size and shape of the defined kernel. We can make
    use of NumPy’s ones() function to define a kernel. There are a lot of other functions like NumPy zeros,
    customized kernels, and others that can be used to define kernels based on the problem in hand.
    """
    kernel = np.ones((10, 10), np.uint8)
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # invert the image
    invert = cv2.bitwise_not(binr)
    return cv2.erode(invert, kernel, iterations=1)


def dilation(img):
    """
    Dilation involves dilating the outer surface (the foreground) of the image. As binary images only contain two
    pixels 0 and 255, it primarily involves expanding the foreground of the image and it is suggested to have the
    foreground as white. The thickness of erosion depends on the size and shape of the defined kernel. We can make
    use of NumPy’s ones() function to define a kernel. There are a lot of other functions like NumPy zeros,
    customized kernels, and others that can be used to define kernels based on the problem at hand. It is exactly
    opposite to the erosion operation
    """
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # invert the image
    invert = cv2.bitwise_not(binr)
    return cv2.dilate(invert, kernel, iterations=1)


def opening(img):
    """
    Opening involves erosion followed by dilation in the outer surface (the foreground) of the image. All the
    above-said constraints for erosion and dilation applies here. It is a blend of the two prime methods. It is
    generally used to remove the noise in the image.
    """
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1)


def closing(img):
    """
    Closing involves dilation followed by erosion in the outer surface (the foreground) of the image. All the
    above-said constraints for erosion and dilation applies here. It is a blend of the two prime methods. It is
    generally used to remove the noise in the image.
    """
    # binarize the image
    binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)
