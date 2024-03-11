import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded

if __name__ == '__main__':
    img = cv.imread('coins.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask_circles = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_filtered = cv2.morphologyEx(mask_circles, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    imgSat = imgHSV[:, :, 1]

    _, mask_circles_lower = cv2.threshold(imgSat, 40, 255, cv2.THRESH_BINARY)
    mask_filtered2 = cv2.morphologyEx(mask_circles_lower, cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask_filtered2 = cv2.morphologyEx(mask_filtered2, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
    marker = cv2.bitwise_and(mask_filtered2, mask_filtered)
    reconstructed = morphological_reconstruction(marker, mask_filtered)

    plt.figure(figsize=(8, 20))
    plt.subplot(231)
    plt.imshow(img)
    plt.subplot(232)
    plt.imshow(img_gray)
    plt.subplot(233)
    plt.imshow(mask_filtered)
    plt.subplot(131)
    plt.imshow(imgSat)
    plt.subplot(132)
    plt.imshow(mask_filtered2)
    plt.subplot(133)
    plt.imshow(reconstructed)

    plt.show()