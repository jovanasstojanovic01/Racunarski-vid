import numpy as np
import cv2
import matplotlib.pyplot as plt

def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(3, 3), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded

if __name__ == '__main__':
    img = cv2.imread("./slika2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Duplicate edge pixels to expand the image
    img_expanded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # Create a white border marker
    marker = np.ones_like(img_expanded[:, :, 0]) * 255
    marker[1:-1, 1:-1] = 0  # Set inner region to black

    plt.imshow(marker, cmap='gray')
    plt.show()

    # Create a copy of the expanded image to preserve original values
    img_expanded_copy = img_expanded.copy()

    # Convert the expanded image to grayscale
    img_gray_expanded = cv2.cvtColor(img_expanded, cv2.COLOR_RGB2GRAY)

    # Perform morphological reconstruction on the copy
    reconstructed = morphological_reconstruction(marker, img_gray_expanded)

    # Fix the edges in the reconstructed image
    # reconstructed = fix_edges(reconstructed)

    # Create a new image containing what is in 'img' but not in 'reconstructed'
    difference_img = cv2.subtract(img_gray_expanded, reconstructed)

    # Reduce the image by one pixel on all sides
    difference_img = difference_img[1:-1, 1:-1]

    # Display the original image, the binary mask, and the reconstructed image
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.subplot(132), plt.imshow(reconstructed, cmap='gray'), plt.title('Reconstructed Image')
    plt.subplot(133), plt.imshow(difference_img, cmap='gray'), plt.title('Difference Image')
    plt.show()
    