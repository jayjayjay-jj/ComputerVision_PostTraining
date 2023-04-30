import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('lena.jpg')
height, width = image.shape[:2]

def showResult(nrow = None, ncol = None, res_stack = None):
    plt.figure(figsize=(12, 12))
    for i, (label, image) in enumerate(res_stack):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off')
    plt.show()

opencv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Average gray -> Each colour got 100% / 3 -> 33% or 0.33
average_gray = np.dot(image, [0.33, 0.33, 0.33])

blue_image, green_image, red_image = image[:, :, 0], image[:, :, 1], image[:, :, 2]
max_channel = max(np.max(blue_image), np.max(green_image), np.max(red_image))
min_channel = min(np.min(blue_image), np.min(green_image), np.min(red_image))
print(max_channel, min_channel)

# Light gray -> Based on the formula
light_gray = np.dot(image, [(max_channel + min_channel) / 2, (max_channel + min_channel) / 2, (max_channel + min_channel) / 2])

# Luminosity gray -> Based on the formula (0.07 Blue + 0.71 Green + 0.21 Red)
luminosity_gray = np.dot(image, [0.07, 0.71, 0.21])

# Weighted Average Gray -> Based on the formula (0.114 Blue + 0.587 Green + 0.299 Red)
weighted_average_gray = np.dot(image, [0.114, 0.587, 0.299])

gray_labels = ['Open CV', 'Average Gray', 'Light Gray', 'Luminosity Gray', 'Weight Average Gray']
gray_images = [opencv_gray, average_gray, light_gray, luminosity_gray, weighted_average_gray]

showResult(3, 2, zip(gray_labels, gray_images))

# Set the threshold to 100, above = 255, below = 0
thresh = 100
thresh_image = opencv_gray.copy()

for i in range(height):
    for j in range(width):
        if thresh_image[i, j] > thresh:
            thresh_image[i, j] = 255
        else :
            thresh_image[i, j] = 0

showResult(1, 1, zip(['Manual Threshold'], [thresh_image]))

_, bin_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_BINARY)
_, binv_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_BINARY_INV)
_, mask_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_MASK)
_, otsu_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_OTSU)
_, tin_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_TOZERO)
_, tinv_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_TOZERO_INV)
_, tri_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_TRIANGLE)
_, trunc_thresh = cv2.threshold(opencv_gray, 100, 255, cv2.THRESH_TRUNC)

thresh_labels = ['man', 'bin', 'binv', 'mask', 'otsu', 'tin', 'tinv', 'tri', 'trunc']
thresh_images = [thresh_image, bin_thresh, binv_thresh, mask_thresh, otsu_thresh, tin_thresh, tinv_thresh, tri_thresh, trunc_thresh]

showResult(3, 3, zip(thresh_labels, thresh_images))


def manual_mean_filter(source, ksize):
    np_source = np.array(source)
    for i in range(height - ksize - 1):
        for j in range(height-ksize - 1):
            matrix = np.array(np_source[i: (i+ksize), j: (j+ksize)]).flatten()
            mean = np.mean(matrix)
            np_source[i + ksize//2, j + ksize//2] = mean

    return np_source


def manual_median_filter(source, ksize):
    np_source = np.array(source)
    for i in range(height - ksize - 1):
        for j in range(height-ksize - 1):
            matrix = np.array(np_source[i: (i+ksize), j: (j+ksize)]).flatten()
            median = np.median(matrix)
            np_source[i + ksize//2, j + ksize//2] = median

    return np_source


b, g, r = cv2.split(image)
mean_b = manual_mean_filter(b, 3)
mean_g = manual_mean_filter(g, 3)
mean_r = manual_mean_filter(r, 3)

median_b = manual_median_filter(b, 3)
median_g = manual_median_filter(g, 3)
median_r = manual_median_filter(r, 3)

merged_mean = cv2.merge((mean_b, mean_g, mean_r))
merged_median = cv2.merge((median_b, median_g, median_r))

blur_image = opencv_gray.copy()

blur = cv2.blur(blur_image, (3, 3))
median_blur = cv2.medianBlur(blur_image, 3)
gauss_blur = cv2.GaussianBlur(blur_image, (3, 3), 2.0)
bilateral_blur = cv2.bilateralFilter(blur_image, 3, 150, 150)

blur_labels = ['blur', 'median blur', 'gauss blur', 'bilateral blur', 'merged mean', 'merged median']
blur_images = [blur, median_blur, gauss_blur, bilateral_blur, merged_mean, merged_median]
showResult(2, 3, zip(blur_labels, blur_images))
