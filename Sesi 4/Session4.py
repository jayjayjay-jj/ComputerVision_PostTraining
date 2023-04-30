import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('fruits.jpg')
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def showResult(nrow = None, ncol = None, res_stack = None):
    plt.figure(figsize=(12, 12))
    for i, (label, image) in enumerate(res_stack):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(label)
        plt.axis('off')
    plt.show()


# Laplacian
laplace_8U = cv2.Laplacian(igray, cv2.CV_8U)
laplace_16S = cv2.Laplacian(igray, cv2.CV_16S)
laplace_32F = cv2.Laplacian(igray, cv2.CV_32F)
laplace_64F = cv2.Laplacian(igray, cv2.CV_64F)

laplace_labels = ['8U', '16S', '32F', '64F']
laplace_images = [laplace_8U, laplace_16S, laplace_32F, laplace_64F]

showResult(2, 2, zip(laplace_labels, laplace_images))


# Sobel
ksize = 3  # every pixel will be calculated in resulting a new pixel
sobel_x = cv2.Sobel(igray, cv2.CV_32F, 1, 0, ksize)
sobel_y = cv2.Sobel(igray, cv2.CV_32F, 0, 1, ksize)

sobel_labels = ['Sobel X', 'Sobel Y']
sobel_images = [sobel_x, sobel_y]

showResult(1, 2, zip(sobel_labels, sobel_images))

merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
merged_sobel *= 255  /merged_sobel.max()
showResult(1, 1, zip(['Merged Sobel'], [merged_sobel]))


# Canny -> Using threshold to get the edge
canny_50_100 = cv2.Canny(igray, 50, 100)
canny_50_150 = cv2.Canny(igray, 50, 150)
canny_75_150 = cv2.Canny(igray, 75, 150)
canny_75_225 = cv2.Canny(igray, 75, 225)

canny_labels = ['Canny 50 100', 'Canny 50 150', 'Canny 75 150', 'Canny 75 225']
canny_images = [canny_50_100, canny_50_150, canny_75_150, canny_75_225]

showResult(2, 2, zip(canny_labels, canny_images))
