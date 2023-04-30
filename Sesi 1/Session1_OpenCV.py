import cv2
import numpy as np

image = cv2.imread('lena.jpg')

def showResult(winname = None, image = None):
    # winname : name of desktop apps
    # image : image want to be shown
    cv2.imshow(winname, image)
    cv2.waitKey(0)           # So that this won't be auto closed after run
    cv2.destroyAllWindows()  # Destroy memory

showResult('Lena Image', image)
# Close using ESC, don't press X

# RGB for the color standard -> BGR if read by Python
print(image.shape) # The three channel in RGB

blue_image = image.copy()
blue_image[:, :, (1, 2)] = 0
showResult('Blue Lena Image', blue_image)

green_image = image.copy()
green_image[:, :, (0, 2)] = 0
showResult('Blue Lena Image', green_image)

red_image = image.copy()
red_image[:, :, (0, 1)] = 0
showResult('Blue Lena Image', red_image)

image_vstack = np.vstack((blue_image, green_image, red_image))
image_hstack = np.hstack((blue_image, green_image, red_image))

showResult('Vertically Stacked Lena Image', image_vstack)
showResult('Horizontally Stacked Lena Image', image_hstack)
