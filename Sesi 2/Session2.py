import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('model.jpg')
igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cmap = we wanna show gray
def showResult(label = None, image = None, cmap = None):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)   # 1 Row, 2 Columns 
    plt.hist(image.flat, bins=256, range=(0, 256))          
    # Show the histogram from the image, image flat turn the pixel into dimension, bins is used for the division of 255-color-depth, range 
    plt.title(label)
    plt.xlabel('Intensity Value')
    plt.ylabel('Intensity Quantity')
    plt.subplot(1, 2, 2) 
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

normal_image = igray.copy()
showResult('Model Image in Histogram', normal_image, 'gray')

nequ_hist = cv2.equalizeHist(igray)
showResult('nequ', nequ_hist, 'gray')

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cequ_hist = clahe.apply(igray)
showResult('cequ', cequ_hist, 'gray')

hist_labels = ['Normal', 'Nequ', 'Cequ']
hist_images = [normal_image, nequ_hist, cequ_hist]

plt.figure(figsize=(12, 12))
for i, (label, image) in enumerate(zip(hist_labels, hist_images)):
    plt.subplot(3, 1, i+1)   # 1 Row, 2 Columns 
    plt.hist(image.flat, bins=256, range=(0, 256))          
    # Show the histogram from the image, image flat turn the pixel into dimension, bins is used for the division of 255-color-depth, range 
    plt.title(label)
    plt.xlabel('Intensity Value')
    plt.ylabel('Intensity Quantity')

plt.show()

plt.figure(figsize=(12, 12))
for i, (label, image) in enumerate(zip(hist_labels, hist_images)):
    plt.subplot(1, 3, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.axis('off')

plt.show()
