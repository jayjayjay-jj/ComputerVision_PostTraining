import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os

# Train
train_path = 'images/train'
train_dir_list = os.listdir(train_path)

image_list = []
image_class_list = []

for idx, train_dir in enumerate(train_dir_list):
    dir_path = os.listdir(f'{train_path}/{train_dir}')
    for image_path in dir_path:
        image_list.append(f'{train_path}/{train_dir}/{image_path}')
        image_class_list.append(idx)

# for i in image_list:
#    print(i)

sift = cv2.SIFT_create()

descriptor_list = []

for image_path in image_list:
    _, descriptor = sift.detectAndCompute(cv2.imread(image_path), None)
    descriptor_list.append(descriptor)


# Fill in descriptor_stack index 0
descriptor_stack = descriptor_list[0]

# Stack one by one
for descriptor in descriptor_list[1:]:
    descriptor_stack = np.vstack((descriptor_stack, descriptor))

descriptor_stack = np.float32(descriptor_stack)

# K-means clustering -> Take some classes and join together by the class(es)
centroids, _ = kmeans(descriptor_stack, 100, 1)

image_features = np.zeros((len(image_list), len(centroids)), "float32")

for i in range (0, len(image_list)):
    words, _ = vq(descriptor_list[i], centroids)
    for word in words:
        image_features[i][word] += 1

stdScaler = StandardScaler().fit(image_features)
image_features = stdScaler.transform(image_features)

# Linear SVC
svc = LinearSVC()
svc.fit(image_features, np.array(image_class_list))

# Testing
test_path = 'images/test'
image_list = []

for path in os.listdir(test_path):
    image_list.append(f'{test_path}/{path}')

descriptor_list = []

for image_path in image_list:
    _, descriptor = sift.detectAndCompute(cv2.imread(image_path), None)
    descriptor_list.append(descriptor)


# Fill in descriptor_stack index 0
descriptor_stack = descriptor_list[0]

# Stack one by one
for descriptor in descriptor_list[1:]:
    descriptor_stack = np.vstack((descriptor_stack, descriptor))

descriptor_stack = np.float32(descriptor_stack)

# K-means clustering -> Take some classes and join together by the class(es)
centroids, _ = kmeans(descriptor_stack, 100, 1)

test_features = np.zeros((len(image_list), len(centroids)), "float32")

for i in range (0, len(image_list)):
    words, _ = vq(descriptor_list[i], centroids)
    for word in words:
        test_features[i][word] += 1
    

test_features = stdScaler.transform(test_features)
result = svc.predict(test_features)

for class_id, image_path in zip(result, image_list):
    print(f'{image_path} : {train_dir_list[class_id]}')

