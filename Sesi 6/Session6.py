import cv2
import matplotlib.pyplot as plt
import numpy as np

object_image = cv2.imread('marjan.png')
scene_image = cv2.imread('marjan_banyak.png')


# Sift
# -> Will use Euclidean Distance to search for the similarity (Change to Float32)
SIFT = cv2.SIFT_create()

sift_kp_object, sift_ds_object = SIFT.detectAndCompute(object_image, None)
sift_kp_scene, sift_ds_scene = SIFT.detectAndCompute(scene_image, None)

sift_ds_object = np.float32(sift_ds_object)
sift_ds_scene = np.float32(sift_ds_scene)


# Akaze
# -> Will use Euclidean Distance to search for the similarity (Change to Float32)
AKAZE = cv2.AKAZE_create()

akaze_kp_object, akaze_ds_object = AKAZE.detectAndCompute(object_image, None)
akaze_kp_scene, akaze_ds_scene = AKAZE.detectAndCompute(scene_image, None)

akaze_ds_object = np.float32(akaze_ds_object)
akaze_ds_scene = np.float32(akaze_ds_scene)


# Orb
# -> Will use Hamming (So it is not necessary to change into Float32)
ORB = cv2.ORB_create()

orb_kp_object, orb_ds_object = ORB.detectAndCompute(object_image, None)
orb_kp_scene, orb_ds_scene = ORB.detectAndCompute(scene_image, None)

flann = cv2.FlannBasedMatcher(dict(algorithm = 1), dict(checks = 50))
bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)


# Matching
sift_match = flann.knnMatch(sift_ds_object, sift_ds_scene, 2)

akaze_match = flann.knnMatch(akaze_ds_object, akaze_ds_scene, 2)

orb_match = bfmatcher.match(orb_ds_object, orb_ds_scene)
orb_match = sorted(orb_match, key = lambda x : x.distance)


# Create Masking -> we don't change from the original image, every pixel will started from 0, if match it'll be changed into 1
def createMasking(mask, match):
    for i, (fm, sm) in enumerate(match):
        if fm.distance < 0.7 * sm.distance:
            mask[i] = [1, 0]
    return mask


sift_matches_mask = [[0, 0] for i in range(0, len(sift_match))]
sift_matches_mask = createMasking(sift_matches_mask, sift_match)

akaze_matches_mask = [[0, 0] for i in range(0, len(akaze_match))]
akaze_matches_mask = createMasking(akaze_matches_mask, akaze_match)

sift_result = cv2.drawMatchesKnn(
    object_image, 
    sift_kp_object,
    scene_image,
    sift_kp_scene,
    sift_match,
    None,
    matchColor=[255, 0, 0],
    singlePointColor=[0, 255, 0],
    matchesMask=sift_matches_mask
)

akaze_result = cv2.drawMatchesKnn(
    object_image,
    akaze_kp_object,
    scene_image,
    akaze_kp_scene,
    akaze_match,
    None,
    matchColor=[255, 0, 0],
    singlePointColor=[0, 255, 0],
    matchesMask=akaze_matches_mask
)

orb_result = cv2.drawMatches(
    object_image,
    orb_kp_object,
    scene_image,
    orb_kp_scene,
    orb_match[:20],
    None,
    matchColor=[255, 0, 0],
    singlePointColor=[0, 255, 0],
    flags=2
)

res_labels = ['Sift', 'Akaze', 'Orb']
res_images = [sift_result, akaze_result, orb_result]

plt.figure(figsize=(12, 12))
for i, (lbl, img) in enumerate(zip(res_labels, res_images)):
    plt.subplot(2, 2, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(lbl)

plt.show()