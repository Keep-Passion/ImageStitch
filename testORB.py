import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('111.jpg', 0)

# Initiate STAR detector
descriptor = cv2.ORB_create(5000000)
# 检测SIFT特征点，并计算描述子
kps, features = descriptor.detectAndCompute(img, None)
print(len(features))
print(features.shape)
# img2 = cv2.drawKeypoints(img, kps,color=(0,255,0), flags=0)
# plt.imshow(img2),plt.show()