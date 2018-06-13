import cv2

surf = cv2.xfeatures2d.SURF_create()

matcher = cv2.DescriptorMatcher_create("BruteForce")
print(surf.getHessianThreshold())
print(surf.getNOctaves())
print(surf.getNOctaveLayers())
print(surf.getExtended())
print(surf.getUpright())