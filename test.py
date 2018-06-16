import cv2

surf = cv2.xfeatures2d.SURF_create()

matcher = cv2.DescriptorMatcher_create("BruteForce")
print(surf.getHessianThreshold())
print(surf.getNOctaves())
print(surf.getNOctaveLayers())
print(surf.getExtended())
print(surf.getUpright())

orb = cv2.ORB_create()
print(orb.nfeatures)

orbScaleFactor = 1.2
orbNlevels = 8
orbEdgeThreshold = 31
orbFirstLevel = 0
orbWTA_K = 2
orbPatchSize = 31
orbFastThreshold = 20
orbBlurForDescriptor = False