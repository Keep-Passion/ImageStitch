# 使用opencv中的CLAHE函数，并使用滑块控制
# 注： skimage中也有CLAHE函数，但是无法使用滑块控制
import cv2
import numpy as np


# 当调节滑块时，调用这个函数。这个没有使用到
def do_nothing(x):
    pass

# inputImage = cv2.imread('.\\Scene\\scene01.jpg', 0)
inputImage = cv2.imread('C:\\Coding_Test\\python\\ImageStitch\\images\\superalloyTurbineblade\\1\\SHT-A5-100-071215_007.JPG', 0)

cv2.namedWindow("CLAHE", 0)
cv2.createTrackbar('clip_limit', 'CLAHE', 20, 50, do_nothing)
cv2.createTrackbar('kenel_size', 'CLAHE', 2, 20, do_nothing)
inverseImage = cv2.equalizeHist(inputImage)
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    # globalEqualizeHistImage = skimage.exposure.equalize_hist(inverseImage, nbins=256, mask=None)
    clip_limit = cv2.getTrackbarPos('clip_limit', 'CLAHE')
    kenel_size = cv2.getTrackbarPos('kenel_size', 'CLAHE')
    # claheImage = np.uint8(skimage.exposure.equalize_adapthist(inverseImage, clip_limit=clip_limit * 1.0/100,kenel_size=kenel_size)*256)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kenel_size * 5, kenel_size * 5))
    cl1 = clahe.apply(inputImage)

    cv2.imshow("CLAHE", np.hstack([inputImage, cl1]))

cv2.destroyAllWindows()