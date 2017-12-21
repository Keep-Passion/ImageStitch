import cv2
import glob

fileList = glob.glob("E:\\SegmentResult\\*.jpg")
num = len(fileList)
for i in range(0, num):
    image = cv2.imread(fileList[i], 0)
    cv2.imwrite(str(i+1) + ".tif", image)