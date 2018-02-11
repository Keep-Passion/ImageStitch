import glob
import cv2

projectAddress = "images\\zirconSmall"
Num = 51

for i in range(0, Num):
    fileList = glob.glob(projectAddress + "\\" + str(i+1) + "\\" + "*.jpg")
    img = cv2.imread(fileList[0], 0)
    print(str(img.shape[0]) + "*" + str(img.shape[1]))