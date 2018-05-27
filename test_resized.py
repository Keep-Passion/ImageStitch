import cv2
import glob
import os

def resize(img, resizeTimes):
    (h, w) = img.shape
    resizeH = int(h * resizeTimes)
    resizeW = int(w * resizeTimes)
    return cv2.resize(img, (resizeW, resizeH), interpolation=cv2.INTER_AREA)

outputProject = ".\\images\\zirconLargeResized_4_INTER_AREA\\"
inputProject = ".\\images\\zirconLarge\\"

Num = 97
for i in range(0, Num):
    fileList = glob.glob(inputProject + str(i+1) + "\\" + "*.jpg")
    fileNum = len(fileList)
    if os.path.exists(outputProject + str(i+1) + "\\") is False:
        os.makedirs(outputProject + str(i+1))
    for j in range(0, fileNum):
        fileName = fileList[j].split("\\")[-1]
        img = cv2.imread(fileList[j], 0)
        imgResized = resize(img, 0.25)
        cv2.imwrite(outputProject + str(i + 1) + "\\" + fileName, imgResized)



