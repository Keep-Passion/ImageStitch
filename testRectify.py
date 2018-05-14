import glob
import cv2
import ImageUtility as utility

fileList = glob.glob("C:\\Coding_Test\\python\\ImageStitch\\images\\needRectify\\*.jpg")

for i in range(len(fileList)):
    image = cv2.imread(fileList[i], 0)
    imageName = fileList[i].split("\\")[-1]
    print(imageName)
    method = utility.Method()
    resultImage = method.rectifyFinalImg(image)
    cv2.imwrite("C:\\Coding_Test\\python\\ImageStitch\\result\\rectifyResult\\" + imageName, resultImage)