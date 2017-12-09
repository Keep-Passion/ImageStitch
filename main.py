import Stitcher
import numpy as np
import cv2
import glob

if __name__=="__main__":
    projectAddress = ".\\images\\iron"
    fileNum = 50
    for i in range(0, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")

        outputAddress = "result\\"
        evaluate = (True, "evaluate.txt")
        isPrintLog = True
        stitcher = Stitcher.Stitcher(outputAddress, evaluate, isPrintLog)
        registrateMethod = ("featureSearchWithIncrease", "surf", 0.75, ("mode", 100), (150, -1))
        fuseMethod = ("notFuse", "Test")

        (status, result) = stitcher.pairwiseStitch(fileList, registrateMethod, fuseMethod, direction="vertical")
        if status == True:
            print("拼接成功")
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        # else:
        #     # cv2.namedWindow("Result", 0)
        #     # cv2.imshow("Result", result)
        #     # cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()
