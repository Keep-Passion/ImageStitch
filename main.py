import Stitcher
import numpy as np
import cv2
import glob

def ironPariwiseStitch():
    # Image stitching For iron By pairwise stitching
    projectAddress = ".\\images\\iron"
    fileNum = 50
    for i in range(0, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        outputAddress = "result\\ironTrigonometric\\"
        evaluate = (True, "evaluate.txt")
        isPrintLog = True
        stitcher = Stitcher.Stitcher(outputAddress, evaluate, isPrintLog)
        # registrateMethod = ("featureSearchWithIncrease", "orb", 0.6, ("mode", 4), (150, -1))
        registrateMethod = ("featureSearchWithIncrease", "orb", 0.6, ("ransac", 10), (150, -1))
        fuseMethod = ("trigonometric", "Test")

        (status, result) = stitcher.pairwiseStitch(fileList, registrateMethod, fuseMethod, direction="vertical")
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("拼接失败")

def dendriticCrystalGridStitch():
    # Image stitching For Al11La3 By grid stitching
    projectAddress = ".\\images\\dendriticCrystal"
    fileNum = 11
    filePositionList = []
    for i in range(0, fileNum):
        filePosition = [[1, 15], [16, 30], [31, 45], [46, 60], [61, 75], [76, 90]]
        filePositionList.append(filePosition)
    filePosition6 = [[1, 15], [16, 30], [31, 45], [46, 61], [62, 76], [77, 91]]
    filePositionList[6] = filePosition6
    for i in range(6, fileNum):
        outputAddress = "result\\dendriticCrystalNotFuse\\"
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        filePosition = filePositionList[i]
        evaluate = (True, "evaluate.txt")
        isPrintLog = True
        stitcher = Stitcher.Stitcher(outputAddress, evaluate, isPrintLog)
        registrateMethod = ("featureSearchWithIncrease", "orb", 0.6, ("mode", 10), (150, -1))
        fuseMethod = ("notFuse", "Test")
        (status, result) = stitcher.gridStitch(fileList, filePosition, registrateMethod, fuseMethod, shootOrder="snakeByCol")
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("拼接失败")

if __name__=="__main__":
    ironPariwiseStitch()
    # dendriticCrystalGridStitch()
