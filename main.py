import Stitcher
import numpy as np
import cv2
import glob

if __name__=="__main__":
    # # Image stitching For iron By pairwise stitching
    # projectAddress = ".\\images\\iron"
    # fileNum = 50
    # for i in range(0, fileNum):
    #     fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
    #     fileList = glob.glob(fileAddress + "*.jpg")
    #
    #     outputAddress = "result\\ironMinimum\\"
    #     evaluate = (True, "evaluate.txt")
    #     isPrintLog = True
    #     stitcher = Stitcher.Stitcher(outputAddress, evaluate, isPrintLog)
    #     registrateMethod = ("featureSearchWithIncrease", "surf", 0.75, ("mode", 100), (150, -1))
    #     # registrateMethod = ("featureSearchWithIncrease", "surf", 0.75, ("ransac", 100), (150, -1))
    #     fuseMethod = ("minimum", "Test")
    #
    #     (status, result) = stitcher.pairwiseStitch(fileList, registrateMethod, fuseMethod, direction="vertical")
    #     if status == True:
    #         cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
    #     if status == False:
    #         print("拼接失败")

    # Image stitching For Al11La3 By grid stitching
    projectAddress = ".\\images\\dendriticCrystal"
    fileNum = 2
    for i in range(1, fileNum):
        outputAddress = "result\\dendriticCrystalNotFuse\\"
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        filePosition = [[1, 15], [16, 30], [31, 45], [46, 60], [61, 75], [76, 90]]
        evaluate = (True, "evaluate.txt")
        isPrintLog = True
        stitcher = Stitcher.Stitcher(outputAddress, evaluate, isPrintLog)
        registrateMethod = ("featureSearchWithIncrease", "surf", 0.5, ("mode", 100), (150, -1))
        fuseMethod = ("notFuse", "Test")
        (status, result) = stitcher.gridStitch(fileList, filePosition, registrateMethod, fuseMethod, shootOrder="snakeByCol")
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("拼接失败")