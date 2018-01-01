from Stitcher import Stitcher
import cv2
import time
import glob
import os

resultAddress = "result\\featureSearchIncre\\"

def pairwiseStitch(fileList):
    stitcher = Stitcher()
    stitcher.printAndWrite("Stitching " + str(fileList[0]) + " and " + str(fileList[1]))
    imageA = cv2.imread(fileList[0], 0)
    imageB = cv2.imread(fileList[1], 0)
    startTime = time.time()
    (status, offset) = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])
    endTime = time.time()
    if status == False:
        stitcher.printAndWrite(offset)
        return (status, offset)
    else:
        stitcher.printAndWrite("  The time of registering is " + str(endTime - startTime) + "s")
        startTime = time.time()
        (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = stitcher.getStitchByOffset([imageA, imageB], offset)
        endTime = time.time()
        stitcher.printAndWrite("  The time of fusing is " + str(endTime - startTime) + "s")
        return (status, stitchImage)

def gridStitch(fileList):
    stitcher = Stitcher()
    stitcher.printAndWrite("Stitching the directory which have" + str(fileList[0]))
    fileNum = len(fileList)
    offsetList = []

    # calculating the offset for small image
    startTime = time.time()
    for fileIndex in range(0, fileNum - 1):
        stitcher.printAndWrite("stitching" + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]))
        imageA = cv2.imread(fileList[fileIndex], 0)
        imageB = cv2.imread(fileList[fileIndex + 1], 0)
        # clahe = cv2.createCLAHE(clipLimit=stitcher.clipLimit, tileGridSize=(stitcher.tileSize, stitcher.tileSize))
        # tempA = clahe.apply(imageA)
        # tempB = clahe.apply(imageB)
        # tempA = cv2.equalizeHist(imageA)
        # tempB = cv2.equalizeHist(imageB)
        (status, offset) = stitcher.calculateOffsetForFeatureSearch([imageA, imageB])
        # (status, offset) = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])
        if status == False:
            return (False, "  " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]) + str(offset))
        else:
            offsetList.append(offset)
    endTime = time.time()
    stitcher.printAndWrite("The time of registering is " + str(endTime - startTime) + "s")
    stitcher.printAndWrite("  The offsetList is " + str(offsetList))

    # stitching and fusing
    stitcher.printAndWrite("start stitching")
    startTime = time.time()
    dxSum = 0; dySum = 0
    stitchImage = cv2.imread(fileList[0], 0)
    for fileIndex in range(0, fileNum - 1):
        stitcher.printAndWrite("  stitching " + str(fileList[fileIndex + 1]))
        imageB = cv2.imread(fileList[fileIndex + 1], 0)
        dxSum = offsetList[fileIndex][0] + dxSum
        dySum = offsetList[fileIndex][1] + dySum
        offset = [dxSum, dySum]
        stitcher.printAndWrite("  The offsetX is " + str(offsetList[fileIndex][0]) + " and the offsetY is " + str(offsetList[fileIndex][1]))
        stitcher.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
        (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = stitcher.getStitchByOffset([stitchImage, imageB], offset)
        if dxSum < 0:
             dxSum = 0
        if dySum < 0:
             dySum = 0

    endTime = time.time()
    stitcher.printAndWrite("The time of fusing is " + str(endTime - startTime) + "s")
    return (True, stitchImage)

def ironPariwiseStitch():
    # Image stitching For iron By pairwise stitching
    projectAddress = ".\\images\\iron"
    fileNum = 50
    for i in range(0, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        outputAddress = resultAddress + "iron" + str.capitalize(Stitcher.fuseMethod) + "\\"
        if not os.path.exists(outputAddress):
            os.makedirs(outputAddress)
        Stitcher.outputAddress = outputAddress
        (status, result) = pairwiseStitch(fileList)
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("stitching Failed")

def dendriticCrystalGridStitch():
    # Image stitching For iron By pairwise stitching
    projectAddress = ".\\images\\dendriticCrystal"
    fileNum = 11
    for i in range(3, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        outputAddress = resultAddress + "dendriticCrystal" + str.capitalize(Stitcher.fuseMethod) + "\\"
        if not os.path.exists(outputAddress):
            os.makedirs(outputAddress)
        Stitcher.outputAddress = outputAddress
        (status, result) = gridStitch(fileList)
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("stitching Failed")

def superalloyTurbinebladeGridStitch():
    # Image stitching For iron By pairwise stitching
    projectAddress = ".\\images\\superalloyTurbineblade"
    fileNum = 1
    for i in range(0, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        outputAddress = resultAddress + "superalloyTurbineblade" + str.capitalize(Stitcher.fuseMethod) + "\\"
        if not os.path.exists(outputAddress):
            os.makedirs(outputAddress)
        Stitcher.outputAddress = outputAddress
        (status, result) = gridStitch(fileList)
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("stitching Failed")


if __name__=="__main__":
    Stitcher.featureMethod = "sift"     # "sift","surf" or "orb"
    Stitcher.searchRatio = 0.9           # 0.75 is common value for matches
    Stitcher.offsetCaculate = "mode"    # "mode" or "ransac"
    Stitcher.offsetEvaluate = 2      # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
    Stitcher.roiRatio = 0.1             # roi length for stitching in first direction
    Stitcher.fuseMethod = "notFuse"
    Stitcher.clipLimit = 25
    Stitcher.tileSize = 10

    # ironPariwiseStitch()
    # dendriticCrystalGridStitch()
    superalloyTurbinebladeGridStitch()