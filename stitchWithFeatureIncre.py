from Stitcher import Stitcher
import cv2
import time
import glob
import os
import skimage
import PIL
from scipy.misc import imsave

resultAddress = ".\\result\\featureSearchIncre\\"

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
        (status, offset) = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])
        if status == False:
            print("  " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]) + str(offset))
            break
            # return (False, "  " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]) + str(offset))
        else:
            offsetList.append(offset)
    endTime = time.time()
    stitcher.printAndWrite("The time of registering is " + str(endTime - startTime) + "s")
    stitcher.printAndWrite("  The offsetList is " + str(offsetList))
    # offsetList = [[-138, -4730], [-237, -4725], [267, -4760], [-128, -4606], [-286, -4673], [-179, -4702], [81, -4696], [-85, -4783],
    #  [39, -4879], [-206, -4575], [-282, -4697], [84, -4702], [180, -4746], [-152, -4615], [-129, -4622], [-18, -4727],
    #  [-278, -4706], [-32, -4713], [8, -4698], [242, -4625], [100, -4814], [-12, -4641]]

    # stitching and fusing
    stitcher.printAndWrite("start stitching")
    startTime = time.time()
    dxSum = 0; dySum = 0
    stitchImage = cv2.imread(fileList[0], 0)
    fileNum = len(offsetList)
    for fileIndex in range(0, fileNum):
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
    for i in range(4, fileNum):
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

def zirconGridStitch():
    # Image stitching For iron By pairwise stitching
    projectAddress = ".\\images\\zirconLarge"
    fileNum = 1
    for i in range(0, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        outputAddress = resultAddress + "zirconLarge" + str.capitalize(Stitcher.fuseMethod) + "\\"
        if not os.path.exists(outputAddress):
            os.makedirs(outputAddress)
        Stitcher.outputAddress = outputAddress
        (status, result) = gridStitch(fileList)
        print(outputAddress + "stitching_result_" + str(i + 1) + ".tif")
        cv2.imwrite(outputAddress + "stitching_result_" + str(i + 1) + ".tif", result)
        # skimage.io.imsave(outputAddress + "stitching_result_" + str(i + 1) + ".jpg", result)
        # imsave(outputAddress + "stitching_result_" + str(i + 1) + ".jpg", result)
        print("over")
        if status == False:
            print("stitching Failed")

if __name__=="__main__":
    Stitcher.featureMethod = "sift"      # "sift","surf" or "orb"
    Stitcher.searchRatio = 0.9           # 0.75 is common value for matches
    Stitcher.offsetCaculate = "mode"     # "mode" or "ransac"
    Stitcher.offsetEvaluate = 2           # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
    Stitcher.roiRatio = 0.2               # roi length for stitching in first direction
    Stitcher.fuseMethod = "notFuse"

    # ironPariwiseStitch()
    # dendriticCrystalGridStitch()
    Stitcher.direction = 4
    Stitcher.directIncre = 0
    Stitcher.isEnhance = True
    Stitcher.isClahe = True
    zirconGridStitch()