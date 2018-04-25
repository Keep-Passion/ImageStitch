from Stitcher import Stitcher
import cv2
import time
import glob
import os

fileAddress = "images\\zirconLargeResized_4\\1\\"
fileExtension = "jpg"
outputAddress = "result\\stitchResult.png"

method = "featureSearchIncre"
Stitcher.featureMethod = "sift"     # "sift","surf" or "orb"
Stitcher.searchRatio = 0.9          # 0.75 is common value for matches
Stitcher.offsetCaculate = "mode"    # "mode" or "ransac"
Stitcher.offsetEvaluate = 2         # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
Stitcher.roiRatio = 0.2             # roi length for stitching in first direction
Stitcher.fuseMethod = "notFuse"
Stitcher.direction = 4
Stitcher.directIncre = 0
Stitcher.phaseResponseThreshold = 0.1
stitcher = Stitcher()

def zirconMicroStitch():
    fileList = glob.glob(fileAddress + "*." + fileExtension)
    status, result = stitcher.flowStitch(fileList, stitcher.calculateOffsetForPhaseCorrleateIncre)
    # status, result = stitcher.flowStitch(fileList, stitcher.calculateOffsetForFeatureSearchIncre)
    cv2.imwrite(outputAddress, result)

def zirconMicroStitchWithEnhance():
    Stitcher.isEnhance = True
    Stitcher.isClahe = True
    fileList = glob.glob(fileAddress + "*." + fileExtension)
    status, result = stitcher.flowStitch(fileList, stitcher.calculateOffsetForFeatureSearchIncre)
    cv2.imwrite(outputAddress, result)

def stitchWithFeatureSearchImageSet():
    method = "featureSearch"
    Stitcher.fuseMethod = "notFuse"
    stitcher = Stitcher()
    stitcher.phaseResponseThreshold = 0.3
    stitcher.directIncre = 0
    # projectAddress = "images\\zirconLarge"
    # outputAddress = "result\\" + method + "\\zirconLarge" + str.capitalize(Stitcher.fuseMethod) + "\\"
    # projectAddress = "images\\zirconLargeResized_8_INTER_AREA"
    # outputAddress = "result\\" + method + "\\zirconLargeResized_8_INTER_AREA" + str.capitalize(Stitcher.fuseMethod) + "\\"
    projectAddress = "images\\zirconSmall"
    outputAddress = "result\\" + method + "\\zirconSmall" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMutiple(projectAddress, outputAddress, 168, stitcher.calculateOffsetForFeatureSearch,
                            startNum=168, fileExtension="jpg", outputfileExtension="jpg")

if __name__=="__main__":
    stitchWithFeatureSearchImageSet()


