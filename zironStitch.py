from Stitcher import Stitcher
import cv2
import time
import glob
import os

fileAddress = "images\\zirconSmall\\10\\"
fileExtension = "jpg"
outputAddress = "result\\stitchResult.jpg"


method = "featureSearchIncre"
Stitcher.featureMethod = "sift"     # "sift","surf" or "orb"
Stitcher.searchRatio = 0.9          # 0.75 is common value for matches
Stitcher.offsetCaculate = "mode"    # "mode" or "ransac"
Stitcher.offsetEvaluate = 2         # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
Stitcher.roiRatio = 0.1             # roi length for stitching in first direction
Stitcher.fuseMethod = "notFuse"
Stitcher.direction = 4
Stitcher.directIncre = 0
stitcher = Stitcher()

def zirconStitch():
    fileList = glob.glob(fileAddress + "*." + fileExtension)
    status, result = stitcher.flowStitch(fileList, stitcher.calculateOffsetForFeatureSearch)
    cv2.imwrite(outputAddress, result)

def zirconStitchWithEnhance():
    Stitcher.isEnhance = True
    Stitcher.isClahe = True
    fileList = glob.glob(fileAddress + "*." + fileExtension)
    status, result = stitcher.flowStitch(fileList, stitcher.calculateOffsetForFeatureSearch)
    cv2.imwrite(outputAddress, result)

def stitchWithFeatureSearchImageSet():
    method = "featureSearch"
    Stitcher.fuseMethod = "notFuse"
    Stitcher.isEnhance = True
    Stitcher.isClahe = True
    stitcher = Stitcher()
    projectAddress = "images\\zirconSmall"
    outputAddress = "result\\" + method + "\\zirconSmall" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitch(projectAddress, outputAddress, 51, stitcher.calculateOffsetForFeatureSearch,
                            startNum=50, fileExtension="jpg", outputfileExtension="jpg")

if __name__=="__main__":
    # zirconStitch()
    zirconStitchWithEnhance()
    # stitchWithFeatureSearchImageSet()