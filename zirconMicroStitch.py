from Stitcher import Stitcher
import cv2
import time
import glob
import os

fileAddress = "images\\zirconLarge\\3\\"
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
stitcher = Stitcher()

def zirconMicroStitch():
    fileList = glob.glob(fileAddress + "*." + fileExtension)
    status, result = stitcher.flowStitch(fileList, stitcher.calculateOffsetForFeatureSearchIncre)
    cv2.imwrite(outputAddress, result)

def zirconMicroStitchWithEnhance():
    Stitcher.isEnhance = True
    Stitcher.isClahe = True
    fileList = glob.glob(fileAddress + "*." + fileExtension)
    status, result = stitcher.flowStitch(fileList, stitcher.calculateOffsetForFeatureSearchIncre)
    cv2.imwrite(outputAddress, result)

if __name__=="__main__":
    zirconMicroStitch()
    # zirconMicroStitchWithEnhance()


