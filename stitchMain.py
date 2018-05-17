from Stitcher import Stitcher
import cv2
import time
import glob
import os


def stitchWithFeatureIncre():
    method = "featureSearchIncre"
    Stitcher.featureMethod = "surf"     # "sift","surf" or "orb"
    Stitcher.searchRatio = 0.75          # 0.75 is common value for matches
    Stitcher.offsetCaculate = "mode"    # "mode" or "ransac"
    Stitcher.offsetEvaluate = 3         # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
    Stitcher.roiRatio = 0.2             # roi length for stitching in first direction
    Stitcher.fuseMethod = "fadeInAndFadeOut"
    stitcher = Stitcher()

    # projectAddress = "images\\iron"
    # outputAddress = "result\\" + method + "\\iron" + str.capitalize(Stitcher.fuseMethod) + "\\"
    # stitcher.imageSetStitch(projectAddress, outputAddress, 50, stitcher.calculateOffsetForFeatureSearchIncre,
    #                         startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    projectAddress = "images\\dendriticCrystal\\"
    outputAddress = "result\\" + method + "\\dendriticCrystal" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitch(projectAddress, outputAddress, 11, stitcher.calculateOffsetForFeatureSearchIncre,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

def stitchWithPhase():
    method = "phaseCorrelate"
    Stitcher.fuseMethod = "notFuse"
    stitcher = Stitcher()
    projectAddress = "images\\zirconSmall"
    outputAddress = "result\\" + method + "\\zirconSmall" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitch(projectAddress, outputAddress, 51, stitcher.calculateOffsetForPhaseCorrleate,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")
    Stitcher.phase.shutdown()

if __name__=="__main__":
    stitchWithFeatureIncre()
    # stitchWithPhase()