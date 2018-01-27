import numpy as np
import cv2
from scipy.stats import mode
import time
import os
import glob
import skimage.measure
from numba import jit
import ImageUtility as Utility
import ImageFusion
from phasecorrelation import *

class Stitcher(Utility.Method):
    '''
	    图像拼接类，包括所有跟材料显微组织图像配准相关函数
	'''
    direction = 1
    directIncre = 1
    featureMethod = "surf"      # "sift","surf" or "orb"
    searchRatio = 0.75           # 0.75 is common value for matches
    offsetCaculate = "mode"     # "mode" or "ransac"
    offsetEvaluate = 10          # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
    roiRatio = 0.1              # roi length for stitching in first direction
    fuseMethod = "notFuse"
    isEnhance = False
    isClahe = False
    clipLimit = 20
    tileSize = 5
    phase = phaseCorrelation()

    def directionIncrease(self, direction):
        direction += self.directIncre
        if direction == 5:
            direction = 1
        if direction == 0:
            direction = 4
        return direction

    def calculateOffset(self, images, registrateMethod, direction="horizontal"):
        '''
        Stitch two images
        :param images: [imageA, imageB]
        :param registrateMethod: list:
        :param fuseMethod:
        :param direction: stitching direction
        :return:
        '''
        (imageA, imageB) = images
        offset = [0, 0]
        status = False
        H = np.eye(3, dtype=np.float64)
        if  registrateMethod[0] == "phaseCorrection":
            return (False, "  We don't develop the phase Correction method, Plesae wait for updating", 0)
        elif  registrateMethod[0] == "featureSearchWithIncrease":
            featureMethod = registrateMethod[1]        # "sift","surf" or "orb"
            searchRatio = registrateMethod[2]          # 0.75 is common value for matches
            offsetCaculate = registrateMethod[3][0]    # "mode" or "ransac"
            offsetEvaluate = registrateMethod[3][1]    # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
            roiFirstLength = registrateMethod[4][0]     # roi length for stitching in first direction
            roiSecondLength = registrateMethod[4][1]    # roi length for stitching in second direction

            if direction == "horizontal":
                maxI = int(imageA.shape[1] / (2 * roiFirstLength)) + 1
            elif direction == "vertical":
                maxI = int(imageA.shape[0] / (2 * roiFirstLength)) + 1
            for i in range(1, maxI+1):
                self.printAndWrite("  i=" + str(i) + " and maxI="+str(maxI+1))
                # get the roi region of images
                roiImageA = self.getROIRegion(imageA, direction=direction, order="first", searchLength=i * roiFirstLength,
                                                  searchLengthForLarge=roiSecondLength)
                roiImageB = self.getROIRegion(imageB, direction=direction, order="second", searchLength=i * roiFirstLength,
                                                  searchLengthForLarge=roiSecondLength)
                # get the feature points
                (kpsA, featuresA) = self.detectAndDescribe(roiImageA, featureMethod=featureMethod)
                (kpsB, featuresB) = self.detectAndDescribe(roiImageB, featureMethod=featureMethod)
                matches = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, searchRatio)

                # match all the feature points
                localStartTime = time.time()
                if offsetCaculate == "mode":
                    (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate)
                elif offsetCaculate == "ransac":
                    (status, offset, adjustH) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate)
                    H = adjustH
                if direction == "horizontal" and status == True:
                    offset[1] = offset[1] + imageA.shape[1] - i * roiFirstLength
                elif direction == "vertical" and status == True:
                    offset[0] = offset[0] + imageA.shape[0] - i * roiFirstLength
                if status == True:
                    break
        if status == False:
            return (status, "  The two images can not match", 0)
        elif status == True:
            localEndTime = time.time()
            self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
            self.printAndWrite("  The time of mode/ransac is " + str(localEndTime - localStartTime) + "s")
            return (status, offset, H)

    def flowStitch(self, fileList, caculateOffsetMethod):
        self.printAndWrite("Stitching the directory which have " + str(fileList[0]))
        fileNum = len(fileList)
        offsetList = []
        describtion = ""
        # calculating the offset for small image
        startTime = time.time()
        status = True
        for fileIndex in range(0, fileNum - 1):
            self.printAndWrite("stitching " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex + 1]))
            imageA = cv2.imread(fileList[fileIndex], 0)
            imageB = cv2.imread(fileList[fileIndex + 1], 0)
            if caculateOffsetMethod == self.calculateOffsetForPhaseCorrleate:
                (status, offset) = self.calculateOffsetForPhaseCorrleate([fileList[fileIndex], fileList[fileIndex + 1]])
            else:
                (status, offset) = caculateOffsetMethod([imageA, imageB])
            if status == False:
                describtion = "  " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]) + " can not be stitched"
                break
            else:
                offsetList.append(offset)
        endTime = time.time()
        self.printAndWrite("The time of registering is " + str(endTime - startTime) + "s")
        self.printAndWrite("  The offsetList is " + str(offsetList))

        # stitching and fusing
        self.printAndWrite("start stitching")
        startTime = time.time()
        dxSum = 0;
        dySum = 0
        stitchImage = cv2.imread(fileList[0], 0)
        offsetListNum = len(offsetList)
        for fileIndex in range(0, offsetListNum):
            self.printAndWrite("  stitching " + str(fileList[fileIndex + 1]))
            imageB = cv2.imread(fileList[fileIndex + 1], 0)
            dxSum = offsetList[fileIndex][0] + dxSum
            dySum = offsetList[fileIndex][1] + dySum
            offset = [dxSum, dySum]
            self.printAndWrite("  The offsetX is " + str(offsetList[fileIndex][0]) + " and the offsetY is " + str(
                offsetList[fileIndex][1]))
            self.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
            (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = self.getStitchByOffset(
                [stitchImage, imageB], offset)
            if dxSum < 0:
                dxSum = 0
            if dySum < 0:
                dySum = 0

        endTime = time.time()
        self.printAndWrite("The time of fusing is " + str(endTime - startTime) + "s")

        if status == False:
            self.printAndWrite(describtion)
        return (status, stitchImage)

    def imageSetStitch(self, projectAddress, outputAddress, fileNum, caculateOffsetMethod, startNum = 1, fileExtension = "jpg", outputfileExtension = "jpg"):
        for i in range(startNum, fileNum+1):
            fileAddress = projectAddress + "\\" + str(i) + "\\"
            fileList = glob.glob(fileAddress + "*." + fileExtension)
            if not os.path.exists(outputAddress):
                os.makedirs(outputAddress)
            Stitcher.outputAddress = outputAddress
            (status, result) = self.flowStitch(fileList, caculateOffsetMethod)
            #if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "." + outputfileExtension, result)
            if status == False:
                print("stitching Failed")

    def calculateOffsetForPhaseCorrleate(self, dirAddress):
        (dir1, dir2) = dirAddress
        offset = [0, 0]
        status = True
        # phase = phaseCorrelation()
        offsetList = self.phase.phaseCorrelation(dir1, dir2)
        # print(offset)
        # phase.shutdown()
        offset = []
        offset.append(np.int(np.round(offsetList[1])))
        offset.append(np.int(np.round(offsetList[0])))
        # offset[0] = np.round(offsetList[0])
        # offset[1] = np.round(offsetList[1])
        self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
        return (status, offset)

    def calculateOffsetForFeatureSearch(self, images):
        '''
        Stitch two images
        :param images: [imageA, imageB]
        :param registrateMethod: list:
        :param fuseMethod:
        :param direction: stitching direction
        :return:
        '''
        (imageA, imageB) = images
        offset = [0, 0]
        status = False
        if self.isEnhance == True:
            if self.isClahe == True:
                clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileSize, self.tileSize))
                imageA = clahe.apply(imageA)
                imageB = clahe.apply(imageB)
            elif self.isClahe == False:
                imageA = cv2.equalizeHist(imageA)
                imageB = cv2.equalizeHist(imageB)
        # get the feature points
        (kpsA, featuresA) = self.detectAndDescribe(imageA, featureMethod=self.featureMethod)
        (kpsB, featuresB) = self.detectAndDescribe(imageB, featureMethod=self.featureMethod)
        if featuresA is not None and featuresB is not None:
            matches = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, self.searchRatio)
            # match all the feature points
            if self.offsetCaculate == "mode":
                (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
            elif self.offsetCaculate == "ransac":
                (status, offset, adjustH) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
        if status == False:
            return (status, "  The two images can not match")
        elif status == True:
            self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
            return (status, offset)

    def calculateOffsetForFeatureSearchIncre(self, images):
        '''
        Stitch two images
        :param images: [imageA, imageB]
        :param registrateMethod: list:
        :param fuseMethod:
        :param direction: stitching direction
        :return:
        '''
        (imageA, imageB) = images
        offset = [0, 0]
        status = False
        maxI = (np.floor(0.5 / self.roiRatio) + 1).astype(int)+ 1
        iniDirection = self.direction
        localDirection = iniDirection
        for i in range(1, maxI):
            self.printAndWrite("  i=" + str(i) + " and maxI="+str(maxI))
            while(True):
                # get the roi region of images
                self.printAndWrite("  localDirection=" + str(localDirection))
                roiImageA = self.getROIRegionForIncreMethod(imageA, direction=localDirection, order="first", searchRatio = i * self.roiRatio)
                roiImageB = self.getROIRegionForIncreMethod(imageB, direction=localDirection, order="second", searchRatio = i * self.roiRatio)

                if self.isEnhance == True:
                    if self.isClahe == True:
                        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,tileGridSize=(self.tileSize, self.tileSize))
                        roiImageA = clahe.apply(roiImageA)
                        roiImageB = clahe.apply(roiImageB)
                    elif self.isClahe == False:
                        roiImageA = cv2.equalizeHist(roiImageA)
                        roiImageB = cv2.equalizeHist(roiImageB)
                # get the feature points
                (kpsA, featuresA) = self.detectAndDescribe(roiImageA, featureMethod=self.featureMethod)
                (kpsB, featuresB) = self.detectAndDescribe(roiImageB, featureMethod=self.featureMethod)
                if featuresA is not None and featuresB is not None:
                    matches = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, self.searchRatio)
                    # match all the feature points
                    if self.offsetCaculate == "mode":
                        (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
                    elif self.offsetCaculate == "ransac":
                        (status, offset, adjustH) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
                if status == True:
                    break
                else:
                    localDirection = self.directionIncrease(localDirection)
                if localDirection == iniDirection:
                    break
            if status == True:
                if localDirection == 1:
                    offset[0] = offset[0] + imageA.shape[0] - int(i * self.roiRatio * imageA.shape[0])
                elif localDirection == 2:
                    offset[1] = offset[1] + imageA.shape[1] - int(i * self.roiRatio * imageA.shape[1])
                elif localDirection == 3:
                    offset[0] = offset[0] - (imageB.shape[0] - int(i * self.roiRatio * imageB.shape[0]))
                elif localDirection == 4:
                    offset[1] = offset[1] - (imageB.shape[1] - int(i * self.roiRatio * imageB.shape[1]))
                self.direction = localDirection
                break
        if status == False:
            return (status, "  The two images can not match")
        elif status == True:
            self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
            return (status, offset)

    def getStitchByOffset(self, images, offset):
        (imageA, imageB) = images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        dx = offset[0]; dy = offset[1]

        if abs(dy) >= abs(dx):
            direction = "horizontal"
        elif abs(dy) < abs(dx):
            direction = "vertical"

        if dx >= 0 and dy >= 0:
            # The first image is located at the left top, the second image located at the right bottom
            stitchImage = np.zeros((max(hA, dx + hB), max(dy + wB, wA)), dtype=np.uint8)
            roi_ltx = dx; roi_lty = dy
            roi_rbx = min(dx + hB, hA); roi_rby = min(dy + wB, wA)
            stitchImage[0: hA, 0:wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[dx: dx+hB, dy: dy+wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx >= 0 and dy < 0:
            # The first image is located at the right top, the second image located at the left bottom
            stitchImage = np.zeros((max(hA, dx + hB), -dy + wA), dtype=np.uint8)
            roi_ltx = dx; roi_lty = -dy
            roi_rbx = hA;  roi_rby = min(-dy + wA, wB)
            stitchImage[0: hA, -dy:-dy + wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[dx: dx+hB, 0: wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx < 0 and dy >= 0:
            # The first image is located at the left bottom, the second image located at the right top
            stitchImage = np.zeros((max(-dx + hA, hB), max(dy + wB, wA)), dtype=np.uint8)
            roi_ltx = -dx; roi_lty = dy
            roi_rbx = min(-dx + hA, hB);  roi_rby = min(dy + wB, wA)
            stitchImage[-dx: -dx + hA, 0: wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[0: hB, dy: dy + wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx < 0 and dy < 0:
            # The first image is located at the right bottom, the second image located at the left top
            stitchImage = np.zeros((-dx + hA, -dy + wA), dtype=np.uint8)
            roi_ltx = -dx; roi_lty = -dy
            roi_rbx = wA;  roi_rby = hA
            stitchImage[-dx: -dx + hA, -dy: -dy + wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[0: hB, 0: wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        fuseRegion = self.fuseImage([roiImageRegionA, roiImageRegionB])
        stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby] = fuseRegion.copy()
        return (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB)

    def fuseImage(self, images):
        (imageA, imageB) = images
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        imageA[imageA == 0] = imageB[imageA == 0]
        imageB[imageB == 0] = imageA[imageB == 0]
        imageFusion = ImageFusion.ImageFusion()
        if self.fuseMethod == "notFuse":
            fuseRegion = imageB
        elif self.fuseMethod == "average":
            fuseRegion = imageFusion.fuseByAverage(images)
        elif self.fuseMethod == "maximum":
            fuseRegion = imageFusion.fuseByMaximum(images)
        elif self.fuseMethod == "minimum":
            fuseRegion = imageFusion.fuseByMinimum(images)
        elif self.fuseMethod == "fadeInAndFadeOut":
            fuseRegion = imageFusion.fuseByFadeInAndFadeOut(images,self.direction)
        elif self.fuseMethod == "multiBandBlending":
            fuseRegion = imageFusion.fuseByMultiBandBlending(images)
        elif self.fuseMethod == "trigonometric":
            fuseRegion = imageFusion.fuseByTrigonometric(images,self.direction)
        elif self.fuseMethod == "optimalSeamLine":
            fuseRegion = imageFusion.fuseByOptimalSeamLine(images, self.direction)
        return fuseRegion






if __name__=="__main__":
    stitcher = Stitcher()
    imageA = cv2.imread(".\\images\\dendriticCrystal\\1\\1-044.jpg", 0)
    imageB = cv2.imread(".\\images\\dendriticCrystal\\1\\1-045.jpg", 0)
    offset = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])