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
import myGpuSurf

class ImageFeature():
    # 用来保存串行全局拼接中的第二张图像的特征点和描述子，为后续加速拼接使用
    isBreak = True      # 判断是否上一次中断
    kps = None
    feature = None


class Stitcher(Utility.Method):
    '''
	    图像拼接类，包括所有跟材料显微组织图像配准相关函数
	'''
    direction = 1               # 1： 第一张图像在上，第二张图像在下；   2： 第一张图像在左，第二张图像在右；
                                # 3： 第一张图像在下，第二张图像在上；   4： 第一张图像在右，第二张图像在左；
    directIncre = 1
    featureMethod = "surf"      # "sift","surf" or "orb"
    searchRatio = 0.75          # 0.75 is common value for matches
    offsetCaculate = "mode"     # "mode" or "ransac"
    offsetEvaluate = 10         # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
    roiRatio = 0.1              # roi length for stitching in first direction
    fuseMethod = "notFuse"
    isEnhance = False
    isClahe = False
    clipLimit = 20
    tileSize = 5
    phaseResponseThreshold = 0.15
    overlapRatio = []
    tempImageFeature = ImageFeature()
    isGPUAvailable = True
    keypointsRatio = 0.005

    def npToListForKeypoints(self, array):
        '''
        Convert array to List, used for keypoints from GPUDLL to python List
        :param array: array from GPUDLL
        :return:
        '''
        kps = []
        row, col = array.shape
        for i in range(row):
            kps.append([array[i, 0], array[i, 1]])
        return kps

    def npToListForMatches(self, array):
        '''
        Convert array to List, used for DMatches from GPUDLL to python List
        :param array: array from GPUDLL
        :return:
        '''
        descritpors = []
        row, col = array.shape
        for i in range(row):
            descritpors.append((array[i, 0], array[i, 1]))
        return descritpors

    def directionIncrease(self, direction):
        direction += self.directIncre
        if direction == 5:
            direction = 1
        if direction == 0:
            direction = 4
        return direction

    def flowStitch(self, fileList, caculateOffsetMethod):
        self.printAndWrite("Stitching the directory which have " + str(fileList[0]))
        fileNum = len(fileList)
        offsetList = []
        describtion = ""
        # calculating the offset for small image
        startTime = time.time()
        status = True
        endfileIndex = 0
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
                endfileIndex = fileIndex + 1
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
        return ((status, endfileIndex), stitchImage)

    def flowStitchWithMutiple(self, fileList, caculateOffsetMethod):
        result = []
        totalNum = len(fileList)
        startNum = 0
        while 1:
            (status, stitchResult) = self.flowStitch(fileList[startNum: totalNum], caculateOffsetMethod)
            result.append(stitchResult)
            self.tempImageFeature.isBreak = True
            if status[1] == 1:
                startNum = startNum + status[1] + 1
            else:
                startNum = startNum + status[1] + 1

            self.printAndWrite("status[1] = " + str(status[1]))
            self.printAndWrite("startNum = "+str(startNum))
            if startNum == totalNum:
                break
            if startNum == (totalNum - 1):
                result.append(cv2.imread(fileList[startNum], 0))
                break
            self.printAndWrite("stitching Break, start from " + str(fileList[startNum]) + " again")
        return result

    def imageSetStitch(self, projectAddress, outputAddress, fileNum, caculateOffsetMethod, startNum = 1, fileExtension = "jpg", outputfileExtension = "jpg"):
        for i in range(startNum, fileNum+1):
            fileAddress = projectAddress + "\\" + str(i) + "\\"
            fileList = glob.glob(fileAddress + "*." + fileExtension)
            if not os.path.exists(outputAddress):
                os.makedirs(outputAddress)
            Stitcher.outputAddress = outputAddress
            (status, result) = self.flowStitch(fileList, caculateOffsetMethod)
            self.tempImageFeature.isBreak = True
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "." + outputfileExtension, result)
            if status == False:
                self.printAndWrite("stitching Failed")

    def imageSetStitchWithMutiple(self, projectAddress, outputAddress, fileNum, caculateOffsetMethod, startNum = 1, fileExtension = "jpg", outputfileExtension = "jpg"):
        for i in range(startNum, fileNum+1):
            fileAddress = projectAddress + "\\" + str(i) + "\\"
            fileList = glob.glob(fileAddress + "*." + fileExtension)
            if not os.path.exists(outputAddress):
                os.makedirs(outputAddress)
            Stitcher.outputAddress = outputAddress
            result = self.flowStitchWithMutiple(fileList, caculateOffsetMethod)
            self.tempImageFeature.isBreak = True
            if len(result) == 1:
                cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "." + outputfileExtension, result[0])
                # cv2.imwrite(outputAddress + "\\" + outputName + "." + outputfileExtension, result[0])
            else:
                for j in range(0, len(result)):
                    cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "_" + str(j+1) + "." + outputfileExtension, result[j])
                    # cv2.imwrite(outputAddress + "\\" + outputName + "_" + str(j + 1) + "." + outputfileExtension,result[j])

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

    def calculateOffsetForPhaseCorrleateIncre(self, images):
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

                # hann = cv2.createHanningWindow(winSize=(roiImageA.shape[1], roiImageA.shape[0]), type=5)
                # (offsetTemp, response) = cv2.phaseCorrelate(np.float32(roiImageA), np.float32(roiImageB), window=hann)
                (offsetTemp, response) = cv2.phaseCorrelate(np.float64(roiImageA), np.float64(roiImageB))
                offset[0] = np.int(offsetTemp[1])
                offset[1] = np.int(offsetTemp[0])
                self.printAndWrite("offset: " + str(offset))
                self.printAndWrite("respnse: " + str(response))
                if response > self.phaseResponseThreshold:
                    status = True
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
        if self.tempImageFeature.isBreak == True:
            if self.isGPUAvailable == True:
                myGpuSurf.matchFeaturesBySurf(imageA, imageB, self.keypointsRatio, self.searchRatio)
                kpsA = self.npToListForKeypoints(myGpuSurf.getImageAKeyPoints())
                featuresA = myGpuSurf.getImageADescriptors()
                kpsB = self.npToListForKeypoints(myGpuSurf.getImageBKeyPoints())
                featuresB = myGpuSurf.getImageBDescriptors()
            else:
                (kpsA, featuresA) = self.detectAndDescribe(imageA, featureMethod=self.featureMethod)
                (kpsB, featuresB) = self.detectAndDescribe(imageB, featureMethod=self.featureMethod)
            self.tempImageFeature.isBreak = False
            self.tempImageFeature.kps = kpsB
            self.tempImageFeature.feature = featuresB
        else:
            kpsA = self.tempImageFeature.kps
            featuresA = self.tempImageFeature.feature
            if self.isGPUAvailable == True:
                myGpuSurf.matchFeaturesBySurf(imageA, imageB, self.searchRatio)
                kpsB = self.npToListForKeypoints(myGpuSurf.getImageBKeyPoints())
                featuresB = myGpuSurf.getImageBDescriptors()
            else:
                (kpsB, featuresB) = self.detectAndDescribe(imageB, featureMethod=self.featureMethod)
            self.tempImageFeature.isBreak = False
            self.tempImageFeature.kps = kpsB
            self.tempImageFeature.feature = featuresB
        if featuresA is not None and featuresB is not None:
            if self.isGPUAvailable == True:
                matches = self.npToListForMatches(myGpuSurf.getGoodMatches())
            else:
                matches = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, self.searchRatio)
            # match all the feature points
            if self.offsetCaculate == "mode":
                (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
            elif self.offsetCaculate == "ransac":
                (status, offset, adjustH) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
        if status == False:
            self.tempImageFeature.isBreak = True
            return (status, "  The two images can not match")
        elif status == True:
            self.tempImageFeature.isBreak = False
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
                if self.isGPUAvailable == True:
                    myGpuSurf.matchFeaturesBySurf(roiImageA, roiImageB, self.keypointsRatio, self.searchRatio)
                    kpsA = self.npToListForKeypoints(myGpuSurf.getImageAKeyPoints())
                    featuresA = myGpuSurf.getImageADescriptors()
                    kpsB = self.npToListForKeypoints(myGpuSurf.getImageBKeyPoints())
                    featuresB = myGpuSurf.getImageBDescriptors()
                else:
                    (kpsA, featuresA) = self.detectAndDescribe(roiImageA, featureMethod=self.featureMethod)
                    (kpsB, featuresB) = self.detectAndDescribe(roiImageB, featureMethod=self.featureMethod)
                if featuresA is not None and featuresB is not None:
                    if self.isGPUAvailable == True:
                        matches = self.npToListForMatches(myGpuSurf.getGoodMatches())
                    else:
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
        mask = np.zeros(imageB.shape, dtype=np.uint8)

        if dx >= 0 and dy >= 0:
            # The first image is located at the left top, the second image located at the right bottom
            stitchImage = np.zeros((max(hA, dx + hB), max(dy + wB, wA)), dtype=np.int)-1
            roi_ltx = dx; roi_lty = dy
            roi_rbx = min(dx + hB, hA); roi_rby = min(dy + wB, wA)
            stitchImage[0: hA, 0:wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[dx: dx+hB, dy: dy+wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx >= 0 and dy < 0:
            # The first image is located at the right top, the second image located at the left bottom
            stitchImage = np.zeros((max(hA, dx + hB), -dy + wA), dtype=np.int)-1
            roi_ltx = dx;  roi_lty = -dy
            roi_rbx = hA;  roi_rby = min(-dy + wA, wB)
            stitchImage[0: hA, -dy:-dy + wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[dx: dx+hB, 0: wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx < 0 and dy >= 0:
            # The first image is located at the left bottom, the second image located at the right top
            stitchImage = np.zeros((max(-dx + hA, hB), max(dy + wB, wA)), dtype=np.int)-1
            roi_ltx = -dx; roi_lty = dy
            roi_rbx = min(-dx + hA, hB);  roi_rby = min(dy + wB, wA)
            stitchImage[-dx: -dx + hA, 0: wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[0: hB, dy: dy + wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx < 0 and dy < 0:
            # The first image is located at the right bottom, the second image located at the left top
            stitchImage = np.zeros((-dx + hA, -dy + wA), dtype=np.int)-1
            roi_ltx = -dx; roi_lty = - dy
            roi_rbx = hB;  roi_rby = wB
            stitchImage[-dx: -dx + hA, -dy: -dy + wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[0: hB, 0: wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        # cv2.imshow("roiImageRegionA", roiImageRegionA)
        # cv2.imshow("roiImageRegionB", roiImageRegionB)
        # cv2.waitKey(0)
        fuseRegion = self.fuseImage([roiImageRegionA, roiImageRegionB], dx, dy)
        stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby] = fuseRegion.copy()
        stitchImage[stitchImage == -1] = 0
        stitchImage = stitchImage.astype(np.uint8)
        return (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB)

    def fuseImage(self, images, dx, dy):
        (imageA, imageB) = images
        # cv2.imshow("A", imageA)
        # cv2.imshow("B", imageB)
        # cv2.waitKey(0)
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        # imageA[imageA == 0] = imageB[imageA == 0]
        # imageB[imageB == 0] = imageA[imageB == 0]
        imageFusion = ImageFusion.ImageFusion()
        if self.fuseMethod == "notFuse":
            imageB[imageA == -1] = imageB[imageA == -1]
            imageA[imageB == -1] = imageA[imageB == -1]
            fuseRegion = imageB
        elif self.fuseMethod == "average":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByAverage([imageA, imageB])
        elif self.fuseMethod == "maximum":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByMaximum([imageA, imageB])
        elif self.fuseMethod == "minimum":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            fuseRegion = imageFusion.fuseByMinimum([imageA, imageB])
        elif self.fuseMethod == "fadeInAndFadeOut":
            fuseRegion = imageFusion.fuseByFadeInAndFadeOut(images, dx, dy)
        elif self.fuseMethod == "trigonometric":
            fuseRegion = imageFusion.fuseByTrigonometric(images, dx, dy)
        elif self.fuseMethod == "multiBandBlending":
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            # imageA = imageA.astye(np.uint8);  imageB = imageB.astye(np.uint8);
            fuseRegion = imageFusion.fuseByMultiBandBlending([imageA, imageB])
        elif self.fuseMethod == "optimalSeamLine":
            fuseRegion = imageFusion.fuseByOptimalSeamLine(images, self.direction)
        return fuseRegion


if __name__=="__main__":
    stitcher = Stitcher()
    imageA = cv2.imread(".\\images\\dendriticCrystal\\1\\1-044.jpg", 0)
    imageB = cv2.imread(".\\images\\dendriticCrystal\\1\\1-045.jpg", 0)
    offset = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])