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
import time

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
    fuseMethod = "notFuse"
    phaseResponseThreshold = 0.15
    phase = phaseCorrelation()
    tempImageFeature = ImageFeature()

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
            # imageA = cv2.imread(fileList[fileIndex], 0)
            # imageB = cv2.imread(fileList[fileIndex + 1], 0)
            imageA = cv2.imdecode(np.fromfile(fileList[fileIndex], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            imageB = cv2.imdecode(np.fromfile(fileList[fileIndex + 1], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
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

        # stitching and fusing
        self.printAndWrite("start stitching")
        startTime = time.time()
        # offsetList = [[1784, 2], [1805, 2], [1809, 2], [1775, 2], [1760, 2], [1846, 2], [1809, 1], [1812, 2], [1786, 1], [1818, 3], [1786, 2], [1802, 2], [1722, 1], [1211, 1], [-10, 2411], [-1734, -1], [-1808, -1], [-1788, -3], [-1754, -1], [-1727, -2], [-1790, -3], [-1785, -2], [-1778, -1], [-1807, -2], [-1767, -2], [-1822, -3], [-1677, -2], [-1778, -2], [-1440, -1], [-2, 2410], [1758, 2], [1792, 2], [1794, 2], [1840, 3], [1782, 2], [1802, 3], [1782, 2], [1763, 3], [1738, 2], [1837, 3], [1781, 2], [1788, 18], [1712, 0], [1271, -11], [-3, 2478], [-1787, -1], [-1812, -2], [-1822, -2], [-1762, -1], [-1725, -2], [-1884, -2], [-1754, -2], [-1747, -1], [-1666, -1], [-1874, -3], [-1695, -2], [-1672, -1], [-1816, -2], [-1411, -1], [-4, 2431], [1874, 3], [1706, -3], [1782, 2], [1794, 2], [1732, 3], [1838, 3], [1721, 1], [1783, 3], [1805, 2], [1725, 3], [1828, 1], [1774, 3], [1776, 1], [1201, 1], [-16, 2405], [-1821, 0], [-1843, -2], [-1758, -2], [-1742, -3], [-1814, -2], [-1817, -2], [-1848, -2], [-1768, -2], [-1749, -2], [-1765, -2], [-1659, -2], [-1832, -2], [-1791, -2], [-1197, -1]]
        stitchImage = self.getStitchByOffset(fileList, offsetList)
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

            # self.printAndWrite("status[1] = " + str(status[1]))
            # self.printAndWrite("startNum = "+str(startNum))
            if startNum == totalNum:
                break
            if startNum == (totalNum - 1):
                # result.append(cv2.imread(fileList[startNum], 0))
                result.append(cv2.imdecode(np.fromfile(fileList[startNum], dtype=np.uint8), cv2.IMREAD_GRAYSCALE))
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
            startTime = time.time()
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
            endTime = time.time()
            print("Time Consuming for " + fileAddress + " is " + str(endTime - startTime))

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
            # self.printAndWrite("  i=" + str(i) + " and maxI="+str(maxI))
            while(True):
                # get the roi region of images
                # self.printAndWrite("  localDirection=" + str(localDirection))
                roiImageA = self.getROIRegionForIncreMethod(imageA, direction=localDirection, order="first", searchRatio = i * self.roiRatio)
                roiImageB = self.getROIRegionForIncreMethod(imageB, direction=localDirection, order="second", searchRatio = i * self.roiRatio)

                # hann = cv2.createHanningWindow(winSize=(roiImageA.shape[1], roiImageA.shape[0]), type=5)
                # (offsetTemp, response) = cv2.phaseCorrelate(np.float32(roiImageA), np.float32(roiImageB), window=hann)
                (offsetTemp, response) = cv2.phaseCorrelate(np.float64(roiImageA), np.float64(roiImageB))
                offset[0] = np.int(offsetTemp[1])
                offset[1] = np.int(offsetTemp[0])
                # self.printAndWrite("offset: " + str(offset))
                # self.printAndWrite("respnse: " + str(response))
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
            (kpsA, featuresA) = self.detectAndDescribe(imageA, featureMethod=self.featureMethod)
            (kpsB, featuresB) = self.detectAndDescribe(imageB, featureMethod=self.featureMethod)
            self.tempImageFeature.isBreak = False
            self.tempImageFeature.kps = kpsB
            self.tempImageFeature.feature = featuresB
        else:
            kpsA = self.tempImageFeature.kps
            featuresA = self.tempImageFeature.feature
            (kpsB, featuresB) = self.detectAndDescribe(imageB, featureMethod=self.featureMethod)
            self.tempImageFeature.isBreak = False
            self.tempImageFeature.kps = kpsB
            self.tempImageFeature.feature = featuresB
        if featuresA is not None and featuresB is not None:
            matches = self.matchDescriptors(featuresA, featuresB)
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
            # self.printAndWrite("  i=" + str(i) + " and maxI="+str(maxI))
            while(True):
                # get the roi region of images
                # self.printAndWrite("  localDirection=" + str(localDirection))
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
                kpsA, featuresA = self.detectAndDescribe(roiImageA, featureMethod=self.featureMethod)
                kpsB, featuresB = self.detectAndDescribe(roiImageB, featureMethod=self.featureMethod)
                if featuresA is not None and featuresB is not None:
                    matches = self.matchDescriptors(featuresA, featuresB)
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

    def getStitchByOffset(self, fileList, offsetListOrigin):
        '''
        通过偏移量列表和文件列表得到最终的拼接结果
        :param fileList: 图像列表
        :param offsetListOrigin: 偏移量列表
        :return: ndaarry，图像
        '''
        # 如果你不细心，不要碰这段代码
        # 已优化到根据指针来控制拼接，CPU下最快了
        dxSum = dySum = 0
        imageList = []
        # imageList.append(cv2.imread(fileList[0], 0))
        imageList.append(cv2.imdecode(np.fromfile(fileList[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE))
        resultRow = imageList[0].shape[0]         # 拼接最终结果的横轴长度,先赋值第一个图像的横轴
        resultCol = imageList[0].shape[1]         # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴
        offsetListOrigin.insert(0, [0, 0])        # 增加第一张图像相对于最终结果的原点的偏移量

        rangeX = [[0,0] for x in range(len(offsetListOrigin))]  # 主要用于记录X方向最大最小边界
        rangeY = [[0, 0] for x in range(len(offsetListOrigin))] # 主要用于记录Y方向最大最小边界
        offsetList = offsetListOrigin.copy()
        rangeX[0][1] = imageList[0].shape[0]
        rangeY[0][1] = imageList[0].shape[1]

        for i in range(1, len(offsetList)):
            # self.printAndWrite("  stitching " + str(fileList[i]))
            # 适用于流形拼接的校正,并更新最终图像大小
            # tempImage = cv2.imread(fileList[i], 0)
            tempImage = cv2.imdecode(np.fromfile(fileList[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            dxSum = dxSum + offsetList[i][0]
            dySum = dySum + offsetList[i][1]
            # self.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
            if dxSum <= 0:
                for j in range(0, i):
                    offsetList[j][0] = offsetList[j][0] + abs(dxSum)
                    rangeX[j][0] = rangeX[j][0] + abs(dxSum)
                    rangeX[j][1] = rangeX[j][1] + abs(dxSum)
                resultRow = resultRow + abs(dxSum)
                rangeX[i][1] = resultRow
                dxSum = rangeX[i][0] = offsetList[i][0] = 0
            else:
                offsetList[i][0] = dxSum
                resultRow = max(resultRow, dxSum + tempImage.shape[0])
                rangeX[i][1] = resultRow
            if dySum <= 0:
                for j in range(0, i):
                    offsetList[j][1] = offsetList[j][1] + abs(dySum)
                    rangeY[j][0] = rangeY[j][0] + abs(dySum)
                    rangeY[j][1] = rangeY[j][1] + abs(dySum)
                resultCol = resultCol + abs(dySum)
                rangeY[i][1] = resultCol
                dySum = rangeY[i][0] = offsetList[i][1] = 0
            else:
                offsetList[i][1] = dySum
                resultCol = max(resultCol, dySum + tempImage.shape[1])
                rangeY[i][1] = resultCol
            imageList.append(tempImage)
        stitchResult = np.zeros((resultRow, resultCol), np.int) - 1
        self.printAndWrite("  The rectified offsetList is " + str(offsetList))
        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值
        for i in range(0, len(offsetList)):
            self.printAndWrite("  stitching " + str(fileList[i]))
            if i == 0:
                stitchResult[offsetList[0][0]: offsetList[0][0] + imageList[0].shape[0], offsetList[0][1]: offsetList[0][1] + imageList[0].shape[1]] = imageList[0]
            else:
                if self.fuseMethod == "notFuse":
                    # 适用于无图像融合，直接覆盖
                    # self.printAndWrite("Stitch " + str(i+1) + "th, the roi_ltx is " + str(offsetList[i][0]) + " and the roi_lty is " + str(offsetList[i][1]))
                    stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                else:
                    # 适用于图像融合算法，切出 roiA 和 roiB 供图像融合
                    minOccupyX = rangeX[i-1][0]
                    maxOccupyX = rangeX[i-1][1]
                    minOccupyY = rangeY[i-1][0]
                    maxOccupyY = rangeY[i-1][1]
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the offsetList[i][0] is " + str(
                    #     offsetList[i][0]) + " and the offsetList[i][1] is " + str(offsetList[i][1]))
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the minOccupyX is " + str(
                    #     minOccupyX) + " and the maxOccupyX is " + str(maxOccupyX) + " and the minOccupyY is " + str(
                    #     minOccupyY) + " and the maxOccupyY is " + str(maxOccupyY))
                    roi_ltx = max(offsetList[i][0], minOccupyX)
                    roi_lty = max(offsetList[i][1], minOccupyY)
                    roi_rbx = min(offsetList[i][0] + imageList[i].shape[0], maxOccupyX)
                    roi_rby = min(offsetList[i][1] + imageList[i].shape[1], maxOccupyY)
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the roi_ltx is " + str(
                    #     roi_ltx) + " and the roi_lty is " + str(roi_lty) + " and the roi_rbx is " + str(
                    #     roi_rbx) + " and the roi_rby is " + str(roi_rby))
                    roiImageRegionA = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                    roiImageRegionB = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                    stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuseImage([roiImageRegionA, roiImageRegionB], offsetListOrigin[i][0], offsetListOrigin[i][1])
        stitchResult[stitchResult == -1] = 0
        return stitchResult.astype(np.uint8)

    def fuseImage(self, images, dx, dy):
        (imageA, imageB) = images
        # cv2.namedWindow("A", 0)
        # cv2.namedWindow("B", 0)
        # cv2.imshow("A", imageA.astype(np.uint8))
        # cv2.imshow("B", imageB.astype(np.uint8))
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