import numpy as np
import cv2
from scipy.stats import mode
import time
import ImageFusion
import os

class Stitcher:
    '''
	    图像拼接类，包括所有跟材料显微组织图像配准相关函数
	'''
    outputAddress = "result/"
    isEvaluate = False
    evaluateFile = "evaluate.txt"
    isPrintLog = False

    def __init__(self, outputAddress, evaluate, isPrintLog):
        self.outputAddress = outputAddress
        self.isEvaluate = evaluate[0]
        self.evaluateFile = evaluate[1]
        self.isPrintLog = isPrintLog

    def printAndWrite(self, content):
        if self.isPrintLog:
            print(content)
        if self.isEvaluate:
            f = open(self.outputAddress + self.evaluateFile, "a")
            f.write(content)
            f.write("\n")
            f.close()

    def pairwiseStitch(self, fileList, registrateMethod, fuseMethod, direction="horizontal"):
        self.printAndWrite("Stitching " + str(fileList[0]) + " and " + str(fileList[1]))

        imageA = cv2.imread(fileList[0], 0)
        imageB = cv2.imread(fileList[1], 0)
        startTime = time.time()
        (status, offset) = self.calculateOffset([imageA,imageB], registrateMethod, fuseMethod, direction=direction)
        endTime = time.time()

        if status == False:
            self.printAndWrite(result)
        else:
            (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = self.getStitchByOffset([imageA, imageB], offset, fuseMethod=fuseMethod)
            self.printAndWrite("  The time of stitching is " + str(endTime - startTime) + "s")
        return (status, stitchImage)

    def stitchOneColumn(self, startIndex, endIndex, numPixelControl, Files):
        pass

    def stitchOneRow(self, startIndex, endIndex, numPixelControl, Files):
        pass

    def stitchTwoColumns(self, leftColumnAddress, rightColumnAddress, numPixelControl):
        pass

    def stitchTwoRows(self, upColumnAddress, downColumnAddress, numPixelControl):
        pass

    def calculateOffset(self, images, registrateMethod, fuseMethod, direction="horizontal"):
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
        if  registrateMethod[0] == "phaseCorrection":
            return (False, "  We don't develop the phase Correction method, Plesae wait for updating")
        elif  registrateMethod[0] == "featureSearchWithIncrease":
            featureMethod = registrateMethod[1]        # "sift","surf" or "orb"
            searchRatio = registrateMethod[2]          # 0.75 is common value for matches
            offsetCaculate = registrateMethod[3][0]    # "mode" or "ransac"
            offsetEvaluate = registrateMethod[3][1]    # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
            roiFirstLength = registrateMethod[4][0]     # roi length for stitching in first direction
            roiSecondLength = registrateMethod[4][1]    # roi length for stitching in second direction

            if direction == "horizontal":
                maxI = int(imageA.shape[1] / (2 * roiFirstLength))
            elif direction == "vertical":
                maxI = int(imageA.shape[0] / (2 * roiFirstLength))
            for i in range(1, maxI+1):
                # get the roi region of images
                print("  i is: " + str(i))
                roiImageA = self.getROIRegion(imageA, direction=direction, order="first", searchLength=i * roiFirstLength,
                                                  searchLengthForLarge=roiSecondLength)
                roiImageB = self.getROIRegion(imageB, direction=direction, order="second", searchLength=i * roiFirstLength,
                                                  searchLengthForLarge=roiSecondLength)
                # get the feature points
                (kpsA, featuresA) = self.detectAndDescribe(roiImageA, featureMethod=featureMethod)
                (kpsB, featuresB) = self.detectAndDescribe(roiImageB, featureMethod=featureMethod)

                # match all the feature points and return the list of offset
                (dxArray, dyArray) = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, searchRatio)
                if direction == "horizontal":
                        dyArray = dyArray + imageA.shape[1] - i * roiFirstLength
                elif direction == "vertical":
                        dxArray = dxArray + imageA.shape[0] - i * roiFirstLength
                if offsetCaculate == "mode":
                    (status, offset) = self.getOffsetByMode(dxArray, dyArray, evaluateNum=offsetEvaluate)
                elif offsetCaculate == "ransac":
                    return (False, "  We don't develop the stitching with ransac method, Plesae wait for updating")
                if status == True:
                    break
        self.printAndWrite("  The offset of stitching: dx is "+ str(offset[0]) + " dy is " + str(offset[1]))
        return (True, offset)

    def getROIRegion(self, image, direction="horizontal", order="first", searchLength=150, searchLengthForLarge=-1):
        '''对原始图像裁剪感兴趣区域
        :param originalImage:需要裁剪的原始图像
        :param direction:拼接的方向
        :param order:该图片的顺序，是属于第一还是第二张图像
        :param searchLength:搜索区域大小
        :param searchLengthForLarge:对于行拼接和列拼接的搜索区域大小
        :return:返回感兴趣区域图像
        :type searchLength: np.int
        '''
        row, col = image.shape[:2]
        if direction == "horizontal":
            if order == "first":
                if searchLengthForLarge == -1:
                    roiRegion = image[:, col - searchLength:col]
                elif searchLengthForLarge > 0:
                    roiRegion = image[row - searchLengthForLarge:row, col - searchLength:col]
            elif order == "second":
                if searchLengthForLarge == -1:
                    roiRegion = image[:, 0: searchLength]
                elif searchLengthForLarge > 0:
                    roiRegion = image[0:searchLengthForLarge, 0: searchLength]
        elif direction == "vertical":
            if order == "first":
                if searchLengthForLarge == -1:
                    roiRegion = image[row - searchLength:row, :]
                elif searchLengthForLarge > 0:
                    roiRegion = image[row - searchLength:row, col - searchLengthForLarge:col]
            elif order == "second":
                if searchLengthForLarge == -1:
                    roiRegion = image[0: searchLength, :]
                elif searchLengthForLarge > 0:
                    roiRegion = image[0: searchLength, 0:searchLengthForLarge]
        return roiRegion

    def detectAndDescribe(self, image, featureMethod):
        '''
    	计算图像的特征点集合，并返回该点集＆描述特征
    	:param image:需要分析的图像
    	:return:返回特征点集，及对应的描述特征
    	'''
        # 将彩色图片转换成灰度图
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        if featureMethod == "sift":
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif featureMethod == "surf":
            descriptor = cv2.xfeatures2d.SURF_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio):
        '''
        匹配特征点
        :param self:
        :param featuresA: 第一张图像的特征点描述符
        :param featuresB: 第二张图像的特征点描述符
        :param ratio: 最近邻和次近邻的比例
        :return:返回匹配的对数
        '''
        # 建立暴力匹配器
        matcher = cv2.DescriptorMatcher_create("BruteForce")

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        self.printAndWrite("  The number of matching is " + str(len(matches)))
        dxList = []; dyList = []
        # # 获取输入图片及其搜索区域
        # (imageA, imageB) = images
        # (hA, wA) = imageA.shape[:2]
        # (hB, wB) = imageB.shape[:2]
        for trainIdx, queryIdx in matches:
            ptA = (int(kpsA[queryIdx][1]), int(kpsA[queryIdx][0]))
            ptB = (int(kpsB[trainIdx][1]), int(kpsB[trainIdx][0]))
            # if direction == "horizontal":
            dxList.append(ptA[0] - ptB[0])
            dyList.append(ptA[1] - ptB[1])
            # elif direction == "vertical":
            #     dx.append((hA - ptA[0]) + ptB[0])
            #     dy.append(ptA[1] - ptB[1])
        return (np.array(dxList) ,np.array(dyList))

    def getOffsetByMode(self, dxArray, dyArray, evaluateNum=20):
        if len(dxArray) < evaluateNum:
            return (False, (0, 0))
        else:
            return (True, (int(mode(dxArray, axis=None)[0]), int(mode(dyArray, axis=None)[0])))

    def creatOffsetImage(self, image, direction, offset):
        h, w = image.shape[:2]
        returnImage = np.zeros(image.shape,dtype=np.uint8)
        if direction == "horizontal":
            if offset >= 0:
                returnImage[0:h-offset,:] = image[offset: h, :]
            elif offset < 0:
                returnImage[(-1 * offset):h, :] = image[0:h+offset,:]
        elif direction == "vertical":
            if offset >= 0:
                returnImage[:, 0:w-offset] = image[:, offset:w]
            elif offset < 0:
                returnImage[:, (-1 * offset):w] = image[:, 0:w+offset]
        return returnImage

    def getStitchByOffset(self, images, offset, fuseMethod):
        (imageA, imageB) = images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        dx = offset[0]; dy = offset[1]

        if abs(dy) >= abs(dx):
            direction = "horizontal"
        elif abs(dy) < abs(dx):
            direction = "vertical"

        if dx >= 0 and dy >= 0:
            print("here")
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
            stitchImage[dx: dx+wB, 0: wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx < 0 and dy >= 0:
            # The first image is located at the left bottom, the second image located at the right top
            stitchImage = np.zeros((-dx + hA, max(dy + wB, wA)), dtype=np.uint8)
            roi_ltx = -dx; roi_lty = dy
            roi_rbx = min(-dx + wA, wB);  roi_rby = min(dy + hB, hA)
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
        # print(roiImageRegionA.shape)
        # print(roiImageRegionB.shape)
        # cv2.imshow("imageA", roiImageRegionA)
        # cv2.imshow("imageB", roiImageRegionB)
        # cv2.waitKey(0)
        fuseRegion = self.fuseImage([roiImageRegionA, roiImageRegionB], direction=direction, fuseMethod=fuseMethod)
        stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby] = fuseRegion.copy()
        return (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB)

        #     if dy > 0 and dx < 0:
        #         cutImageA = self.creatOffsetImage(imageA, direction, dx)
        #         stitchImage = np.hstack((cutImageA[:, 0:dy], imageB[0:hA+dx, :]))
        #         if fuseMethod[0] != "notFuse":
        #             fuseRegion = self.fuseImage([cutImageA[0:hA+dx, dy:wA], imageB[0:hA+dx, 0:dy]], direction=direction,
        #                                          fuseMethod=fuseMethod)
        #             stitchImage[:, dy:wA] = fuseRegion[:]
        #     elif dy > 0 and dx >= 0:
        #         cutImageA = self.creatOffsetImage(imageA, direction, dx)
        #         stitchImage = np.hstack((cutImageA[0:hA-dx, 0:dy], imageB[0:hA-dx, :]))
        #         # 判断是否对图像进行融合
        #         if fuseMethod[0] != "notFuse":
        #             fuseRegion = self.fuseImage([cutImageA[0:hA-dx, dy:wA], imageB[0:hA-dx,0:dy]], direction=direction,
        #                                          fuseMethod=fuseMethod)
        #             stitchImage[:, dy:wA] = fuseRegion[:]
        # elif abs(dx) > abs(dy):           # stitch in vertical direction
        #     if dx > 0 and dy <= 0:
        #         cutImageA = self.creatOffsetImage(imageA, direction, dy)
        #         stitchImage = np.vstack((cutImageA[0:hA-dx, (-dy):wA], imageB[:, 0:wB+dy]))
        #         if fuseMethod[0] != "notFuse":
        #             fuseRegion = self.fuseImage([cutImageA[hA-dx:hA, (-dy):wA], imageB[0:dx, 0:wB+dy]], direction=direction,
        #                                          fuseMethod=fuseMethod)
        #             stitchImage[hA-dx:hA, :] = fuseRegion[:]
        #     elif dx > 0 and dy >= 0:
        #         cutImageA = self.creatOffsetImage(imageA, direction, dy)
        #         stitchImage = np.vstack((cutImageA[0:hA-dx, dy:wA], imageB[:, 0:wB-dy]))
        #         if fuseMethod[0] != "notFuse":
        #             fuseRegion = self.fuseImage([cutImageA[hA-dx:hA, dy:wA], imageB[0:dx, 0:wB-dy]], direction=direction,
        #                                          fuseMethod=fuseMethod)
        #             stitchImage[hA - dx:hA, :] = fuseRegion[:]
        # return stitchImage

    def fuseImage(self, images, fuseMethod, direction="horizontal"):
        (imageA, imageB) = images
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        if fuseMethod[0] == "notFuse":
            fuseRegion = imageB
        elif fuseMethod[0] == "average":
            fuseRegion = ImageFusion.fuseByAverage(images)
        elif fuseMethod[0] == "maximum":
            fuseRegion = ImageFusion.fuseByMaximum(images)
        elif fuseMethod[0] == "minimum":
            fuseRegion = ImageFusion.fuseByMinimum(images)
        elif fuseMethod[0] == "fadeInAndFadeOut":
            fuseRegion = ImageFusion.fuseByFadeInAndFadeOut(images,direction)
        elif fuseMethod[0] == "multiBandBlending":
            fuseRegion = ImageFusion.fuseByMultiBandBlending(images)
        elif fuseMethod[0] == "trigonometric":
            fuseRegion = ImageFusion.fuseByTrigonometric(images,direction)
        elif fuseMethod[0] == "optimalSeamLine":
            fuseRegion = ImageFusion.fuseByOptimalSeamLine(images, direction)
        return fuseRegion

if __name__=="__main__":
    outputAddress = "result/"
    evaluate = (True, "evaluate.txt")
    isPrintLog = True
    stitcher = Stitcher(outputAddress, evaluate, isPrintLog)
    fileList = ["images/dendriticCrystal/iron/1-001.jpg", "images/dendriticCrystal/iron/1-002.jpg"]
    # fileList = ["images/dendriticCrystal/1/1-030.jpg", "images/dendriticCrystal/1/1-031.jpg"]
    registrateMethod = ("featureSearchWithIncrease", "surf", 0.75, ("mode", 100),(150, -1))
    fuseMethod = (True, "fadeInAndFadeOut")
    (status, result) = stitcher.pairwiseStitch(fileList, registrateMethod, fuseMethod, direction="vertical")
    cv2.namedWindow("Result", 0)
    cv2.imshow("Result", result)
    cv2.imwrite("Result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()