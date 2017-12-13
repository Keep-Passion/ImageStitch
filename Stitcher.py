import numpy as np
import cv2
from scipy.stats import mode
import time
import ImageFusion
import os
import skimage.measure
from numba import jit

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
        (status, offset, H) = self.calculateOffset([imageA,imageB], registrateMethod, direction=direction)
        if registrateMethod[3][0] == "ransac":
            imageA = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
            (status, offset, H) = self.calculateOffset([imageA, imageB], registrateMethod,direction=direction)
            # print(H)
            # cv2.namedWindow("imageA",0)
            # cv2.imshow("imageA", imageA)
            # cv2.waitKey(0)
        endTime = time.time()

        if status == False:
            self.printAndWrite(offset)
            return (status, offset)
        else:
            self.printAndWrite("  The time of registering is " + str(endTime - startTime) + "s")
            startTime = time.time()
            (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = self.getStitchByOffset([imageA, imageB], offset, fuseMethod=fuseMethod)
            endTime = time.time()
            self.printAndWrite("  The time of fusing is " + str(endTime - startTime) + "s")
            return (status, stitchImage)

    # @jit
    def gridStitch(self, fileList, filePosition, registrateMethod, fuseMethod, shootOrder="snakeByCol"):
        largeBlockNum = len(filePosition)
        self.printAndWrite("Stitching the directory which have" + str(fileList[0]))
        offsetList = []
        # calculating the offset for small image
        startTime = time.time()
        for i in range(0, largeBlockNum):
            offsetList.append([])
            if i % 2 == 0:
                indexStart = filePosition[i][0]; indexEnd = filePosition[i][1]; indexCrement = 1
            elif i % 2 == 1:
                indexStart = filePosition[i][1]; indexEnd = filePosition[i][0]; indexCrement = -1
            for fileIndex in range(indexStart, indexEnd, indexCrement):
                self.printAndWrite("stitching" + str(fileList[fileIndex - 1]) + " and " + str(fileList[fileIndex + indexCrement - 1]))
                imageA = cv2.imread(fileList[fileIndex - 1], 0)
                imageB = cv2.imread(fileList[fileIndex + indexCrement - 1], 0)
                if shootOrder == "snakeByCol":
                    (status, offset, H) = self.calculateOffset([imageA, imageB], registrateMethod, direction="vertical")
                elif shootOrder == "snakeByRow":
                    (status, offset, H) = self.calculateOffset([imageA, imageB], registrateMethod, direction="horizontal")
                if status == False:
                    return (False, "  " + str(fileList[fileIndex - 1]) + " and "+ str(fileList[fileIndex + indexCrement - 1]) + str(offset))
                else:
                    offsetList[i].append(offset)

        # calculating the offset for big block
        # filePosition = [[1, 15], [16, 30], [31, 45], [46, 60], [61, 75], [76, 90]]
        self.printAndWrite("register large block")
        offsetBlockList = []
        for i in range(0, largeBlockNum - 1):
            if i % 2 == 0:
                indexA = filePosition[i][0]; indexB = filePosition[i + 1][1];
            elif i % 2 == 1:
                indexA = filePosition[i][1]; indexB = filePosition[i + 1][0];
            print("stitching " + str(fileList[indexA - 1]) + " and " + str(fileList[indexB - 1]))
            imageA = cv2.imread(fileList[indexA - 1], 0)
            imageB = cv2.imread(fileList[indexB - 1], 0)
            if shootOrder == "snakeByCol":
                (status, offset, H) = self.calculateOffset([imageA, imageB], registrateMethod,
                                                           direction="horizontal")
            elif shootOrder == "snakeByRow":
                (status, offset, H) = self.calculateOffset([imageA, imageB], registrateMethod,
                                                        direction="vertial")
            if status == False:
                return (False, "  Stitching the large block " + str(fileList[indexA - 1]) + " and "+ str(fileList[indexB - 1]) + str(offset))
            else:
                offsetBlockList.append(offset)
        endTime = time.time()
        self.printAndWrite("  The time of registing is " + str(endTime - startTime) + "s")
        # print(offsetList)
        # print(offsetBlockList)

        # stitching and fusing
        # stiching One block
        self.printAndWrite("start stitching")
        startTime = time.time()
        largeBlcokImage = []
        for i in range(0, largeBlockNum):
            # print(" one block:" + str(i))
            indexNum = len(offsetList[i])
            if i % 2 == 0:
                indexStart = filePosition[i][0];indexEnd = filePosition[i][1];indexCrement = 1
            elif i % 2 == 1:
                indexStart = filePosition[i][1];indexEnd = filePosition[i][0];indexCrement = -1
            count = 0
            dxSum = 0; dySum = 0

            stitchImage = cv2.imread(fileList[indexStart - 1], 0)
            for fileIndex in range(indexStart, indexEnd, indexCrement):
                imageA = cv2.imread(fileList[fileIndex - 1], 0)
                imageB = cv2.imread(fileList[fileIndex + indexCrement - 1], 0)
                dxSum = offsetList[i][count][0] + dxSum
                dySum = offsetList[i][count][1] + dySum
                offset = [dxSum,  dySum]
                # offset = [offsetList[i][count][0], offsetList[i][count][1]]
                (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = self.getStitchByOffset([stitchImage, imageB], offset, fuseMethod)
                if offsetList[i][count][1] < 0:
                    dySum = dySum - offsetList[i][count][1]
                count = count + 1
            # cv2.imwrite(self.outputAddress + "\\" + str(i) + ".jpg", stitchImage)
            largeBlcokImage.append(stitchImage)

        # stiching multi block Image
        totalStitch = largeBlcokImage[0]
        dxSum = 0; dySum = 0;
        for i in range(0, largeBlockNum - 1):
            # print(" big block:" + str(i))
            imageB = largeBlcokImage[i + 1]
            dxSum = offsetBlockList[i][0] + dxSum
            dySum = offsetBlockList[i][1] + dySum
            offset = [dxSum, dySum]
            (totalStitch, fuseRegion, roiImageRegionA, roiImageRegionB) = self.getStitchByOffset([totalStitch, imageB],
                                                                                                 offset, fuseMethod)
            if offsetBlockList[i][0] < 0:
                dxSum = dxSum - offsetBlockList[i][0]
        endTime = time.time()
        self.printAndWrite("  The time of fusing is " + str(endTime - startTime) + "s")
        return (True, totalStitch)

    # @jit
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
                ptsA = np.float32([kpsA[i] for (_, i) in matches]); ptsA[:, [0, 1]] = ptsA[:, [1, 0]];
                ptsB = np.float32([kpsB[i] for (i, _) in matches]); ptsB[:, [0, 1]] = ptsB[:, [1, 0]];

                if direction == "horizontal" :
                    ptsA[:, 1] = ptsA[:, 1] + imageA.shape[1] - i * roiFirstLength
                    ptsB[:, 1] = ptsB[:, 1] + ptsA[:, 1]
                elif direction == "vertical":
                    ptsA[:, 0] = ptsA[:, 0] + imageA.shape[0] - i * roiFirstLength
                    ptsB[:, 0] = ptsB[:, 0] + ptsA[:, 0]
                print("A")
                print(ptsA)
                print("B")
                print(ptsB)
                # match all the feature points
                localStartTime = time.time()

                if offsetCaculate == "mode":
                    (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate)
                elif offsetCaculate == "ransac":
                    (status, offset, adjustH) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate)
                    H = adjustH
                # print(offset)
                # if direction == "horizontal":
                #     offset[1] = offset[1] + imageA.shape[1] - i * roiFirstLength
                # elif direction == "vertical":
                #     offset[0] = offset[0] + imageA.shape[0] - i * roiFirstLength
                if status == True:
                    break
        if status == False:
            return (status, "  The two images can not match", 0)
        elif status == True:
            localEndTime = time.time()
            self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
            self.printAndWrite("  The time of mode/ransac is " + str(localEndTime - localStartTime) + "s")
            return (status, offset, H)

    # @jit
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

    # @jit
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
        elif featureMethod == "orb":
            descriptor = cv2.ORB_create(5000000)
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return (kps, features)

    # @jit
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
        self.printAndWrite("  The number of matches is " + str(len(matches)))
        return matches

    # @jit
    def getOffsetByMode(self, kpsA, kpsB, matches, offsetEvaluate=100):
        totalStatus = True
        if len(matches) < offsetEvaluate:
            totalStatus = False
            return (False, [0, 0])
        dxList = []; dyList = [];
        for trainIdx, queryIdx in matches:
            ptA = (kpsA[queryIdx][1], kpsA[queryIdx][0])
            ptB = (kpsB[trainIdx][1], kpsB[trainIdx][0])
            dxList.append(round(ptB[0] - ptA[0]))
            dyList.append(round(ptB[1] - ptA[1]))
        dxMode, count = mode(np.array(dxList), axis=None)
        dyMode, count = mode(np.array(dyList), axis=None)
        dx = int(dxMode); dy = int(dyMode)
        return (True, [dx, dy])

    # @jit
    def getOffsetByRansac(self, kpsA, kpsB, matches, offsetEvaluate=100):
        totalStatus = False

        if len(matches) == 0:
            return (totalStatus, [0, 0], 0)
        # 计算视角变换矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3, 0.9)
        print(H)
        trueCount = 0
        for i in range(0, len(status)):
            if status[i] == True:
                trueCount = trueCount + 1
        if trueCount >= offsetEvaluate:
            totalStatus = True
            adjustH = H.copy()
            adjustH[0, 2] = 0;adjustH[1, 2] = 0
            adjustH[2, 0] = 0;adjustH[2, 1] = 0
            return (totalStatus ,[np.array(H).astype(np.int)[1,2] * (-1), np.array(H).astype(np.int)[0,2] * (-1)], adjustH)
        else:
            return (totalStatus, [0, 0], 0)

    # @jit
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
        fuseRegion = self.fuseImage([roiImageRegionA, roiImageRegionB], direction=direction, fuseMethod=fuseMethod)
        stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby] = fuseRegion.copy()
        return (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB)

    # @jit
    def fuseImage(self, images, fuseMethod, direction="horizontal"):
        (imageA, imageB) = images
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        imageA[imageA == 0] = imageB[imageA == 0]
        imageB[imageB == 0] = imageA[imageB == 0]
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
    # fileList = [".\\images\\dendriticCrystal\\2\\2-030.jpg", ".\\images\\dendriticCrystal\\2\\2-031.jpg"]
    # outputAddress = "result"
    # evaluate = (True, "evaluate.txt")
    # isPrintLog = True
    # stitcher = Stitcher(outputAddress, evaluate, isPrintLog)
    # registrateMethod = ("featureSearchWithIncrease", "surf", 0.65, ("mode", 100), (150, -1))
    # fuseMethod = ("notFuse", "Test")
    #
    # (status, result) = stitcher.pairwiseStitch(fileList, registrateMethod, fuseMethod, direction="horizontal")
    # if status == True:
    #     cv2.imwrite(outputAddress + "\\stitching_result.jpg", result)
    # if status == False:
    #     print("拼接失败")
    outputAddress = "result"
    fileList = [".\\result\\dendriticCrystalNotFuse\\0.jpg", ".\\result\\dendriticCrystalNotFuse\\1.jpg", ".\\result\\dendriticCrystalNotFuse\\2.jpg"]
    offset = [(-103, 2449), (-4, 4872)]
    evaluate = (True, "evaluate.txt")
    isPrintLog = True
    stitcher = Stitcher(outputAddress, evaluate, isPrintLog)
    registrateMethod = ("featureSearchWithIncrease", "surf", 0.65, ("mode", 100), (150, -1))
    fuseMethod = ("trigonometric", "Test")
    imageA = cv2.imread(fileList[0], 0)
    imageB = cv2.imread(fileList[1], 0)
    imageC = cv2.imread(fileList[2], 0)
    (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = stitcher.getStitchByOffset([imageA, imageB], offset[0], fuseMethod)
    (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = stitcher.getStitchByOffset([stitchImage, imageC],
                                                                                             offset[1], fuseMethod)
    cv2.imwrite("result.jpg", stitchImage)