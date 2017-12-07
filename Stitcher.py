import numpy as np
import cv2
from scipy.stats import mode
import time
import ImageFusion

class Stitcher:
    '''
	    图像拼接类，包括所有跟材料显微组织图像配准相关函数
	'''
    dataType = "Images"             # "Images" or "Video"
    stitchType = "pairwiseStitch"   # "pairwiseStitch" or "gridStitch"
    shootingOrder = "colBycol"
    fileList = []


    outputAddress = "result/"
    isEvaluate = False
    evaluateFile = "evaluate.txt"
    isPrintLog = False

    def __init__(self, dataType, stitchType, registrateMethod, fuseMethod, outputAddress, evaluate, isPrintLog):
        self.dataType = dataType
        if dataType == "Images":
            shootingOrder = stitchType[1]
            fileList = stitchType[2]
        self.outputAddress = outputAddress
        self.isEvaluate = evaluate[0]
        self.evaluateFile = evaluate[1]
        self.isPrintLog = isPrintLog



    def stitchOneColumn(self, startIndex, endIndex, numPixelControl, Files):
        pass

    def stitchOneRow(self, startIndex, endIndex, numPixelControl, Files):
        pass

    def stitchTwoColumns(self, leftColumnAddress, rightColumnAddress, numPixelControl):
        pass

    def stitchTwoRows(self, upColumnAddress, downColumnAddress, numPixelControl):
        pass

    def stitchTwoImages(self, images, registrateMethod, fuseMethod, direction="horizontal"):
        '''
        Stitch two images
        :param images: [imageA, imageB]
        :param registrateMethod: list:
        :param fuseMethod:
        :param outputAddress:
        :param direction: stitching direction
        :return:
        '''
        if registrateMethod[0] == "phaseCorrection":
            return (False, "We don't develop the phase Correction method, Plesae wait for updating")
        elif registrateMethod[0] == "featureSearch":
            searchMethod = registrateMethod[1]         # "sift","surf" or "orb"
            searchRatio = registrateMethod[2]          # 0.75 is common value for matches
            offsetCaculate = registrateMethod[3][0]    # "mode" or "ransac"
            offsetEvaluate = registrateMethod[3][1]    # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
            (imageA, imageB) = images
            if registrateMethod[4][0] == False:        # means whether or not use roi region
                roiImageA = imageA
                roiImageB = imageB
            else:
                roiFirstLength = registrateMethod[4][1]     # roi length for stitching in first direction
                roiSecondLength = registrateMethod[4][1]    # roi length for stitching in second direction
                # get the roi region of images
                roiImageA = self.getROIRegion(imageA, direction=direction, order="first", searchLength=roiFirstLength,
                                              searchLengthForLarge=roiSecondLength)
                roiImageB = self.getROIRegion(imageB, direction=direction, order="second", searchLength=roiFirstLength,
                                              searchLengthForLarge=roiSecondLength)
            # get the feature points
            (kpsA, featuresA) = self.detectAndDescribe(roiImageA, featureMethod=featureMethod)
            (kpsB, featuresB) = self.detectAndDescribe(roiImageB, featureMethod=featureMethod)

            # match all the feature points
            matches = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, searchRatio)
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            if offsetCaculate == "mode":
                (status, offset) = self.getOffsetByMode(images, matches, kpsA, kpsB, featuresA, featuresB, evaluateNum=offsetEvaluate, direction="horizontal")
                if status == False:
                    return (False, "two images can not match")
            elif offsetCaculate == "ransac":
                return (False, "We don't develop the stitching with ransac method, Plesae wait for updating")
            dx = offset[0]; dy = offset[1]
            if self.isEvaluate == True:
                pass
            result = self.getStitchByOffset(images, dx, dy, evaluate, direction="horizontal", fuseMethod=fuseMethod)

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
            if searchLength >= 0 and searchLength <= 1:
                searchLength = row * searchLength
            if searchLengthForLarge >= 0 and searchLength <= 1:
                searchLengthForLarge = col * searchLengthForLarge
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
            if searchLength >= 0 and searchLength <= 1:
                searchLength = col * searchLength
            if searchLengthForLarge >= 0 and searchLength <= 1:
                searchLengthForLarge = row * searchLengthForLarge
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
        :param kpsA:第一张图像的特征点的坐标
        :param kpsB: 第二张图像的特征点的
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
        return matches

    def getOffsetByMode(self, images, matches, kpsA, kpsB, featuresA, featuresB, evaluateNum=20, direction="horizontal"):
        # 建立状态，判断是否匹配成功
        if len(matches) < evaluateNum:
            return (False, (0, 0 ))
        dx = []; dy = []
        # 获取输入图片及其搜索区域
        (imageA, imageB) = images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        for trainIdx, queryIdx in matches:
            ptA = (int(kpsA[queryIdx][1]), int(kpsA[queryIdx][0]))
            ptB = (int(kpsB[trainIdx][1]), int(kpsB[trainIdx][0]))
            # if direction == "horizontal":
            dx.append(ptA[0] - ptB[0])
            dy.append(ptA[1] - ptB[1])
            # elif direction == "vertical":
            #     dx.append((hA - ptA[0]) + ptB[0])
            #     dy.append(ptA[1] - ptB[1])
        return (True, (int(mode(np.array(dx), axis=None)[0]), int(mode(np.array(dy), axis=None)[0])))

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

    def getStitchByOffset(self, images, dx, dy, direction="horizontal", fuseMethod="fadeInAndFadeOut"):
        (imageA, imageB) = images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        if direction == "horizontal":
            if dx < 0:
                cutImageA = self.creatOffsetImage(imageA, direction, dx)
                stitchImage = np.hstack((cutImageA[0:hA+dx, 0:wA-dy], imageB[0:hA+dx, :]))
                if fuseMethod != "notFuse":
                    fuseRegion = self.fuseImage([cutImageA[0:hA+dx, wA-dy:wA], imageB[0:hA+dx, 0:dy]], direction=direction,
                                                 fuseMethod=fuseMethod)
                    stitchImage[:, wA - dy:wA] = fuseRegion[:]
            elif dx > 0:
                cutImageA = self.creatOffsetImage(imageA, direction, dx)
                stitchImage = np.hstack((cutImageA[0:hA-dx, 0:wA-dy], imageB[0:hA-dx, :]))
                # 判断是否对图像进行融合
                if fuseMethod != "notFuse":
                    fuseRegion = self.fuseImage([cutImageA[0:hA-dx,wA-dy:wA], imageB[0:hA-dx,0:dy]], direction=direction,
                                                 fuseMethod=fuseMethod)
                    stitchImage[:, wA-dy:wA] = fuseRegion[:]
        elif direction == "vertical":
            if dy < 0:
                cutImageA = self.creatOffsetImage(imageA, direction, dy)
                stitchImage = np.vstack((cutImageA[0:hA-dx, (-dy):wA], imageB[:, 0:wB+dy]))
                if fuseMethod != "notFuse":
                    fuseRegion = self.fuseImage([cutImageA[hA-dx:hA, (-dy):wA], imageB[0:dx, 0:wB+dy]], direction=direction,
                                                 fuseMethod=fuseMethod)
                    stitchImage[hA-dx:hA, :] = fuseRegion[:]
            elif dy > 0:
                cutImageA = self.creatOffsetImage(imageA, direction, dy)
                stitchImage = np.vstack((cutImageA[0:hA-dx, dy:wA], imageB[:, 0:wB-dy]))
                if fuseMethod != "notFuse":
                    fuseRegion = self.fuseImage([cutImageA[hA-dx:hA, dy:wA], imageB[0:dx, 0:wB-dy]], direction=direction,
                                                 fuseMethod=fuseMethod)
                    stitchImage[hA - dx:hA, :] = fuseRegion[:]
        return stitchImage

    def fuseImage(self, images, evaluate, direction="horizontal", fuseMethod = "fadeInAndFadeOut"):
        (imageA, imageB) = images
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        # cv2.namedWindow("imageA", 0)
        # cv2.namedWindow("imageB", 0)
        # cv2.imshow("imageA", imageA)  # 测试使用
        # cv2.imshow("imageB", imageB)  # 测试使用
        # cv2.waitKey(0)
        if fuseMethod == "average":
            fuseRegion = ImageFusion.fuseByAverage(images)
        elif fuseMethod == "maximum":
            fuseRegion = ImageFusion.fuseByMaximum(images)
        elif fuseMethod == "minimum":
            fuseRegion = ImageFusion.fuseByMinimum(images)
        elif fuseMethod == "fadeInAndFadeOut":
            fuseRegion = ImageFusion.fuseByFadeInAndFadeOut(images,direction)
        elif fuseMethod == "multiBandBlending":
            fuseRegion = ImageFusion.fuseByMultiBandBlending(images)
        elif fuseMethod == "trigonometric":
            fuseRegion = ImageFusion.fuseByTrigonometric(images,direction)
        elif fuseMethod == "optimalSeamLine":
            fuseRegion = ImageFusion.fuseByOptimalSeamLine(images, direction)
        return fuseRegion


    # 拼接函数，根据位移拼接
    def stitchByOffset(self, images, ratio=0.75, reprojThresh=4.0, featureMethod="sift", direction="horizontal", searchLength=150, searchLengthForLarge=-1, fuseMethod="fadeInAndFadeOut"):
        # 获取输入图片及其搜索区域
        (imageA, imageB) = images
        roiImageA = self.getROIRegion(imageA, direction=direction, order="first", searchLength=searchLength, searchLengthForLarge=searchLengthForLarge)
        roiImageB = self.getROIRegion(imageB, direction=direction, order="second", searchLength=searchLength, searchLengthForLarge=searchLengthForLarge)

        # 获取特征点
        (kpsA, featuresA) = self.detectAndDescribe(roiImageA, featureMethod=featureMethod)
        (kpsB, featuresB) = self.detectAndDescribe(roiImageB, featureMethod=featureMethod)

        # 匹配两张图片的所有特征点，返回匹配结果
        matches = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio)
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        # print(H)
        # print(status)
        # 根据匹配的特征点计算偏移量
        dx, dy = self.getOffsetByMode([roiImageA, roiImageB], matches, kpsA, kpsB, featuresA, featuresB, direction=direction)
        print(" The offset of stitching: dx is " + str(dx) + " and dy is " + str(dy))

        # 根据偏移量创建拼接好的整体图像
        resultStitched = self.getStitchByOffset(images, dx, dy, direction=direction, fuseMethod = fuseMethod)

        return resultStitched

if __name__=="__main__":
    # imageA = cv2.cvtColor(cv2.imread("images/dendriticCrystal/1/1-001.jpg"), cv2.COLOR_RGB2GRAY)
    # imageB = cv2.cvtColor(cv2.imread("images/dendriticCrystal/1/1-030.jpg"), cv2.COLOR_RGB2GRAY)
    imageA = cv2.cvtColor(cv2.imread("images/dendriticCrystal/iron/1-001.jpg"), cv2.COLOR_RGB2GRAY)
    imageB = cv2.cvtColor(cv2.imread("images/dendriticCrystal/iron/1-002.jpg"), cv2.COLOR_RGB2GRAY)
    startTime = time.time()
    stitcher = Stitcher()
    result = stitcher.stitchByOffset([imageA, imageB], ratio=0.75, reprojThresh=4.0, featureMethod="surf", direction="vertical", searchLength=150, searchLengthForLarge=-1, fuseMethod="optimalSeamLine") # optimalSeamLine
    endTime = time.time()
    print(" The cost time is :" + str(endTime-startTime) +"s")
    cv2.namedWindow("Result", 0)
    cv2.imshow("Result", result)
    cv2.imwrite("Result.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()