import numpy as np
import cv2
from scipy.stats import mode
import time
import ImageFusion

class Stitcher:
    '''
	    图像拼接类，包括所有跟材料显微组织图像配准相关函数
	'''

    def stitchOneColumn(self, startIndex, endIndex, numPixelControl, Files):
        pass

    def stitchOneRow(self, startIndex, endIndex, numPixelControl, Files):
        pass

    def stitchTwoColumns(self, leftColumnAddress, rightColumnAddress, numPixelControl):
        pass

    def stitchTwoRows(self, upColumnAddress, downColumnAddress, numPixelControl):
        pass

    def fuseImage(self, images, direction="horizontal", fuseMethod = "linearBlending"):
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
        elif fuseMethod == "linearBlending":
            fuseRegion = ImageFusion.fuseByLinearBlending(images,direction)
        elif fuseMethod == "multiBandBlending":
            fuseRegion = ImageFusion.fuseByMultiBandBlending(images)
        return fuseRegion

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

    def getROIRegion(self, image, direction="horizontal", order="first", searchLength=150, searchLengthForLarge=-1):
        '''对原始图像裁剪感兴趣区域
        :param originalImage:需要裁剪的原始图像
        :param m_direction:拼接的方向
        :param m_order:该图片的顺序，是属于第一还是第二张图像
        :param m_searchLength:搜索区域大小
        :param m_searchLengthForLarge:对于行拼接和列拼接的搜索区域大小
        :return:返回感兴趣区域图像
        :type m_searchLength: np.int
        '''
        # roiRegion = originalImage.copy()
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

    def getOffsetByMode(self, images, matches, kpsA, kpsB, featuresA, featuresB, direction="horizontal"):
        # 建立状态，判断是否匹配成功
        status = False
        dx = []; dy = []
        # 获取输入图片及其搜索区域
        (imageA, imageB) = images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        for trainIdx, queryIdx in matches:
            ptA = (int(kpsA[queryIdx][1]), int(kpsA[queryIdx][0]))
            ptB = (int(kpsB[trainIdx][1]), int(kpsB[trainIdx][0]))
            if direction == "horizontal":
                dx.append(ptA[0] - ptB[0])
                dy.append((wA - ptA[1]) + ptB[1])
            elif direction == "vertical":
                dx.append((hA - ptA[0]) + ptB[0])
                dy.append(ptA[1] - ptB[1])
        return int(mode(np.array(dx), axis=None)[0]), int(mode(np.array(dy), axis=None)[0])

    def getStitchByOffset(self, images, dx, dy, direction="horizontal", fuseMethod="linearBlending"):
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

    # 拼接函数，根据位移拼接
    def stitchByOffset(self, images, ratio=0.75, reprojThresh=4.0, featureMethod="sift", direction="horizontal", searchLength=150, searchLengthForLarge=-1, fuseMethod="linearBlending"):
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
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        print(H)
        print(status)
        # 根据匹配的特征点计算偏移量
        dx, dy = self.getOffsetByMode([roiImageA, roiImageB], matches, kpsA, kpsB, featuresA, featuresB, direction=direction)
        print(" The offset of stitching: dx is " + str(dx) + " and dy is " + str(dy))

        # 根据偏移量创建拼接好的整体图像
        resultStitched = self.getStitchByOffset(images, dx, dy, direction=direction, fuseMethod = fuseMethod)

        return resultStitched

    # # 拼接函数，根据投影变换拼接
    # def stitchByPerspective(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False, method="SIFT",
    #            direction="horizontal"):
    #     # 获取输入图片
    #     (imageB, imageA) = images
    #
    #     # 检测A、B图片的SIFT关键特征点，并计算特征描述子,
    #     # kps为一个矩阵，每一行代表一个特征点的坐标，features为一个矩阵，每一行代表一个特征点的特征描述符
    #
    #     (kpsA, featuresA) = self.detectAndDescribe(imageA, method=method)
    #     (kpsB, featuresB) = self.detectAndDescribe(imageB, method=method)
    #
    #     # 匹配两张图片的所有特征点，返回匹配结果
    #     M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    #
    #     # 如果返回结果为空，没有匹配成功的特征点，退出算法
    #     if M is None:
    #         return None
    #
    #
    #
    #     # 否则，提取匹配结果
    #     # H是3x3视角变换矩阵
    #     (matches, H, status) = M
    #     if m_direction == "horizontal":
    #         # 将图片A进行投射变换，result是变换后图片，H为变换矩阵，第三个参数是变换后的大小(y,x)
    #         result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    #         # cv2.imshow("test", np.hstack((imageA,result))) # 展现了原图和投射变换的差别
    #         cv2.waitKey(0)
    #         # 将图片B传入result图片最左端
    #         result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    #     elif m_direction == "vertical":
    #         result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0] + imageB.shape[0]))
    #         result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    #
    #     # 检测是否需要显示图片匹配
    #     if showMatches:
    #         # 生成匹配图片
    #         vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status, direction=direction)
    #         # 返回结果
    #         return (result, vis)
    #         # 返回匹配结果
    #     return result
    # # def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # #     # 建立暴力匹配器
    # #     matcher = cv2.DescriptorMatcher_create("BruteForce")
    # #
    # #     # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
    # #     rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    # #     matches = []
    # #     for m in rawMatches:
    # #         # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
    # #         if len(m) == 2 and m[0].distance < m[1].distance * ratio:
    # #             # 存储两个点在featuresA, featuresB中的索引值
    # #             matches.append((m[0].trainIdx, m[0].queryIdx))
    # #
    # #     # 当筛选后的匹配对大于4时，计算视角变换矩阵
    # #     if len(matches) > 4:
    # #         # 获取匹配对的点坐标
    # #         ptsA = np.float32([kpsA[i] for (_, i) in matches])
    # #         ptsB = np.float32([kpsB[i] for (i, _) in matches])
    # #
    # #         # 计算视角变换矩阵
    #         (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
    # #
    # #         # 返回结果
    # #         return (matches, H, status)
    # #
    # #     # 如果匹配对小于4时，返回None
    # #     return None
    # def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status, direction="horizontal"):
    #     # 初始化可视化图片，将A、B图左右连接到一起
    #     (hA, wA) = imageA.shape[:2]
    #     (hB, wB) = imageB.shape[:2]
    #     if direction == "horizontal":
    #         vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    #         vis[0:hA, 0:wA] = imageA
    #         vis[0:hB, wA:] = imageB
    #
    #         # 联合遍历，画出匹配对
    #         for ((trainIdx, queryIdx), s) in zip(matches, status):
    #             # 当点对匹配成功时，画到可视化图上
    #             if s == 1:
    #                 # 画出匹配对
    #                 ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
    #                 ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
    #                 cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    #     elif direction == "vertical":
    #         vis = np.zeros((hA + hB, max(wA, wB), 3), dtype="uint8")
    #         vis[0:hA, 0:wA] = imageA
    #         vis[hA:, 0:wB] = imageB
    #
    #         # 联合遍历，画出匹配对
    #         for ((trainIdx, queryIdx), s) in zip(matches, status):
    #             # 当点对匹配成功时，画到可视化图上
    #             if s == 1:
    #                 # 画出匹配对
    #                 ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
    #                 ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1] + hA))
    #                 cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    #     # 返回可视化结果
    #     return vis


if __name__=="__main__":
    # imageA = cv2.cvtColor(cv2.imread("images/dendriticCrystal/1/1-001.jpg"), cv2.COLOR_RGB2GRAY)
    # imageB = cv2.cvtColor(cv2.imread("images/dendriticCrystal/1/1-002.jpg"), cv2.COLOR_RGB2GRAY)
    imageA = cv2.cvtColor(cv2.imread("images/dendriticCrystal/iron/1-001.jpg"), cv2.COLOR_RGB2GRAY)
    imageB = cv2.cvtColor(cv2.imread("images/dendriticCrystal/iron/1-002.jpg"), cv2.COLOR_RGB2GRAY)
    startTime = time.time()
    stitcher = Stitcher()
    result = stitcher.stitchByOffset([imageA, imageB], ratio=0.75, reprojThresh=4.0, featureMethod="surf", direction="vertical", searchLength=150, searchLengthForLarge=-1, fuseMethod="linearBlending")
    endTime = time.time()
    print(" The cost time is :" + str(endTime-startTime) +"s")
    cv2.namedWindow("Result", 0)
    cv2.imshow("Result", result)
    cv2.imwrite("111.jpg", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# "average""maximum""minimum""linearBlending""multiBandBlending"