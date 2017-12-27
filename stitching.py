import numpy as np
import cv2


class Method():
    outputAddress = "result/"
    isEvaluate = False
    evaluateFile = "evaluate.txt"
    isPrintLog = False
    isParallel = False
    isNGPUWork = False

    def __init__(self, outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork):
        self.outputAddress = outputAddress
        self.isEvaluate = isEvaluate
        self.evaluateFile = evaluateFile
        self.isPrintLog = isPrintLog
        self.isParallel = isParallel
        self.isNGPUWork = isNGPUWork

    def printAndWrite(self, content):
        if self.isPrintLog:
            print(content)
        if self.isEvaluate:
            f = open(self.outputAddress + self.evaluateFile, "a")
            f.write(content)
            f.write("\n")
            f.close()

class Stitching():

    def pairwiseStitch(self, fileList, registrateMethod, fuseMethod, direction="horizontal"):
        self.printAndWrite("Stitching " + str(fileList[0]) + " and " + str(fileList[1]))

        imageA = cv2.imread(fileList[0], 0)
        imageB = cv2.imread(fileList[1], 0)
        startTime = time.time()
        (status, offset, H) = self.calculateOffset([imageA,imageB], registrateMethod, direction=direction)
        # if registrateMethod[3][0] == "ransac":
        #     imageA = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]))
        #     print("H")
        #     print(H)
        #     (status, offset, H) = self.calculateOffset([imageA, imageB], registrateMethod,direction=direction)
        #     # print(H)
        #     # cv2.namedWindow("imageA",0)
        #     # cv2.imshow("imageA", imageA)
        #     # cv2.waitKey(0)
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

class Register(Method):
    fileAddress = []

    def __init__(self, outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork, fileAddress):
        super(Register, self).__init__(outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork)
        self.fileAddress = fileAddress


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


class FeatureDetect(Register):
    featureMethod = "sift"
    featureRatio = 0.7
    kpsA = None
    kpsB = None

    def __init__(self, outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork, featureMethod, featureRatio):
        super(Register, self).__init__(outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork)
        self.featureMethod = featureMethod
        self.featureRatio = featureRatio

    def detectAndDescribe(self, image):
        '''
    	计算图像的特征点集合，并返回该点集＆描述特征
    	:param image:需要分析的图像
    	:return:返回特征点集，及对应的描述特征
    	'''
        # 将彩色图片转换成灰度图
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        if self.featureMethod == "sift":
            descriptor = cv2.xfeatures2d.SIFT_create()
        elif self.featureMethod == "surf":
            descriptor = cv2.xfeatures2d.SURF_create()
        elif self.featureMethod == "orb":
            descriptor = cv2.ORB_create(5000000)
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB):
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
            if len(m) == 2 and m[0].distance < m[1].distance * self.featureRatio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))
        self.printAndWrite("  The number of matches is " + str(len(matches)))
        return matches


class phaseCorrelation(Register):
    pass

class stitching(Method):
    def __init__(self, outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork, featureMethod, featureRatio):
        super(Method, self).__init__(outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork)

    def pairWiseStitch(self):
        pass

    def gridStitch(self):
        pass


class Fusion(Method):
    fuseMethod = "notFuse"

    def __init__(self, outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork, fuseMethod):
        Method.__init__(self, outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork)
        self.fuseMethod = fuseMethod
