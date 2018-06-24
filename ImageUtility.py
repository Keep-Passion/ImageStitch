import numpy as np
import cv2
import math
import myGpuFeatures
from scipy.stats import mode

class Method():
    # 关于打印信息的设置
    outputAddress = "result/"
    isEvaluate = False
    evaluateFile = "evaluate.txt"
    isPrintLog = True

    # 关于特征搜索的设置
    featureMethod = "surf"      # "sift","surf" or "orb"
    roiRatio = 0.1              # roi length for stitching in first direction
    searchRatio = 0.75          # 0.75 is common value for matches

    # 关于 GPU 加速的设置
    isGPUAvailable = True

    # 关于 GPU-SURF 的设置
    surfHessianThreshold = 100.0
    surfNOctaves = 4
    surfNOctaveLayers = 3
    surfIsExtended = True
    surfKeypointsRatio = 0.01
    surfIsUpright = False

    # 关于 GPU-ORB 的设置
    orbNfeatures = 500
    orbScaleFactor = 1.2
    orbNlevels = 8
    orbEdgeThreshold = 31
    orbFirstLevel = 0
    orbWTA_K = 2
    orbPatchSize = 31
    orbFastThreshold = 20
    orbBlurForDescriptor = True
    orbMaxDistance = 30

    # 关于特征配准的设置
    offsetCaculate = "mode"     # "mode" or "ransac"
    offsetEvaluate = 3         # 40 menas nums of matches for mode, 3.0 menas  of matches for ransac

    # 关于图像增强的操作
    isEnhance = False
    isClahe = False
    clipLimit = 20
    tileSize = 5

    # 向屏幕和文件打印输出内容
    def printAndWrite(self, content):
        if self.isPrintLog:
            print(content)
        if self.isEvaluate:
            f = open(self.outputAddress + self.evaluateFile, "a")
            f.write(content)
            f.write("\n")
            f.close()

    def getROIRegionForIncreMethod(self, image, direction=1, order="first", searchRatio=0.1):
        row, col = image.shape[:2]
        roiRegion = np.zeros(image.shape, np.uint8)
        if direction == 1:
            searchLength = np.floor(row * searchRatio).astype(int)
            if order == "first":
                roiRegion = image[row - searchLength:row, :]
            elif order == "second":
                roiRegion = image[0: searchLength, :]
        elif direction == 2:
            searchLength = np.floor(col * searchRatio).astype(int)
            if order == "first":
                roiRegion = image[:, col - searchLength:col]
            elif order == "second":
                roiRegion = image[:, 0: searchLength]
        elif direction == 3:
            searchLength = np.floor(row * searchRatio).astype(int)
            if order == "first":
                roiRegion = image[0: searchLength, :]
            elif order == "second":
                roiRegion = image[row - searchLength:row, :]
        elif direction == 4:
            searchLength = np.floor(col * searchRatio).astype(int)
            if order == "first":
                roiRegion = image[:, 0: searchLength]
            elif order == "second":
                roiRegion = image[:, col - searchLength:col]
        return roiRegion

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
        if direction == "horizontal" or direction == 2:
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
        elif direction == "vertical" or direction == 1:
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

    def getOffsetByMode(self, kpsA, kpsB, matches, offsetEvaluate = 10):
        totalStatus = True
        if len(matches) == 0:
            totalStatus = False
            return (totalStatus, [0, 0])
        dxList = []; dyList = [];
        for trainIdx, queryIdx in matches:
            ptA = (kpsA[queryIdx][1], kpsA[queryIdx][0])
            ptB = (kpsB[trainIdx][1], kpsB[trainIdx][0])
            # dxList.append(int(round(ptA[0] - ptB[0])))
            # dyList.append(int(round(ptA[1] - ptB[1])))
            if int(ptA[0] - ptB[0]) == 0 and int(ptA[1] - ptB[1]) == 0:
                continue
            dxList.append(int(ptA[0] - ptB[0]))
            dyList.append(int(ptA[1] - ptB[1]))
        if len(dxList) == 0:
            dxList.append(0); dyList.append(0)
        # Get Mode offset in [dxList, dyList], thanks for clovermini
        zipped = zip(dxList, dyList)
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))

        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        # print("dx = " + str(dx) + ", dy = " + str(dy) + ", num = " + str(num))

        if num < offsetEvaluate:
            totalStatus = False
        # self.printAndWrite("  In Mode, The number of num is " + str(num) + " and the number of offsetEvaluate is "+str(offsetEvaluate))
        return (totalStatus, [dx, dy])

    def getOffsetByRansac(self, kpsA, kpsB, matches, offsetEvaluate=100):
        totalStatus = False
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        if len(matches) == 0:
            return (totalStatus, [0, 0], 0)
        # 计算视角变换矩阵
        H1 = cv2.getAffineTransform(ptsA, ptsB)
        # print("H1")
        # print(H1)
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3, 0.9)
        trueCount = 0
        for i in range(0, len(status)):
            if status[i] == True:
                trueCount = trueCount + 1
        if trueCount >= offsetEvaluate:
            totalStatus = True
            adjustH = H.copy()
            adjustH[0, 2] = 0;adjustH[1, 2] = 0
            adjustH[2, 0] = 0;adjustH[2, 1] = 0
            return (totalStatus ,[np.round(np.array(H).astype(np.int)[1,2]) * (-1), np.round(np.array(H).astype(np.int)[0,2]) * (-1)], adjustH)
        else:
            return (totalStatus, [0, 0], 0)

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

    def npToKpsAndDescriptors(self, array):
        kps = []
        descriptors = array[:, :, 1]
        for i in range(array.shape[0]):
            kps.append([array[i, 0, 0], array[i, 1, 0]])
        return (kps, descriptors)

    def detectAndDescribe(self, image, featureMethod):
        '''
    	计算图像的特征点集合，并返回该点集＆描述特征
    	:param image:需要分析的图像
    	:return:返回特征点集，及对应的描述特征
    	'''
        if self.isGPUAvailable == False: # CPU mode
            if featureMethod == "sift":
                descriptor = cv2.xfeatures2d.SIFT_create()
            elif featureMethod == "surf":
                descriptor = cv2.xfeatures2d.SURF_create()
            elif featureMethod == "orb":
                descriptor = cv2.ORB_create(self.orbNfeatures, self.orbScaleFactor, self.orbNlevels, self.orbEdgeThreshold, self.orbFirstLevel, self.orbWTA_K, 0, self.orbPatchSize, self.orbFastThreshold)
            # 检测SIFT特征点，并计算描述子
            kps, features = descriptor.detectAndCompute(image, None)
            # 将结果转换成NumPy数组
            kps = np.float32([kp.pt for kp in kps])
        else:                           # GPU mode
            if featureMethod == "sift":
                # 目前GPU-SIFT尚未开发，先采用CPU版本的替代
                descriptor = cv2.xfeatures2d.SIFT_create()
                kps, features = descriptor.detectAndCompute(image, None)
                kps = np.float32([kp.pt for kp in kps])
            elif featureMethod == "surf":
                kps, features = self.npToKpsAndDescriptors(myGpuFeatures.detectAndDescribeBySurf(image, self.surfHessianThreshold, self.surfNOctaves,self.surfNOctaveLayers, self.surfIsExtended, self.surfKeypointsRatio, self.surfIsUpright))
            elif featureMethod == "orb":
                kps, features = self.npToKpsAndDescriptors(myGpuFeatures.detectAndDescribeByOrb(image, self.orbNfeatures, self.orbScaleFactor, self.orbNlevels, self.orbEdgeThreshold, self.orbFirstLevel, self.orbWTA_K, 0, self.orbPatchSize, self.orbFastThreshold, self.orbBlurForDescriptor))
        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchDescriptors(self, featuresA, featuresB):
        '''
        匹配特征点
        :param self:
        :param featuresA: 第一张图像的特征点描述符
        :param featuresB: 第二张图像的特征点描述符
        :param ratio: 最近邻和次近邻的比例
        :return:返回匹配的对数
        '''
        if self.isGPUAvailable == False:        # CPU Mode
            # 建立暴力匹配器
            if self.featureMethod == "surf" or self.featureMethod == "sift":
                matcher = cv2.DescriptorMatcher_create("BruteForce")
                # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
                rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
                matches = []
                for m in rawMatches:
                # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
                    if len(m) == 2 and m[0].distance < m[1].distance * self.searchRatio:
                        # 存储两个点在featuresA, featuresB中的索引值
                        matches.append((m[0].trainIdx, m[0].queryIdx))
            elif self.featureMethod == "orb":
                matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
                rawMatches = matcher.match(featuresA, featuresB)
                matches = []
                for m in rawMatches:
                    matches.append((m.trainIdx, m.queryIdx))
            # self.printAndWrite("  The number of matches is " + str(len(matches)))
        else:                                   # GPU Mode
            if self.featureMethod == "surf":
                matches = self.npToListForMatches(myGpuFeatures.matchDescriptors(np.array(featuresA), np.array(featuresB), 2, self.searchRatio))
            elif self.featureMethod == "orb":
                matches = self.npToListForMatches(myGpuFeatures.matchDescriptors(np.array(featuresA), np.array(featuresB), 3, self.orbMaxDistance))
        return matches

    def resizeImg(self, image, resizeTimes, interMethod = cv2.INTER_AREA):
        (h, w) = image.shape
        resizeH = int(h * resizeTimes)
        resizeW = int(w * resizeTimes)
        # cv2.INTER_AREA是测试后最好的方法
        return cv2.resize(image, (resizeW, resizeH), interpolation=interMethod)

    def rectifyFinalImg(self, image, regionLength = 10):

        (h, w) = image.shape
        print("h:" + str(h))
        print("w:" + str(w))
        upperLeft   = np.sum(image[0: regionLength, 0: regionLength])
        upperRight  = np.sum(image[0: regionLength, w - regionLength: w])
        bottomLeft  = np.sum(image[h - regionLength: h, 0: regionLength])
        bottomRight = np.sum(image[h - regionLength: h, w - regionLength: w])

        # 预处理
        zeroCol = image[:, 0]
        noneZeroNum = np.count_nonzero(zeroCol)
        zeroNum = h - noneZeroNum
        print("noneZeroNum:" + str(noneZeroNum))
        print("zeroNum:" + str(zeroNum))
        print("除法:" + str(noneZeroNum / h))
        if (noneZeroNum / h) < 0.3:
            resultImage = image
        elif upperLeft == 0 and bottomRight == 0 and upperRight != 0 and bottomLeft != 0:      # 左边低，右边高
            print(1)
            center = (w // 2, h // 2)
            print(w)
            print(h)
            angle = math.atan(center[1] / center[0] * 180 / math.pi)
            print(str(angle))
            M = cv2.getRotationMatrix2D(center, -1 * angle, 1.0)
            print(M)
            resultImage = cv2.warpAffine(image, M, (w, h))
        elif upperLeft != 0 and bottomRight != 0 and upperRight == 0 and bottomLeft == 0:    # 左边高，右边低
            print(2)
            center = (w // 2, h // 2)
            angle = math.atan(center[1] / center[0] * 180 / math.pi)
            print(str(angle))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            resultImage = cv2.warpAffine(image, M, (w, h))
        else:
            resultImage = image
        return resultImage

if __name__=="__main__":
    image = cv2.imread("D:\\Coding_Test\\Python\\ImageStitch\\images\\zirconSmall\\1\\WJE068-F (1).jpg", 0)
    method = Method()
    kps, descriptors = method.detectAndDescribe(image, 'orb')
    print(len(kps))
    print(descriptors.shape)
    print(kps[0])
    print(descriptors[0, :])