import numpy as np
import cv2
import math
import ImageUtility as Utility


class ImageFusion(Utility.Method):

    isColorMode = False

    # 图像融合类，目前只编写传统方法
    def fuseByAverage(self, images):
        '''
        功能：均值融合
        :param images: 输入两个相同区域的图像
        :return:融合后的图像
        '''
        (imageA, imageB) = images
        # 由于相加后数值可能溢出，需要转变类型
        fuseRegion = np.uint8((imageA.astype(int) + imageB.astype(int)) / 2)
        return fuseRegion

    def fuseByMaximum(self, images):
        '''
        功能：最大值融合
        :param images: 输入两个相同区域的图像
        :return:融合后的图像
        '''
        (imageA, imageB) = images
        fuseRegion = np.maximum(imageA, imageB)
        return fuseRegion

    def fuseByMinimum(self, images):
        '''
        功能：最小值融合
        :param images: 输入两个相同区域的图像
        :return:融合后的图像
        '''
        (imageA, imageB) = images
        fuseRegion = np.minimum(imageA, imageB)
        return fuseRegion

    def getWeightsMatrix(self, images):
        '''
        功能：获取权值矩阵
        :param images:  输入两个相同区域的图像
        :return: weigthA,weightB
        '''
        (imageA, imageB) = images
        weightMatA = np.ones(imageA.shape, dtype=np.float32)
        weightMatB = np.ones(imageA.shape, dtype=np.float32)
        row, col = imageA.shape[:2]
        weightMatB_1 = weightMatB.copy()
        weightMatB_2 = weightMatB.copy()
        # 获取四条线的相加和，判断属于哪种模式
        compareList = []
        compareList.append(np.count_nonzero(imageA[0: row // 2, 0: col // 2] > 0))
        compareList.append(np.count_nonzero(imageA[row // 2: row, 0: col // 2] > 0))
        compareList.append(np.count_nonzero(imageA[row // 2: row, col // 2: col] > 0))
        compareList.append(np.count_nonzero(imageA[0: row // 2, col // 2: col] > 0))
        # self.printAndWrite("    compareList:" + str(compareList))
        index = compareList.index(min(compareList))
        # print("index:", index)
        if index == 2:
            # 重合区域在imageA的上左部分
            # self.printAndWrite("上左")
            rowIndex = 0;
            colIndex = 0;
            for j in range(1, col):
                for i in range(row - 1, -1, -1):
                    # tempSum = imageA[i, col - j].sum()
                    if (self.isColorMode and imageA[i, col - j].sum() != -3) or (self.isColorMode is False and imageA[i, col - j] != -1):
                    # if imageA[i, col - j] != -1:
                        rowIndex = i + 1
                        break
                if rowIndex != 0:
                    break
            for i in range(col - 1, -1, -1):
                # tempSum = imageA[rowIndex, i].sum()
                if (self.isColorMode and imageA[rowIndex, i].sum() != -3) or (self.isColorMode is False and imageA[rowIndex, i] != -1):
                # if imageA[rowIndex, i] != -1:
                    colIndex = i + 1
                    break
            # 赋值
            for i in range(rowIndex + 1):
                if rowIndex == 0:
                    rowIndex = 1
                weightMatB_1[rowIndex - i, :] = (rowIndex - i) * 1 / rowIndex
            for i in range(colIndex + 1):
                if colIndex == 0:
                    colIndex = 1
                weightMatB_2[:, colIndex - i] = (colIndex - i) * 1 / colIndex
            weightMatB = weightMatB_1 * weightMatB_2
            weightMatA = 1 - weightMatB
        # elif leftCenter != 0 and bottomCenter != 0 and upCenter == 0 and rightCenter == 0:
        elif index == 3:
            # 重合区域在imageA的下左部分
            # self.printAndWrite("下左")
            rowIndex = 0;
            colIndex = 0;
            for j in range(1, col):
                for i in range(row):
                    # tempSum = imageA[i, col - j].sum()
                    if (self.isColorMode and imageA[i, col - j].sum() != -3) or (self.isColorMode is False and imageA[i, col - j] != -1):
                    # if imageA[i, col - j] != -1:
                        rowIndex = i - 1
                        break
                if rowIndex != 0:
                    break
            for i in range(col - 1, -1, -1):
                # tempSum = imageA[rowIndex, i].sum()
                if (self.isColorMode and imageA[rowIndex, i].sum() != -3) or (self.isColorMode is False and imageA[rowIndex, i] != -1):
                # if imageA[rowIndex, i] != -1:
                    colIndex = i + 1
                    break
            # 赋值
            for i in range(rowIndex, row):
                if rowIndex == 0:
                    rowIndex = 1
                weightMatB_1[i, :] = (row - i - 1) * 1 / (row - rowIndex - 1)
            for i in range(colIndex + 1):
                if colIndex == 0:
                    colIndex = 1
                weightMatB_2[:, colIndex - i] = (colIndex - i) * 1 / colIndex
            weightMatB = weightMatB_1 * weightMatB_2
            weightMatA = 1 - weightMatB
        # elif rightCenter != 0 and bottomCenter != 0 and upCenter == 0 and leftCenter == 0:
        elif index == 0:
            # 重合区域在imageA的下右部分
            # self.printAndWrite("下右")
            rowIndex = 0;
            colIndex = 0;
            for j in range(0, col):
                for i in range(row):
                    # tempSum = imageA[i, j].sum()
                    if (self.isColorMode and imageA[i, j].sum() != -3) or (self.isColorMode is False and imageA[i, j] != -1):
                    # if imageA[i, j] != -1:
                        rowIndex = i - 1
                        break
                if rowIndex != 0:
                    break
            for i in range(col):
                # tempSum = imageA[rowIndex, i].sum()
                if (self.isColorMode and imageA[rowIndex, i].sum() != -3) or (self.isColorMode is False and imageA[rowIndex, i] != -1):
                # if imageA[rowIndex, i] != -1:
                    colIndex = i - 1
                    break
            # 赋值
            for i in range(rowIndex, row):
                if rowIndex == 0:
                    rowIndex = 1
                weightMatB_1[i, :] = (row - i - 1) * 1 / (row - rowIndex - 1)
            for i in range(colIndex, col):
                if colIndex == 0:
                    colIndex = 1
                weightMatB_2[:, i] = (col - i - 1) * 1 / (col - colIndex - 1)
            weightMatB = weightMatB_1 * weightMatB_2
            weightMatA = 1 - weightMatB
        # elif upCenter != 0 and rightCenter != 0 and leftCenter == 0 and bottomCenter == 0:
        elif index == 1:
            # 重合区域在imageA的上右部分
            # self.printAndWrite("上右")
            rowIndex = 0;
            colIndex = 0;
            for j in range(0, col):
                for i in range(row - 1, -1, -1):
                    # tempSum = imageA[i, j].sum()
                    if (self.isColorMode and imageA[i, j].sum() != -3) or ((self.isColorMode is False) and imageA[i, j] != -1):
                        rowIndex = i + 1
                        break
                if rowIndex != 0:
                    break
            for i in range(col):
                # tempSum = imageA[rowIndex, i].sum()
                if (self.isColorMode and imageA[rowIndex, i].sum() != -3) or ((self.isColorMode is False) and imageA[rowIndex, i] != -1):
                    colIndex = i - 1
                    break
            for i in range(rowIndex + 1):
                if rowIndex == 0:
                    rowIndex = 1
                weightMatB_1[rowIndex - i, :] = (rowIndex - i) * 1 / rowIndex
            for i in range(colIndex, col):
                if colIndex == 0:
                    colIndex = 1
                weightMatB_2[:, i] = (col - i - 1) * 1 / (col - colIndex - 1)
            weightMatB = weightMatB_1 * weightMatB_2
            weightMatA = 1 - weightMatB
        # print(weightMatA)
        # print(weightMatB)
        return (weightMatA, weightMatB)

    def fuseByFadeInAndFadeOut(self, images, dx, dy):
        '''
        功能：渐入渐出融合
        :param images:输入两个相同区域的图像
        :param direction: 横向拼接还是纵向拼接
        :return:融合后的图像
        '''
        # print("dx=", dx, "dy=", dy)
        (imageA, imageB) = images
        # cv2.imshow("A", imageA.astype(np.uint8))
        # cv2.imshow("B", imageB.astype(np.uint8))
        # cv2.waitKey(0)
        # self.printAndWrite("dx={}, dy={}".format(dx, dy))
        row, col = imageA.shape[:2]
        weightMatA = np.ones(imageA.shape, dtype=np.float32)
        weightMatB = np.ones(imageA.shape, dtype=np.float32)
        # self.printAndWrite("    ratio: "  + str(np.count_nonzero(imageA > -1) / imageA.size))
        if np.count_nonzero(imageA > -1) / imageA.size > 0.65:
            # self.printAndWrite("直接融合")
            # 如果对于imageA中，非0值占比例比较大，则认为是普通融合
            # 根据区域的行列大小来判断，如果行数大于列数，是水平方向
            if col <= row:
                # self.printAndWrite("普通融合-水平方向")
                for i in range(0, col):
                    if dy >= 0:
                        weightMatA[:, col - i - 1] = weightMatA[:, col - i - 1] * i * 1.0 / col
                        weightMatB[:, i] = weightMatB[:, i] * i * 1.0 / col
                        # weightMatA[:, i] = weightMatA[:, i] * i * 1.0 / col
                        # weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * i * 1.0 / col
                    elif dy < 0:
                        weightMatA[:, col - i - 1] = weightMatA[:, col - i - 1] * (col - i) * 1.0 / col
                        weightMatB[:, i] = weightMatB[:, i] * (col - i) * 1.0 / col
                        # weightMatA[:, i] = weightMatA[:, i] * (col - i) * 1.0 / col
                        # weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * (col - i) * 1.0 / col
            # 根据区域的行列大小来判断，如果列数大于行数，是竖直方向
            elif row < col:
                # self.printAndWrite("普通融合-竖直方向")
                for i in range(0, row):
                    if dx <= 0:
                        weightMatA[i, :] = weightMatA[i, :] * i * 1.0 / row
                        weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * i * 1.0 / row
                    elif dx > 0:
                        weightMatA[i, :] = weightMatA[i, :] * (row - i) * 1.0 / row
                        weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * (row - i) * 1.0 / row
        else:
            # 如果对于imageA中，非0值占比例比较小，则认为是拐角融合
            # self.printAndWrite("拐角融合")
            weightMatA, weightMatB = self.getWeightsMatrix(images)
        imageA[imageA < 0] = imageB[imageA < 0]
        result = weightMatA * imageA.astype(np.int) + weightMatB * imageB.astype(np.int)
        result[result < 0] = 0;     result[result > 255] = 255
        fuseRegion = np.uint8(result)
        return fuseRegion

    def fuseByTrigonometric(self, images, dx, dy):
        '''
        功能：三角函数融合
        引用自《一种三角函数权重的图像拼接算法》知网
        :param images:输入两个相同区域的图像
        :param direction: 横向拼接还是纵向拼接
        :return:融合后的图像
        '''
        (imageA, imageB) = images
        row, col = imageA.shape[:2]
        weightMatA = np.ones(imageA.shape, dtype=np.float64)
        weightMatB = np.ones(imageA.shape, dtype=np.float64)
        # self.printAndWrite("    ratio: " + str(np.count_nonzero(imageA > -1) / imageA.size))
        if np.count_nonzero(imageA > -1) / imageA.size > 0.65:
            # 如果对于imageA中，非0值占比例比较大，则认为是普通融合
            # 根据区域的行列大小来判断，如果行数大于列数，是水平方向
            if col <= row:
                # self.printAndWrite("普通融合-水平方向")
                for i in range(0, col):
                    if dy >= 0:
                        weightMatA[:, i] = weightMatA[:, i] * i * 1.0 / col
                        weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * i * 1.0 / col
                    elif dy < 0:
                        weightMatA[:, i] = weightMatA[:, i] * (col - i) * 1.0 / col
                        weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * (col - i) * 1.0 / col
            # 根据区域的行列大小来判断，如果列数大于行数，是竖直方向
            elif row < col:
                # self.printAndWrite("普通融合-竖直方向")
                for i in range(0, row):
                    if dx <= 0:
                        weightMatA[i, :] = weightMatA[i, :] * i * 1.0 / row
                        weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * i * 1.0 / row
                    elif dx > 0:
                        weightMatA[i, :] = weightMatA[i, :] * (row - i) * 1.0 / row
                        weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * (row - i) * 1.0 / row
        else:
            # 如果对于imageA中，非0值占比例比较小，则认为是拐角融合
            # self.printAndWrite("拐角融合")
            weightMatA, weightMatB = self.getWeightsMatrix(images)

        weightMatA = np.power(np.sin(weightMatA * math.pi / 2), 2)
        weightMatB = 1 - weightMatA

        imageA[imageA < 0] = imageB[imageA < 0]
        result = weightMatA * imageA.astype(np.int) + weightMatB * imageB.astype(np.int)
        result[result < 0] = 0;     result[result > 255] = 255
        fuseRegion = np.uint8(result)
        return fuseRegion

    # 多样条融合方法
    def fuseByMultiBandBlending(self, images):
        """
        功能：多带样条融合
        :param images:
        :return:
        """
        (imageA, imageB) = images
        imagesReturn = np.uint8(self.BlendArbitrary2(imageA, imageB, 4))
        return imagesReturn

    def BlendArbitrary(self, img1, img2, R, level):
        """
        功能：带权拉普拉斯融合
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param R:
        :param level: 金字塔权重
        :return:
        """
        # img1 and img2 have the same size
        # R represents the region to be combined
        # level is the expected number of levels in the pyramid

        LA, GA = self.LaplacianPyramid(img1, level)
        LB, GB = self.LaplacianPyramid(img2, level)
        GR = self.GaussianPyramid(R, level)
        GRN = []
        for i in range(level):
            GRN.append(np.ones((GR[i].shape[0], GR[i].shape[1])) - GR[i])
        LC = []
        for i in range(level):
            LC.append(LA[i] * GR[level - i -1] + LB[i] * GRN[level - i - 1])
        result = self.reconstruct(LC)
        return  result

    def BlendArbitrary2(self, img1, img2, level):
        # img1 and img2 have the same size
        # R represents the region to be combined
        # level is the expected number of levels in the pyramid
        LA, GA = self.LaplacianPyramid(img1, level)
        LB, GB = self.LaplacianPyramid(img2, level)
        LC = []
        for i in range(level):
            LC.append(LA[i] * 0.5 + LB[i] * 0.5)
        result = self.reconstruct(LC)
        return result

    def LaplacianPyramid(self, img, level):
        gp = self.GaussianPyramid(img, level)
        lp = [gp[level-1]]
        for i in range(level - 1, -1, -1):
            GE = cv2.pyrUp(gp[i])
            GE = cv2.resize(GE, (gp[i - 1].shape[1], gp[i - 1].shape[0]), interpolation=cv2.INTER_CUBIC)
            L = cv2.subtract(gp[i - 1], GE)
            lp.append(L)
        return lp, gp

    def reconstruct(self, input_pyramid):
        out = input_pyramid[0]
        for i in range(1, len(input_pyramid)):
            out = cv2.pyrUp(out)
            out = cv2.resize(out, (input_pyramid[i].shape[1],input_pyramid[i].shape[0]), interpolation = cv2.INTER_CUBIC)
            out = cv2.add(out, input_pyramid[i])
        return out

    def GaussianPyramid(self, R, level):
        G = R.copy().astype(np.float64)
        gp = [G]
        for i in range(level):
            G = cv2.pyrDown(G)
            gp.append(G)
        return gp

    #权值矩阵归一化
    def stretchImage(self, Region):
        minI = Region.min()
        maxI = Region.max()
        out = (Region - minI) / (maxI - minI) * 255
        return out

    # OptialSeamLine's method 最佳缝合线方法
    def fuseByOptimalSeamLine(self, images, direction="horizontal"):
        '''
        基于最佳缝合线的融合方法
        :param images:输入两个相同区域的图像
        :param direction: 横向拼接还是纵向拼接
        :return:融合后的图像
        '''
        (imageA, imageB) = images
        cv2.imshow("imageA", imageA)
        cv2.imshow("imageB", imageB)
        cv2.waitKey(0)
        value = self.caculateVaule(images)
        # print(value)
        mask = 1 - self.findOptimalSeamLine(value, direction)
        # cv2.namedWindow("mask", 0)
        # cv2.imshow("mask", (mask*255).astype(np.uint8))
        # cv2.waitKey(0)
        fuseRegion = imageA.copy()
        fuseRegion[(1 - mask) == 0] = imageA[(1 - mask) == 0]
        fuseRegion[(1 - mask) == 1] = imageB[(1 - mask) == 1]
        drawFuseRegion = self.drawOptimalLine(1- mask, fuseRegion)
        cv2.imwrite("optimalLine.jpg", drawFuseRegion)
        cv2.imwrite("fuseRegion.jpg", np.uint8(self.BlendArbitrary(imageA,imageB, mask, 4)))
        cv2.waitKey(0)
        return np.uint8(self.BlendArbitrary(imageA,imageB, mask, 4))

    def caculateVaule(self, images):
        (imageA, imageB) = images
        row, col = imageA.shape[:2]
        # value = np.zeros(imageA.shape, dtype=np.float32)
        Ecolor = (imageA - imageB).astype(np.float32)
        Sx = np.array([[-2, 0, 2],
                       [-1, 0, 1],
                       [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2],
                       [ 0,  0,  0],
                       [ 2,  1,  2]])
        Egeometry = np.power(cv2.filter2D(Ecolor, -1, Sx), 2) + np.power(cv2.filter2D(Ecolor, -1, Sy), 2)

        diff = np.abs(imageA - imageB) / np.maximum(imageA, imageB).max()
        diffMax = np.amax(diff)

        infinet = 10000
        W = 10
        for i in range(0, row):
            for j in range(0, col):
                if diff[i, j] < 0.7 * diffMax:
                    diff[i, j] = W * diff[i, j] / diffMax
                else:
                    diff[i, j] = infinet
        value = diff * (np.power(Ecolor, 2) + Egeometry)
        return value

    def findOptimalSeamLine(self, value, direction="horizontal"):
        """
        功能：寻找最佳缝合线
        :param value:
        :param direction:
        :return:
        """
        if direction == "vertical":
            value = np.transpose(value)
        row, col = value.shape[:2]
        indexMatrix = np.zeros(value.shape, dtype=np.int)
        dpMatrix = np.zeros(value.shape, dtype=np.float32)
        mask = np.zeros(value.shape, dtype=np.uint8)

        dpMatrix[0, :] = value[0, :]
        indexMatrix[0, :] = indexMatrix[0, :] - 1
        for i in range(1, row):
            for j in range(0, col):
                if j == 0:
                    dpMatrix[i, j] = (np.array([dpMatrix[i - 1, j], dpMatrix[i - 1, j + 1]]) + value[i, j]).min()
                    indexMatrix[i, j] = (np.array([dpMatrix[i - 1, j], dpMatrix[i - 1, j + 1]]) + value[i, j]).argmin()
                    # print("last=" + str(np.array([dpMatrix[i - 1, j], dpMatrix[i - 1, j + 1]])))
                    # print("this=" + str(value[i, j]))
                    # print(dpMatrix[i, j])
                    # print(indexMatrix[i, j])
                elif j == col - 1:
                    dpMatrix[i, j] = (np.array([dpMatrix[i - 1, j - 1], dpMatrix[i - 1, j]]) + value[i, j]).min()
                    indexMatrix[i, j] = (np.array([dpMatrix[i - 1, j - 1], dpMatrix[i - 1, j]]) + value[i, j]).argmin() - 1
                else:
                    dpMatrix[i, j] = (np.array([dpMatrix[i - 1, j - 1], dpMatrix[i - 1, j], dpMatrix[i - 1, j + 1]]) + value[i, j]).min()
                    indexMatrix[i, j] = (np.array([dpMatrix[i - 1, j - 1], dpMatrix[i - 1, j], dpMatrix[i - 1, j + 1]]) + value[i, j]).argmin() - 1
        # print(indexMatrix)
        # generate the mask
        index = dpMatrix[row - 1, :].argmin()
        # print("here" + str(dpMatrix[row - 1, :]))
        # print(index)
        for j in range(index, col):
            mask[row-1, j] = 1
        for i in range(row - 1, 1, -1):
            index = indexMatrix[i, index] + index
            # print(index)
            for j in range(index, col):
                mask[i-1, j] = 1
        if direction == "vertical":
            mask = np.transpose(mask)
        return mask

    def drawOptimalLine(self, mask, fuseRegion):
        """
        功能：绘制最佳缝合线
        :param mask:
        :param fuseRegion:
        :return:
        """
        row, col = mask.shape[:2]
        drawing = np.zeros([row, col, 3], dtype=np.uint8)
        drawing = cv2.cvtColor(fuseRegion, cv2.COLOR_GRAY2BGR)
        for j in range(0, col):
            for i in range(0, row):
                if mask[i, j] == 1:
                    drawing[i, j] = np.array([0, 0, 255])
                    break
        return drawing

if __name__=="__main__":
    # 测试
    num = 6
    A_1 = np.zeros((num, num), dtype=np.uint8)
    for i in range(num):
        for j in range(num):
            if j < 3:
                A_1[i, j] = 1
    for i in range(num):
        for j in range(num):
            if i < 3:
                A_1[i, j] = 1
    # A_1[0, num-1] = 0;A_1[1, num-1] = 0;A_1[2, num-1] = 0;
    # A_1[num-1, 0] = 0;  A_1[num-1, 1] = 0;A_1[num-1, 2] = 0;
    print(A_1)

    A_2 = np.ones((num, num), dtype=np.uint8)
    imageFusion = ImageFusion()
    imageFusion.fuseByFadeInAndFadeOut([A_1, A_2])