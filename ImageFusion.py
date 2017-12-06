import numpy as np
import cv2
import math

def fuseByAverage(images):
    '''
    均值融合
    :param images: 输入两个相同区域的图像
    :return:融合后的图像
    '''
    (imageA, imageB) = images
    # 由于相加后数值溢出，需要转变类型
    fuseRegion = np.uint8((imageA.astype(int) + imageB.astype(int)) / 2)
    return fuseRegion

def fuseByMaximum(images):
    '''
    最大值融合
    :param images: 输入两个相同区域的图像
    :return:融合后的图像
    '''
    (imageA, imageB) = images
    fuseRegion = np.maximum(imageA, imageB)
    return fuseRegion

def fuseByMinimum(images):
    '''
    最小值融合
    :param images: 输入两个相同区域的图像
    :return:融合后的图像
    '''
    (imageA, imageB) = images
    fuseRegion = np.minimum(imageA, imageB)
    return fuseRegion

def fuseByFadeInAndFadeOut(images, direction="horizontal"):
    '''
    渐入渐出融合
    :param images:输入两个相同区域的图像
    :param direction: 横向拼接还是纵向拼接
    :return:融合后的图像
    '''
    (imageA, imageB) = images
    row, col = imageA.shape[:2]
    weightMatA = np.ones(imageA.shape, dtype=np.float32)
    weightMatB = np.ones(imageA.shape, dtype=np.float32)
    if direction == "horizontal":
        for i in range(0, col):
            weightMatA[:, i] = weightMatA[:, i] * (col - i) * 1.0 / col
            weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * (col - i) * 1.0 / col
    elif direction == "vertical":
        for i in range(0, row):
            weightMatA[i, :] = weightMatA[i, :] * (row - i) * 1.0 / row
            weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * (row - i) * 1.0 / row

    # 测试
    # print(weightMatA + weightMatB)
    # print("     The row fo roi region is:" + str(row))
    # print("     The col fo roi region is:" + str(col))
    # if direction == "horizontal":
    #     for i in range(0, col):
    #         print(weightMatA[0, i])
    #     print("***")
    #     for i in range(0, col):
    #         print(weightMatB[0, i])
    # elif direction == "vertical":
    #     for i in range(0, row):
    #         print(weightMatA[i, 0])
    #     print("***")
    #     for i in range(0, row):
    #         print(weightMatB[i, 0])
    fuseRegion = np.uint8((weightMatA * imageA.astype(np.int)) + (weightMatB * imageB.astype(np.int)))
    return fuseRegion

def fuseByTrigonometric(images, direction="horizontal"):
    '''
    三角函数融合
    引用自《一种三角函数权重的图像拼接算法》知网
    :param images:输入两个相同区域的图像
    :param direction: 横向拼接还是纵向拼接
    :return:融合后的图像
    '''
    (imageA, imageB) = images
    row, col = imageA.shape[:2]
    weightMatA = np.ones(imageA.shape, dtype=np.float64)
    weightMatB = np.ones(imageA.shape, dtype=np.float64)
    if direction == "horizontal":
        for i in range(0, col):
            weightMatA[:, i] = weightMatA[:, i] * (col - i) * 1.0 / col
            weightMatB[:, col - i - 1] = weightMatB[:, col - i - 1] * (col - i) * 1.0 / col
    elif direction == "vertical":
        for i in range(0, row):
            weightMatA[i, :] = weightMatA[i, :] * (row - i) * 1.0 / row
            weightMatB[row - i - 1, :] = weightMatB[row - i - 1, :] * (row - i) * 1.0 / row
    weightMatA = np.power(np.sin(weightMatA * math.pi / 2), 2)
    weightMatB = 1 - weightMatA
    # # 测试
    # print(weightMatA + weightMatB)
    # print("     The row fo roi region is:" + str(row))
    # print("     The col fo roi region is:" + str(col))
    # if direction == "horizontal":
    #     for i in range(0, col):
    #         print(weightMatA[0, i])
    #     print("***")
    #     for i in range(0, col):
    #         print(weightMatB[0, i])
    # elif direction == "vertical":
    #     for i in range(0, row):
    #         print(weightMatA[i, 0])
    #     print("***")
    #     for j in range(0, row):
    #         print(weightMatB[j, 0])
    fuseRegion = np.uint8((weightMatA * imageA.astype(np.int)) + (weightMatB * imageB.astype(np.int)))
    return fuseRegion

def fuseByOptimalSeamLine(images, direction="horizontal"):
    '''
    基于最佳缝合线的融合方法
    :param images:输入两个相同区域的图像
    :param direction: 横向拼接还是纵向拼接
    :return:融合后的图像
    '''
    (imageA, imageB) = images
    value = caculateVaule(images)
    print(value)
    print("here")
    mask = findOptimalSeamLine(value, direction)
    fuseRegion = imageA.copy()
    fuseRegion[mask == 0] = imageA[mask == 0]
    fuseRegion[mask == 1] = imageB[mask == 1]
    return fuseRegion

def fuseByMultiBandBlending(images):
    (imageA, imageB) = images
    cv2.imshow("imageA", imageA)
    cv2.imshow("imageB", imageB)
    imagesReturn = np.uint8(BlendArbitrary2(imageA, imageB, 4))
    cv2.imshow("result", imagesReturn)
    cv2.waitKey(0)
    return imagesReturn

def caculateVaule(images):
    (imageA, imageB) = images
    row, col = imageA.shape[:2]
    # value = np.zeros(imageA.shape, dtype=np.float32)
    Ecolor = (imageA - imageB).astype(np.float32)
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Sy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
    Egeometry = np.power(cv2.filter2D(Ecolor, -1, Sx), 2) + np.power(cv2.filter2D(Ecolor, -1, Sy), 2)
    diff = abs(imageA - imageB) / np.maximum(imageA, imageB)
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

def findOptimalSeamLine(value, direction="horizontal"):
    if direction == "vertical":
        value = np.transpose(value)
    row, col = value.shape[:2]
    indexMatrix = np.zeros(value.shape, dtype=np.uint8)
    dpMatrix = np.zeros(value.shape, dtype=np.float32)

    dpMatrix[0, :] = value[0, :]
    indexMatrix[0, :] = indexMatrix[0, :] - 1
    for i in range(1, row):
        for j in range(1, col):
            if j == 0:
                np.array(value[i + 1, j], value[i + 1, j + 1]).min()
                np.array(value[i + 1, j], value[i + 1, j + 1]).argmin() + 1
            elif j == col - 1:
                np.array(value[i + 1, j], value[i + 1, j - 1]).min()
                np.array(value[i + 1, j], value[i + 1, j - 1]).argmin()
            else:
                np.array(value[i + 1, j - 1], value[i + 1, j], value[i + 1, j + 1]).min()
                np.array(value[i + 1, j - 1], value[i + 1, j], value[i + 1, j + 1]).argmin()
#均值融合
def BlendArbitrary2(img1, img2, level):
    # img1 and img2 have the same size
    # R represents the region to be combined
    # level is the expected number of levels in the pyramid
    LA, GA = LaplacianPyramid(img1, level)
    LB, GB = LaplacianPyramid(img2, level)
    LC = []
    for i in range(level):
        LC.append(LA[i] * 0.5 + LB[i] * 0.5)
    result = reconstruct(LC)
    return  result

def LaplacianPyramid(img, level):
    gp = GaussianPyramid(img, level)
    lp = [gp[level-1]]
    for i in range(level - 1, -1, -1):
        GE = cv2.pyrUp(gp[i])
        GE = cv2.resize(GE, (gp[i - 1].shape[1], gp[i - 1].shape[0]), interpolation=cv2.INTER_CUBIC)
        L = cv2.subtract(gp[i - 1], GE)
        lp.append(L)
    return lp, gp

def reconstruct(input_pyramid):
    out = input_pyramid[0]
    for i in range(1, len(input_pyramid)):
        out = cv2.pyrUp(out)
        out = cv2.resize(out, (input_pyramid[i].shape[1],input_pyramid[i].shape[0]), interpolation = cv2.INTER_CUBIC)
        out = cv2.add(out, input_pyramid[i])
    return out

def GaussianPyramid(R, level):
    G = R.copy().astype(np.float64)
    gp = [G]
    for i in range(level):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

#权值矩阵归一化
def stretchImage(Region):
    minI = Region.min()
    maxI = Region.max()
    out = (Region - minI) / (maxI - minI) * 255
    return out
