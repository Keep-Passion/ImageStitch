import numpy as np
import cv2

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

def fuseByLinearBlending(images, direction="horizontal"):
    '''
    线性融合
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
            weightMatB[:, col - i - 1] = weightMatA[:, col - i - 1] * (col - i) * 1.0 / col
    elif direction == "vertical":
        for i in range(0, row):
            weightMatA[i, :] = weightMatA[i, :] * (row - i) * 1.0 / row
            weightMatB[row - i - 1: i] = weightMatA[row - i - 1: i] * (row - i) * 1.0 / row
    fuseRegion = np.uint8((weightMatA * imageA) + (weightMatB * imageB))
    return fuseRegion