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
    :return:
    '''
    (imageA, imageB) = images
    fuseRegion = np.maximum(imageA, imageB)
    return fuseRegion

def fuseByMinimum(images):
    '''
    最小值融合
    :param images: 输入两个相同区域的图像
    :return:
    '''
    (imageA, imageB) = images
    fuseRegion = np.minimum(imageA, imageB)
    return fuseRegion

def fuseByLinearBlending(images, direction="horizontal"):
    (imageA, imageB) = images
    pass