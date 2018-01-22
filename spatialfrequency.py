import numpy as np
import cv2
import math
import time

def genMatR1(img1, img2, edgelen):
    #comparison of NSEN of a pic to generate the matrix for blending
    #img1 and img2 should be gray images with the same size
    out = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
    n = edgelen
    cols = int(math.ceil(img1.shape[1]/n))
    rows = int(math.ceil(img1.shape[0]/n))
    miniout = np.zeros((rows,cols),dtype='uint8')

    for i in xrange(rows - 1):
        for j in xrange(cols - 1):
            if NSEN(img1[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]) > NSEN(img2[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]):
                miniout[i, j] = 1
    miniout = cor(miniout)
    for i in xrange(rows - 1):
        for j in xrange(cols - 1):
            if miniout[i, j] == 1:
                out[i * n:min((i + 1) * n, img1.shape[0]), j * n:min((j + 1) * n, img1.shape[1])] = 1
    out = stretchImage(out)
    # cv2.imwrite("test.png", out)
    return out

def genMatR2(img1, img2, edgelen):
    #comparison of Spacial Frequency of a pic to generate the matrix for blending
    #img1 and img2 should be gray images with the same size
    out = np.zeros((img1.shape[0], img1.shape[1]), dtype='uint8')
    n = edgelen
    cols = int(math.ceil(img1.shape[1]/n))
    rows = int(math.ceil(img1.shape[0]/n))
    miniout = np.zeros((rows,cols),dtype='uint8')

    for i in xrange(rows - 1):
        for j in xrange(cols - 1):
            if SF(img1[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]) >= SF(img2[i * n:min((i + 1) * n, img1.shape[0]),j * n:min((j + 1) * n, img1.shape[1])]):
                miniout[i,j] = 1
    miniout = cor(miniout)
    for i in xrange(rows - 1):
        for j in xrange(cols - 1):
            if miniout[i,j] == 1:
                out[i * n:min((i + 1) * n, img1.shape[0]), j * n:min((j + 1) * n, img1.shape[1])] = 1
    return out

def NSEN(img):
    #calculate NSEN of an image
    n_num = 0.0
    xcols = int(img.shape[1])
    xrows = int(img.shape[0])
    if xcols==0 or xrows == 0:
        return 0
    # for i in xrange(xrows / 2 - 4, xrows / 2 + 4):
    #     for j in xrange(xcols / 2 - 4, xcols / 2 + 4):
    for i in xrange(xrows-1):
        for j in xrange(xcols-1):
            if (img[i, j] > img[i - 1, j] and img[i, j] > img[i + 1, j]) or (
                img[i, j] < img[i - 1, j] and img[i, j] < img[i + 1, j]) or (
                img[i, j] < img[i, j - 1] and img[i, j] < img[i, j + 1]) or (
                img[i, j] > img[i, j - 1] and img[i, j] > img[i, j + 1]) or (
                img[i, j] < img[i - 1, j - 1] and img[i, j] < img[i + 1, j + 1]) or (
                img[i, j] > img[i - 1, j - 1] and img[i, j] > img[i + 1, j + 1]) or (
                img[i, j] < img[i - 1, j + 1] and img[i, j] < img[i + 1, j - 1]) or (
                img[i, j] > img[i - 1, j + 1] and img[i, j] > img[i + 1, j - 1]):
                n_num = n_num + 1
    SEN = n_num/((xcols-2)*(xrows-2))
    # print SEN
    return SEN

def SF(img):
    RF_2 = 0
    CF_2 = 0
    xcols = int(img.shape[1])
    xrows = int(img.shape[0])
    for i in xrange(xrows-1):
        for j in xrange(xcols-1):
            RF_2 = int(RF_2) + (int(img[i, j]) - int(img[i, j - 1]))**2
            CF_2 = int(CF_2) + (int(img[i, j]) - int(img[i - 1, j]))**2
            # print img[i, j], img[i, j - 1], img[i - 1, j]
            # print RF_2, CF_2
            # print '---------'
    SF = math.sqrt(RF_2/xcols/xrows + CF_2/xcols/xrows)
    # print SF
    return SF

#correct the weight-mat with bilateralFilter
def cor(img):
    # start = time.clock()
    out = cv2.bilateralFilter(img, 40, 30, 30)
    # end = time.clock()
    # run_time = end - start
    # print run_time
    return out

def stretchImage(Region):
    minI = Region.min()
    maxI = Region.max()
    if maxI == minI:
        maxI = 1
    out = (Region - minI) / (maxI - minI) * 255
    return out

def genMat(img1, img2, len):
    return genMatR2(img1, img2, len)
