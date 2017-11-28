# coding=utf-8

import numpy as np
import cv2
import spatialfrequency as SF
from scipy.stats import mode
from matplotlib import pyplot as plt

DEVX = 0
DEVY = 0
sf_TRAN = []
def detect(img1, img2, lastResult):
    global DEVX
    global DEVY
    #   SIFT generally produces better results, but it is not FOSS, so chose the feature detector
    #   that suits the needs of your project.  ORB does OK
    use_sift = True

    if use_sift:
        detector = cv2.SURF()
    else:
        detector = cv2.ORB(1000)

    print SF.SF(img2)
    sf_TRAN.append(SF.SF(img2))
    plt.plot(sf_TRAN,".")
    # keypoints as kp, descriptors as desc
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if use_sift:
        if len(kp2) < 5000:
            print "bad pic"
            return lastResult, img1
        bf = cv2.BFMatcher()
        # This returns the top two matches for each feature point (list of list)
        pairMatches = bf.knnMatch(des1, des2, k=2)
        rawMatches = []
        for m, n in pairMatches:
            if m.distance < 0.6 * n.distance:
                rawMatches.append(m)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        rawMatches = bf.match(des1, des2)

    sortMatches = sorted(rawMatches, key=lambda x: x.distance)
    matches = sortMatches[0:300]

    #tmpimg = drawMatches.drawMatches(img1, kp1, img2, kp2, matches)
    #plt.imshow(tmpimg), plt.show()

    x_list = []
    y_list = []
    dx = []
    dy = []
    if len(matches) >= 3:
        for mat in matches:
            (x1, y1) = kp1[mat.queryIdx].pt
            (x2, y2) = kp2[mat.trainIdx].pt
            newptx, newpty  = int(x1 - x2), int(y1 - y2)
            x_list.append(newptx)
            y_list.append(newpty)
            dx.append(x1-x2)
            dy.append(y1-y2)
    else:
        print len(matches)
        print "match failed,pass"
        # img3 = drawMatches.drawMatches(img1,kp1,img2,kp2,matches)
        # cv2.imshow('test',img3)
        # cv2.waitKey(0)
        return lastResult, img1
    #
    # plt.plot(dx, dy, 'r.')
    # plt.xlabel('dx')
    # plt.ylabel('dy')
    # plt.title('points')
    # plt.show()

    rdevx = int(mode(x_list).mode)
    rdevy = int(mode(y_list).mode)
    DEVX += rdevx
    DEVY += rdevy
    # print rdevx, rdevy, DEVX, DEVY

    rows1 = lastResult.shape[0]
    cols1 = lastResult.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    # print rows1, cols1, rows2, cols2

    if (DEVX >= 0 and DEVY >= 0):
        out = np.zeros((max(rows1, DEVY + rows2), max(DEVX + cols2, cols1)), dtype='uint8')
        #左上角图1，右下角图2
        #print '1'
        ROI_ltx = DEVX
        ROI_lty = DEVY
        ROI_rbx = min(DEVX + cols2, cols1)
        ROI_rby = min(DEVY + rows2, rows1)
        out[0:rows1, 0:cols1] = lastResult.copy()
        ROIimg1 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
        out[DEVY:DEVY + rows2, DEVX:DEVX + cols2] = img2.copy()
        ROIimg2 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
    elif (DEVX >=0 and DEVY < 0):
        out = np.zeros((-DEVY + rows1, max(DEVX + cols2, cols1)), dtype='uint8')
        #左下角图1，向右上角图2
        #print '2'
        ROI_ltx = DEVX
        ROI_lty = -DEVY
        ROI_rbx = min(DEVX + cols2, cols1)
        ROI_rby = DEVY + rows2
        out[-DEVY:-DEVY + rows1,0:cols1] = lastResult.copy()
        ROIimg1 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
        out[0:rows2,DEVX:DEVX + cols2] = img2.copy()
        ROIimg2 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
        DEVY = 0
    elif (DEVX < 0 and DEVY >= 0):
        out = np.zeros((max(rows1, DEVY + rows2), -DEVX + cols1), dtype='uint8')
        #右上角图1，左下角图2
        #print '3'
        ROI_ltx = -DEVX
        ROI_lty = DEVY
        ROI_rbx = cols2
        ROI_rby = min(DEVY + rows2, rows1)
        out[0:rows1, -DEVX:-DEVX+cols1] = lastResult.copy()
        ROIimg1 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
        out[DEVY:DEVY + rows2, 0:cols2] = img2.copy()
        ROIimg2 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
        DEVX = 0
    else:
        out = np.zeros((-DEVY + rows1, -DEVX + cols1), dtype='uint8')
        #右下角图1，左上角图2
        #print '4'
        ROI_ltx = -DEVX
        ROI_lty = -DEVY
        ROI_rbx = cols2
        ROI_rby = rows2
        out[-DEVY:-DEVY + rows1, -DEVX:-DEVX + cols1] = lastResult.copy()
        ROIimg1 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
        out[0:rows2, 0:cols2] = img2.copy()
        ROIimg2 = out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx].copy()
        DEVX = 0
        DEVY = 0
    #拉普拉斯算子
    # out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx] = img_fusionLP(ROIimg1, ROIimg2)
    # cv2.imwrite("result/500x_surf_fusionLaplacian.png",out)
    #拉普拉斯金字塔0.5均值
    # out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx] = img_fusionLPyr(ROIimg1, ROIimg2)
    # cv2.imwrite("result/500x_surf_fusionLPyr.png",out)
    #拉普拉斯金字塔 空间频率矩阵
    out[ROI_lty:ROI_rby, ROI_ltx:ROI_rbx] = img_fusionWLPyr(ROIimg1, ROIimg2)
    cv2.imwrite("result/500x_surf_fusionWLPyr.png",out)

    return out, img2
#
#   将权值矩阵归一化
#
def stretchImage(Region):
    minI = Region.min()
    maxI = Region.max()
    if maxI == minI:
        maxI = 1
    out = (Region - minI) / (maxI - minI) * 255
    return out

#
#   计算拉普拉斯算子
#
def doLap(image):
    kernel_size = 5  # Size of the laplacian window
    blur_size = 5  # How big of a kernal to use for the gaussian blur
    # 保持两个值相等或接近

    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

#
#   拉普拉斯算子融合
#
def img_fusionLP(ROI1, ROI2):
    # cv2.imwrite('roi1.jpg', ROI1)
    # cv2.imwrite('roi2.jpg', ROI2)
    # cv2.waitKey(0)
    images = []
    images.append(ROI1)
    images.append(ROI2)
    for y in range(0, images[0].shape[0]):
        for x in range(0, images[0].shape[1]):
            if images[0][y, x]==0:
                images[0][y, x] = images[1][y, x]

    #使用拉普拉斯算子进行边缘提取，比较边缘清晰度进行融合（像素级）
    laps = []
    laps.append(doLap(ROI1))
    laps.append(doLap(ROI2))
    laps = np.asarray(laps)
    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    for y in xrange(0, images[0].shape[0]):
        for x in xrange(0, images[0].shape[1]):
            yxlaps = abs(laps[:, y, x])
            index = (np.where(yxlaps == max(yxlaps)))[0][0]
            output[y, x] = images[index][y, x]

    return output

#均值融合
def img_fusionLPyr(ROI1, ROI2):
    images = []
    images.append(ROI1)
    images.append(ROI2)
    level = int(np.log2(min(ROI1.shape[0],ROI1.shape[1])))
    #print level
    for y in xrange(0, images[0].shape[0]):
        for x in xrange(0, images[0].shape[1]):
            if images[0][y, x]==0:
                images[0][y, x] = images[1][y, x]
    R = SF.genMat(ROI1, ROI2, 4)
    output = BlendArbitrary(ROI1, ROI2, R,level)
    return output

#带权融合
def img_fusionWLPyr(ROI1, ROI2):
    images = []
    images.append(ROI1)
    images.append(ROI2)
    level = int(np.log2(min(ROI1.shape[0],ROI1.shape[1])))
    #print level
    for y in xrange(0, images[0].shape[0]):
        for x in xrange(0, images[0].shape[1]):
            if images[0][y, x]==0:
                images[0][y, x] = images[1][y, x]
    R = SF.genMat(ROI1, ROI2, 4)
    output = BlendArbitrary2(ROI1, ROI2, R,level)
    return output

#带权直接融合
def img_fusionDirect(ROI1, ROI2):
    images = []
    images.append(ROI1)
    images.append(ROI2)
    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    #print level
    R = SF.genMat(ROI1, ROI2, 4)
    for y in xrange(0, images[0].shape[0]):
        for x in xrange(0, images[0].shape[1]):
            if R[0][y, x]==0:
                output[y, x] = images[0][y, x]
            else:
                output[y, x] = images[1][y, x]
    return output

#带权拉普拉斯金字塔融合
def BlendArbitrary(img1, img2, R, level):
    # img1 and img2 have the same size
    # R represents the region to be combined
    # level is the expected number of levels in the pyramid

    LA, GA = LaplacianPyramid(img1, level)
    LB, GB = LaplacianPyramid(img2, level)
    GR = GaussianPyramid(R, level)
    GRN = []
    for i in xrange(level):
        GRN.append(np.ones((GR[i].shape[0], GR[i].shape[1])) - GR[i])
    LC = []
    for i in xrange(level):
        LC.append(LA[i] * GR[level - i -1] + LB[i] * GRN[level - i - 1])
    result = reconstruct(LC)
    return  result

#均值融合
def BlendArbitrary2(img1, img2, R, level):
    # img1 and img2 have the same size
    # R represents the region to be combined
    # level is the expected number of levels in the pyramid
    LA, GA = LaplacianPyramid(img1, level)
    LB, GB = LaplacianPyramid(img2, level)
    LC = []
    for i in xrange(level):
        LC.append(LA[i] * 0.5 + LB[i] * 0.5)
    result = reconstruct(LC)
    return  result

def GaussianPyramid(R, level):
    G = R.copy().astype(np.float64)
    gp = [G]
    for i in xrange(level):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def LaplacianPyramid(img, level):
    gp = GaussianPyramid(img, level)
    lp = [gp[level-1]]
    for i in xrange(level - 1, -1, -1):
        GE = cv2.pyrUp(gp[i])
        GE = cv2.resize(GE, (gp[i - 1].shape[1], gp[i - 1].shape[0]), interpolation=cv2.INTER_CUBIC)
        L = cv2.subtract(gp[i - 1], GE)
        lp.append(L)
    return lp, gp

def reconstruct(input_pyramid):
    out = input_pyramid[0]
    for i in xrange(1, len(input_pyramid)):
        out = cv2.pyrUp(out)
        out = cv2.resize(out, (input_pyramid[i].shape[1],input_pyramid[i].shape[0]), interpolation = cv2.INTER_CUBIC)
        out = cv2.add(out, input_pyramid[i])
    return out

#权值矩阵归一化
def stretchImage(Region):
    minI = Region.min()
    maxI = Region.max()
    out = (Region - minI) / (maxI - minI) * 255
    return out