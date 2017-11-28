# coding=utf-8
import cv2.cv as cv

# 视频采样算法
def sampling(src):
    capture = cv.CaptureFromFile(src)

    nbFrames = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))

    fps = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)


    print 'Num. Frames = ', nbFrames
    print 'Frame Rate = ', fps, 'fps'
    i = 0
    for f in xrange( nbFrames ):
        frameImg = cv.QueryFrame(capture)
        ROI = frameImg[0:frameImg.height - 2, 0:]
        if f % int(fps/3) == 0:
           cv.SaveImage("LPtest/sam"+ str(i).zfill(5) +".jpg", ROI)
           i += 1
