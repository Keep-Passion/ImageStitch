import cv2
import glob
import Stitcher
import os


if __name__=="__main__":
    stitcher = Stitcher.Stitcher()
    stitcher.videoStitch("videos\\temp", isVideo=False)