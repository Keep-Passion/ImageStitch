#coding:utf-8
from jpype import *
import os


def phaseCorrelation(dir1, dir2):
    # dir1 = dir1.replace("\\", "/")
    # dir2 = dir2.replace("\\", "/")
    # dir1 = "C:/Coding_Test/Python/ImageStitch/" + dir1
    # dir2 = "C:/Coding_Test/Python/ImageStitch/" + dir2
    # print(dir1)
    # print(dir2)
    jarpath = os.path.join(os.path.abspath('.'), './jar/')
    # demo1 打印hello world
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % (jarpath + 'Stitching.jar'))
    ImagePlus = JClass("ij.ImagePlus")
    StichingClass = JClass("plugin.Stitching_Pairwise")
    PairWiseStitchingImgLib = JClass("mpicbg.stitching.PairWiseStitchingImgLib")
    StitchingParameters = JClass("mpicbg.stitching.StitchingParameters")

    imp1 = ImagePlus(dir1)
    imp2 = ImagePlus(dir2)
    params = StitchingParameters()
    ##拼接参数设置，与界面统一
    """
    ----  params系列属性说明: -----
    dimensionality，维度，int
    fusionMethod，融合方法，int
        { 0="Linear Blending",
        1="Average", 
        2="Median", 
        3="Max. Intensity", 
        4="Min. Intensity", 
        5="Intensity of random input tile", 
        6="Overlay into composite image", 
        7="Do not fuse images" };
    checkPeaks，检查峰值数目，int
    ignoreZeroValuesFusion，融合时忽略0值（黑色），bool
    displayFusion，显示融合结果，bool
    computeOverlap，计算覆盖大小，bool
    subpixelAccuracy，亚像素精度，bool
    xOffset = 0，yOffset = 0，zOffset = 0;，初始偏移量，0
    channell，channel2 两图的通道选择，int {0 = "Average all channels","Only channel1"}
    timeSelect，连续帧数选择，int
    
    """
    if (imp1.getNSlices() > 1 | imp2.getNSlices() > 1):
        dimensionality = 3
    else:
        dimensionality = 2
    params.dimensionality = dimensionality
    params.fusionMethod = 7
    params.checkPeaks = 5
    params.ignoreZeroValuesFusion = True
    params.displayFusion = False
    params.computeOverlap = True
    params.subpixelAccuracy = False
    params.xOffset = 0.0
    params.yOffset = 0.0
    params.zOffset = 0.0
    params.channel1 = 1
    params.channel2 = 1
    params.timeSelect = 0  # 单帧恒取0
    result = PairWiseStitchingImgLib.stitchPairwise(imp1, imp2, imp1.getRoi(), imp2.getRoi(), 1, 1, params);
    offset = list(result.offset)
    shutdownJVM()
    print(offset)
    return offset



