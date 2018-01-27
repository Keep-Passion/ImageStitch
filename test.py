#coding:utf-8
from jpype import *
import os

class phaseCorrelation:
    def __init__(self):
        jarpath = os.path.join(os.path.abspath('.'), './jar/')
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % (jarpath + 'Stitching.jar'))
        StitchingParameters = JClass("mpicbg.stitching.StitchingParameters")
        self.params = StitchingParameters()

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
        self.params.fusionMethod = 7
        self.params.checkPeaks = 5
        self.params.ignoreZeroValuesFusion = True
        self.params.displayFusion = False
        self.params.computeOverlap = True
        self.params.subpixelAccuracy = False
        self.params.xOffset = 0.0
        self.params.yOffset = 0.0
        self.params.zOffset = 0.0
        self.params.channel1 = 1
        self.params.channel2 = 1
        self.params.timeSelect = 0  # 单帧恒取0

    def shutdown(self):
        shutdownJVM()

    def phaseCorrelation(self,dir1, dir2):
        PairWiseStitchingImgLib = JClass("mpicbg.stitching.PairWiseStitchingImgLib")

        ImagePlus = JClass("ij.ImagePlus")
        imp1 = ImagePlus(dir1)
        imp2 = ImagePlus(dir2)

        if (imp1.getNSlices() > 1 | imp2.getNSlices() > 1):
            dimensionality = 3
        else:
            dimensionality = 2
        self.params.dimensionality = dimensionality

        result = PairWiseStitchingImgLib.stitchPairwise(imp1, imp2, imp1.getRoi(), imp2.getRoi(), 1, 1, self.params);
        return result.offset

if __name__ == '__main__':
    phase = phaseCorrelation()
    dir1 = "images/superalloyTurbineblade/4/SHT-A5-100-071215_001.JPG"
    dir2 = "images/superalloyTurbineblade/4/FQ/SHT-A5-100-071215_002.JPG"
    offset = phase.phaseCorrelation(dir1,dir2)
    print(offset)
    phase.shutdown()

# dir1 = "F:/pycharm projects/material fusion/FQ/SHT-A5-100-071215_001.JPG"
# dir2 = "F:/pycharm projects/material fusion/FQ/SHT-A5-100-071215_002.JPG"
# offset = phaseCorrelation(dir1,dir2)
# print (offset)
