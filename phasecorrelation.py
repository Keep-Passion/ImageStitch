# coding:utf-8
from jpype import *
import os
import time
import cv2
import ImageFusion
from ImageUtility import *

class phaseCorrelation(Method):
    fuseMethod = "notFuse"
    def __init__(self):
        jarpath = os.path.join(os.path.abspath('.'), './jar/')  # jar所在路径
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % (jarpath + 'Stitching_.jar'))  # 包名
        StitchingParameters = JClass("mpicbg.stitching.StitchingParameters")
        self.params = StitchingParameters()  ##拼接参数设置，与界面统一

        """
        ----  系列参数说明: -----

        # 拼接方式
        gridType, gridOrder (int)(int){
            // 0: Grid: row-by-row
            {
                0: "Right & Down",
               1: "Left & Down",
                2: "Right & Up", 
                3: "Left & Up" 
            }
            // 1: Grid: column-by-column
            { 
                0:"Down & Right", 
                1:"Down & Left", 
                2:"Up & Right", 
                3:"Up & Left" 
            }
            // 2: Grid: snake by rows
            {
                0: "Right & Down",
                1: "Left & Down",
                2: "Right & Up",
                3: "Left & Up" };
            // 3: Grid: snake by columns
            {
                0: "Down & Right",
                1: "Down & Left",
                2: "Up & Right",
                3: "Up & Left" 
            }
            // 4: Filename defined position
            {
                0:"Defined by filename"
            }
            // 5: Unknown position
            {
                0: "All files in directory"
            }
            // 6: Positions from file
            {
                0: "Defined by TileConfiguration",
                1: "Defined by image metadata" 
            }
            // 7: Sequential Images
            {
                0: "All files in directory"
            }
        }

        # 融合方法
        fusionMethod (int)
        {
            0: "Linear Blending",
            1: "Average", "Median", 
            2: "Max. Intensity", 
            3: "Min. Intensity", 
            4: "Intensity of random input tile", 
            5: "Do not fuse images (only write TileConfiguration)"
        }

        # Regression_threshold
        regThreshold (float)
        default = 2.0

        # Max/avg_displacement_threshold
        relativeThreshold (float)
        default = 2.0

        # Absolute_displacement_threshold
        absoluteThreshold (float)
        default = 2.0

        # Frame range to compare
        seqRange (int)
        default = 0

        #计算重合区域
        computeOverlap (bool)

        #添加ROI
        addTilesAsRois (bool)

        #输出参数
        outputVariant =
        0 : Fuse and display,
        1 : Write to disk

        #内存占用选项
        cpuMemSelect = 
        0: "Save memory (but be slower)", 
        1: "Save computation time (but use more RAM)" 

        """
        # 拼接方式 = (7,0)
        # ------请勿修改------
        self.params.computeOverlap = True
        self.params.sequential = True
        self.params.displayFusion = False
        self.params.channel1 = 0
        self.params.channel2 = 0
        self.params.timeSelect = 0
        self.params.checkPeaks = 5

        # ------请勿修改------

        seriesFile = None
        self.confirmFiles = False
        self.params.fusionMethod = 0
        self.params.regThreshold = 0.1
        self.params.relativeThreshold = 2.5
        self.params.absoluteThreshold = 3.5
        self.params.seqRange = 1

        self.params.addTilesAsRois = False

        self.params.invertX = False
        self.params.invertY = False
        self.params.ignoreZStage = False

        self.params.subpixelAccuracy = False
        self.params.downSample = False
        self.params.virtual = False
        self.params.cpuMemChoice = 1
        self.params.outputVariant = 1
        self.params.outputDirectory = ""

    def shutdown(self):
        shutdownJVM()

    def getFilelist(self, dir):
        image_files = sorted(os.listdir(dir))
        for img in image_files:
            if img.split(".")[-1].lower() not in ["jpg", "jpeg", "png"]:
                image_files.remove(img)

        fulldir = []
        for img in image_files:
            if ".txt" not in format(img):
                fulldir.append(dir + format(img))
        return fulldir

    def PhaseCorrelationSequential(self, dirIn, TXTOut):
        MyStitching = JClass("plugin.MyStitching")
        optimized = MyStitching().calculateElements(self.params, dirIn, TXTOut)
        return optimized

    def phaseCorrelation(self, dir1, dir2):
        PairWiseStitchingImgLib = JClass("mpicbg.stitching.PairWiseStitchingImgLib")
        HpStack = JClass("fiji.stacks.Hyperstack_rearranger")
        ImagePlus = JClass("ij.ImagePlus")
        imp1 = ImagePlus(dir1)
        imp2 = ImagePlus(dir2)
        imp1 = HpStack.convertToHyperStack(imp1)
        imp1.hide()
        imp2 = HpStack.convertToHyperStack(imp2)
        imp2.hide()

        if (imp1.getNSlices() > 1 | imp2.getNSlices() > 1):
            dimensionality = 3
        else:
            dimensionality = 2
        self.params.dimensionality = dimensionality

        result = PairWiseStitchingImgLib.stitchPairwise(imp1, imp2, imp1.getRoi(), imp2.getRoi(), 1, 1, self.params)
        return result.offset

    def flowStitch(self, fileAddress):
        startTime = time.time()
        dirIn = fileAddress
        TXTOut = "TileConfiguration.txt"
        elements = self.PhaseCorrelationSequential(dirIn, TXTOut)
        ## dx, dy 是所有图对图 1 的偏移，要得到相邻两张两两相减即可
        globaloffsets = []
        offsetList = []
        for element in elements:
            globaloffsets.append([int(element.getModel().ty), int(element.getModel().tx)])

        offsetList.append(globaloffsets[0])
        for j in range(1, len(globaloffsets)):
            offsetList.append([globaloffsets[j][0] - globaloffsets[j - 1][0], globaloffsets[j][1] - globaloffsets[j - 1][1]])
        if offsetList[0][0] == offsetList[0][1] == 0:
            offsetList.remove(offsetList[0])
        endTime = time.time()
        self.printAndWrite("The time of registering is " + str(endTime - startTime) + "s")
        self.printAndWrite("  The offsetList is " + str(offsetList))

        # stitching and fusing
        self.printAndWrite("start stitching")
        startTime = time.time()
        fileList = self.getFilelist(fileAddress)
        print(fileList)
        dxSum = 0;
        dySum = 0;
        stitchImage = cv2.imread(fileList[0], 0)
        offsetListNum = len(offsetList)
        for fileIndex in range(0, offsetListNum):
            self.printAndWrite("  stitching " + str(fileList[fileIndex + 1]))
            imageB = cv2.imread(fileList[fileIndex + 1], 0)
            dxSum = offsetList[fileIndex][0] + dxSum
            dySum = offsetList[fileIndex][1] + dySum
            offset = [dxSum, dySum]
            self.printAndWrite("  The offsetX is " + str(offsetList[fileIndex][0]) + " and the offsetY is " + str(
                offsetList[fileIndex][1]))
            self.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
            (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = self.getStitchByOffset(
                [stitchImage, imageB], offset)
            if dxSum < 0:
                dxSum = 0
            if dySum < 0:
                dySum = 0
        endTime = time.time()
        self.printAndWrite("The time of fusing is " + str(endTime - startTime) + "s")
        return (True, stitchImage)

    # def flowStitch(self, fileAddress):
    #     startTime = time.time()
    #     dirIn = fileAddress
    #     TXTOut = "TileConfiguration.txt"
    #     elements = self.PhaseCorrelationSequential(dirIn, TXTOut)
    #     ## dx, dy 是所有图对图 1 的偏移，要得到相邻两张两两相减即可
    #     offsetList = []
    #     for element in elements:
    #         offsetList.append([int(element.getModel().ty), int(element.getModel().tx)])
    #     endTime = time.time()
    #     self.printAndWrite("The time of registering is " + str(endTime - startTime) + "s")
    #     self.printAndWrite("  The offsetList is " + str(offsetList))
    #
    #     if offsetList[0][0] == offsetList[0][1] == 0:
    #         offsetList.remove(offsetList[0])
    #
    #     # stitching and fusing
    #     self.printAndWrite("start stitching")
    #     startTime = time.time()
    #     fileList = self.getFilelist(fileAddress)
    #     print(fileList)
    #     stitchImage = cv2.imread(fileList[0], 0)
    #     offsetListNum = len(offsetList)
    #     for fileIndex in range(0, offsetListNum - 1):
    #         self.printAndWrite("  stitching " + str(fileList[fileIndex + 1]))
    #         imageB = cv2.imread(fileList[fileIndex + 1], 0)
    #         offset = [offsetList[fileIndex][0], offsetList[fileIndex][1]]
    #         self.printAndWrite("  The offsetX is " + str(offsetList[fileIndex][0]) + " and the offsetY is " + str(
    #             offsetList[fileIndex][1]))
    #         (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = self.getStitchByOffset(
    #             [stitchImage, imageB], offset)
    #     endTime = time.time()
    #     self.printAndWrite("The time of fusing is " + str(endTime - startTime) + "s")
    #     return (True, stitchImage)

    def imageSetStitch(self, projectAddress, outputAddress, fileNum, startNum = 1, outputfileExtension = "jpg"):
        for i in range(startNum, fileNum+1):
            fileAddress = projectAddress + "\\" + str(i) + "\\"
            if not os.path.exists(outputAddress):
                os.makedirs(outputAddress)
            self.outputAddress = outputAddress
            (status, result) = self.flowStitch(fileAddress)
            #if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "." + outputfileExtension, result)
            if status == False:
                print("stitching Failed")

    def getStitchByOffset(self, images, offset):
        (imageA, imageB) = images
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        dx = offset[0]; dy = offset[1]

        if abs(dy) >= abs(dx):
            direction = "horizontal"
        elif abs(dy) < abs(dx):
            direction = "vertical"

        if dx >= 0 and dy >= 0:
            # The first image is located at the left top, the second image located at the right bottom
            stitchImage = np.zeros((max(hA, dx + hB), max(dy + wB, wA)), dtype=np.uint8)
            roi_ltx = dx; roi_lty = dy
            roi_rbx = min(dx + hB, hA); roi_rby = min(dy + wB, wA)
            stitchImage[0: hA, 0:wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[dx: dx+hB, dy: dy+wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx >= 0 and dy < 0:
            # The first image is located at the right top, the second image located at the left bottom
            stitchImage = np.zeros((max(hA, dx + hB), -dy + wA), dtype=np.uint8)
            roi_ltx = dx; roi_lty = -dy
            roi_rbx = hA;  roi_rby = min(-dy + wA, wB)
            stitchImage[0: hA, -dy:-dy + wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[dx: dx+hB, 0: wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx < 0 and dy >= 0:
            # The first image is located at the left bottom, the second image located at the right top
            stitchImage = np.zeros((max(-dx + hA, hB), max(dy + wB, wA)), dtype=np.uint8)
            roi_ltx = -dx; roi_lty = dy
            roi_rbx = min(-dx + hA, hB);  roi_rby = min(dy + wB, wA)
            stitchImage[-dx: -dx + hA, 0: wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[0: hB, dy: dy + wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        elif dx < 0 and dy < 0:
            # The first image is located at the right bottom, the second image located at the left top
            stitchImage = np.zeros((-dx + hA, -dy + wA), dtype=np.uint8)
            roi_ltx = -dx; roi_lty = -dy
            roi_rbx = wA;  roi_rby = hA
            stitchImage[-dx: -dx + hA, -dy: -dy + wA] = imageA
            roiImageRegionA = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
            stitchImage[0: hB, 0: wB] = imageB
            roiImageRegionB = stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby].copy()
        fuseRegion = self.fuseImage([roiImageRegionA, roiImageRegionB])
        stitchImage[roi_ltx: roi_rbx, roi_lty: roi_rby] = fuseRegion.copy()
        return (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB)

    def fuseImage(self, images):
        (imageA, imageB) = images
        fuseRegion = np.zeros(imageA.shape, np.uint8)
        imageA[imageA == 0] = imageB[imageA == 0]
        imageB[imageB == 0] = imageA[imageB == 0]
        imageFusion = ImageFusion.ImageFusion()
        if self.fuseMethod == "notFuse":
            fuseRegion = imageB
        elif self.fuseMethod == "average":
            fuseRegion = imageFusion.fuseByAverage(images)
        elif self.fuseMethod == "maximum":
            fuseRegion = imageFusion.fuseByMaximum(images)
        elif self.fuseMethod == "minimum":
            fuseRegion = imageFusion.fuseByMinimum(images)
        elif self.fuseMethod == "fadeInAndFadeOut":
            fuseRegion = imageFusion.fuseByFadeInAndFadeOut(images,self.direction)
        elif self.fuseMethod == "multiBandBlending":
            fuseRegion = imageFusion.fuseByMultiBandBlending(images)
        elif self.fuseMethod == "trigonometric":
            fuseRegion = imageFusion.fuseByTrigonometric(images,self.direction)
        elif self.fuseMethod == "optimalSeamLine":
            fuseRegion = imageFusion.fuseByOptimalSeamLine(images, self.direction)
        return fuseRegion

if __name__ == '__main__':
    phase = phaseCorrelation()
    dirIn = "./images/zirconSmall/5/"
    TXTOut = "TileConfiguration.txt"
    elements = phase.PhaseCorrelationSequential(dirIn, TXTOut)
    ## dx, dy 是所有图对图 1 的偏移，要得到相邻两张两两相减即可

    globaloffsets = []
    offsets = []
    for element in elements:
        globaloffsets.append([int(element.getModel().tx), int(element.getModel().ty)])

    offsets.append(globaloffsets[0])
    for j in range(1, len(globaloffsets)):
        offsets.append([globaloffsets[j][0] - globaloffsets[j - 1][0], globaloffsets[j][1] - globaloffsets[j - 1][1]])
    print(globaloffsets)
    print(len(globaloffsets))
    print(offsets)
    print(len(offsets))
    # print(dx, dy)

    # shutdownJVM()
