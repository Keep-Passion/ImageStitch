from Stitcher import Stitcher
import cv2
import time
import glob
import os

resultAddress = "result\\featureSearchIncre\\"

def pairwiseStitch(fileList):
    stitcher = Stitcher()
    stitcher.printAndWrite("Stitching " + str(fileList[0]) + " and " + str(fileList[1]))
    imageA = cv2.imread(fileList[0], 0)
    imageB = cv2.imread(fileList[1], 0)
    startTime = time.time()
    (status, offset) = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])
    endTime = time.time()
    if status == False:
        stitcher.printAndWrite(offset)
        return (status, offset)
    else:
        stitcher.printAndWrite("  The time of registering is " + str(endTime - startTime) + "s")
        startTime = time.time()
        (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = stitcher.getStitchByOffset([imageA, imageB], offset)
        endTime = time.time()
        stitcher.printAndWrite("  The time of fusing is " + str(endTime - startTime) + "s")
        return (status, stitchImage)

def gridStitch(fileList):
    stitcher = Stitcher()
    stitcher.printAndWrite("Stitching the directory which have" + str(fileList[0]))
    fileNum = len(fileList)
    offsetList = []

    # calculating the offset for small image
    startTime = time.time()
    for fileIndex in range(0, fileNum - 1):
        stitcher.printAndWrite("stitching" + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]))
        imageA = cv2.imread(fileList[fileIndex], 0)
        imageB = cv2.imread(fileList[fileIndex + 1], 0)
        (status, offset) = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])
        if status == False:
            return (False, "  " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]) + str(offset))
        else:
            offsetList.append(offset)
    endTime = time.time()
    stitcher.printAndWrite("The time of registering is " + str(endTime - startTime) + "s")
    stitcher.printAndWrite("  The offsetList is " + str(offsetList))
    # offsetList = [[1784, 2], [1805, 2], [1810, 2], [1775, 3], [1761, 2], [1847, 3], [1809, 1], [1813, 3], [1787, 2], [1818, 3], [1786, 2], [1803, 3], [1722, 1], [1211, 1], [-10, 2412], [-1735, -2], [-1809, -2], [-1788, -4], [-1754, -2], [-1727, -2], [-1790, -3], [-1785, -3], [-1779, -1], [-1808, -3], [-1768, -3], [-1822, -3], [-1677, -2], [-1778, -2], [-1439, -2], [-3, 2410], [1758, 2], [1792, 2], [1795, 3], [1841, 3], [1783, 3], [1802, 4], [1782, 2], [1763, 3], [1738, 3], [1837, 3], [1781, 3], [1789, 18], [1713, 1], [1270, -12], [-3, 2479], [-1787, -1], [-1812, -2], [-1823, -2], [-1763, -2], [-1726, -2], [-1885, -3], [-1754, -2], [-1747, -2], [-1667, -2], [-1875, -4], [-1696, -3], [-1673, -2], [-1816, -2], [-1411, -1], [-4, 2432], [1874, 3], [1707, -3], [1783, 3], [1795, 3], [1732, 3], [1838, 4], [1721, 1], [1783, 4], [1805, 3], [1726, 4], [1829, 2], [1775, 3], [1776, 1], [1201, 2], [-16, 2405], [-1822, -1], [-1844, -2], [-1758, -3], [-1742, -4], [-1815, -3], [-1817, -3], [-1848, -2], [-1768, -2], [-1750, -2], [-1766, -3], [-1659, -2], [-1833, -2], [-1792, -3], [-1197, -1]]
    # offsetList = [[1734, 2], [1768, 2], [1722, 0], [1772, 2], [1713, 1], [1723, 1], [1816, 2], [1835, 2], [1543, 0], [1807, 2],
    #  [1832, 2], [1794, 1], [1795, -1], [1514, 1], [-4, 2450], [-1746, -2], [-1752, -1], [-1748, -2], [-1731, -2],
    #  [-1777, -1], [-1793, -2], [-1760, -1], [-1745, -2], [-1782, -1], [-1809, -2], [-1798, -2], [-1693, 1], [-1836, -3],
    #  [-1497, 0], [-4, 2423], [1778, 2], [1747, 2], [1824, 1], [1823, 2], [1784, 2], [1771, 0], [1750, 2], [1753, 2],
    #  [1826, 0], [1770, 1], [1771, 1], [1714, 1], [1812, 1], [1351, 1], [-4, 2419], [-1774, -2], [-1735, -1],
    #  [-1800, -2], [-1790, -1], [-1748, -1], [-1789, -1], [-1812, -2], [-1762, -2], [-1783, -2], [-1822, -1],
    #  [-1748, -1], [-1663, 0], [-1770, -2], [-1523, -2], [-4, 2378], [1802, 2], [1753, 3], [1847, 1], [1757, 2],
    #  [1751, 2], [1782, 2], [1833, 1], [1792, 1], [1760, 1], [1776, 2], [1853, 1], [1842, 2], [1822, 1], [1044, -1],
    #  [-4, 2522], [-1777, -1], [-1772, -1], [-1742, -2], [-1791, -2], [-1689, -1], [-1763, -1], [-1827, -2], [-1695, -1],
    #  [-1736, -1], [-1787, -1], [-1793, -1], [-1735, 0], [-1772, -1], [-1514, 0]]

    # stitching and fusing
    stitcher.printAndWrite("start stitching")
    startTime = time.time()
    dxSum = 0; dySum = 0
    stitchImage = cv2.imread(fileList[0], 0)
    for fileIndex in range(0, fileNum - 1):
        stitcher.printAndWrite("  stitching " + str(fileList[fileIndex + 1]))
        imageB = cv2.imread(fileList[fileIndex + 1], 0)
        dxSum = offsetList[fileIndex][0] + dxSum
        dySum = offsetList[fileIndex][1] + dySum
        offset = [dxSum, dySum]
        stitcher.printAndWrite("  The offsetX is " + str(offsetList[fileIndex][0]) + " and the offsetY is " + str(offsetList[fileIndex][1]))
        stitcher.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
        (stitchImage, fuseRegion, roiImageRegionA, roiImageRegionB) = stitcher.getStitchByOffset([stitchImage, imageB], offset)
        if dxSum < 0:
             dxSum = 0
        if dySum < 0:
             dySum = 0

    endTime = time.time()
    stitcher.printAndWrite("The time of fusing is " + str(endTime - startTime) + "s")
    return (True, stitchImage)

def ironPariwiseStitch():
    # Image stitching For iron By pairwise stitching
    projectAddress = ".\\images\\iron"
    fileNum = 50
    for i in range(0, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        outputAddress = resultAddress + "iron" + str.capitalize(Stitcher.fuseMethod) + "\\"
        if not os.path.exists(outputAddress):
            os.makedirs(outputAddress)
        Stitcher.outputAddress = outputAddress
        (status, result) = pairwiseStitch(fileList)
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("stitching Failed")

def dendriticCrystalGridStitch():
    # Image stitching For iron By pairwise stitching
    projectAddress = ".\\images\\dendriticCrystal"
    fileNum = 11
    for i in range(4, fileNum):
        fileAddress = projectAddress + "\\" + str(i + 1) + "\\"
        fileList = glob.glob(fileAddress + "*.jpg")
        outputAddress = resultAddress + "dendriticCrystal" + str.capitalize(Stitcher.fuseMethod) + "\\"
        if not os.path.exists(outputAddress):
            os.makedirs(outputAddress)
        Stitcher.outputAddress = outputAddress
        (status, result) = gridStitch(fileList)
        if status == True:
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i + 1) + ".jpg", result)
        if status == False:
            print("stitching Failed")


if __name__=="__main__":
    Stitcher.featureMethod = "surf"     # "sift","surf" or "orb"
    Stitcher.searchRatio = 0.75          # 0.75 is common value for matches
    Stitcher.offsetCaculate = "mode"    # "mode" or "ransac"
    Stitcher.offsetEvaluate = 10         # 40 menas nums of matches for mode, 4.0 menas  of matches for ransac
    Stitcher.roiRatio = 0.1             # roi length for stitching in first direction
    Stitcher.fuseMethod = "notFuse"

    # ironPariwiseStitch()
    dendriticCrystalGridStitch()