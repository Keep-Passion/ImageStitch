from Stitcher import Stitcher


def stitchWithFeature():
    Stitcher.featureMethod = "surf"             # "sift","surf" or "orb"
    Stitcher.isGPUAvailable = True
    Stitcher.searchRatio = 0.75                 # 0.75 is common value for matches
    Stitcher.offsetCaculate = "mode"              # "mode" or "ransac"
    Stitcher.offsetEvaluate = 3                   # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    Stitcher.roiRatio = 0.2                       # roi length for stitching in first direction
    Stitcher.fuseMethod = "fadeInAndFadeOut"              # "notFuse","average","maximum","minimum","fadeInAndFadeOut","trigonometric", "multiBandBlending"
    stitcher = Stitcher()

    Stitcher.direction = 1;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\iron"
    outputAddress = "result\\iron" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMutiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearchIncre,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 1;  Stitcher.directIncre = 1;
    projectAddress = "demoImages\\dendriticCrystal"
    outputAddress = "result\\dendriticCrystal" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMutiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearchIncre,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconBSE"
    outputAddress = "result\\zirconBSE" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMutiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconCL"
    outputAddress = "result\\zirconCL" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMutiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconREM"
    outputAddress = "result\\zirconREM" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMutiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconTEM"
    outputAddress = "result\\zirconTEM" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMutiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")


if __name__=="__main__":
    stitchWithFeature()
