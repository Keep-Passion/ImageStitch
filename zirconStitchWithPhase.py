from phasecorrelation import *

fileAddress = "./images/zirconSmall/3/"

outputAddress = "result\\stitchResult.png"

def zirconFlowStitch(fileAddress):
    phase = phaseCorrelation()
    (status, result) = phase.flowStitch(fileAddress)
    cv2.imwrite(outputAddress, result)

def zirconImageSetStitch():
    projectAddress = "images\\zirconSmall"
    method = "phaseCorrelate"
    phase = phaseCorrelation()
    phase.fuseMethod = "notFuse"

    outputAddress = "result\\" + method + "\\zirconSmall" + str.capitalize(phase.fuseMethod) + "\\"
    phase.imageSetStitch(projectAddress, outputAddress, 51, startNum=1, outputfileExtension="jpg")

if __name__ == '__main__':
    zirconFlowStitch(fileAddress)
    # zirconImageSetStitch()