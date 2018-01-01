class Method():
    outputAddress = "result/"
    isEvaluate = True
    evaluateFile = "evaluate.txt"
    isPrintLog = True
    isParallel = False
    isNGPUWork = False

    # def __init__(self, outputAddress, isEvaluate, evaluateFile, isPrintLog, isParallel, isNGPUWork):
    #     self.outputAddress = outputAddress
    #     self.isEvaluate = isEvaluate
    #     self.evaluateFile = evaluateFile
    #     self.isPrintLog = isPrintLog
    #     self.isParallel = isParallel
    #     self.isNGPUWork = isNGPUWork

    def printAndWrite(self, content):
        if self.isPrintLog:
            print(content)
        if self.isEvaluate:
            f = open(self.outputAddress + self.evaluateFile, "a")
            f.write(content)
            f.write("\n")
            f.close()

