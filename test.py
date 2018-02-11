import glob

projectAddress = "D:\\OpenCV3.3.1_Build\\x64\\vc14\\lib\\"

libList = glob.glob(projectAddress + "*.lib")
num = len(libList)

for i in range(0, num):
    if "d.lib" in libList[i].split("\\")[-1]:
        print(libList[i].split("\\")[-1])

print()

for i in range(0, num):
    if "d.lib" not in libList[i].split("\\")[-1]:
        print(libList[i].split("\\")[-1])