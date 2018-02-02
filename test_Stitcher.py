import cv2

stitcher = cv2.createStitcher(True)
foo = cv2.imread("C:\\Coding_Test\\Python\\ImageStitch\\images\\zirconSmall\\1\\WJE068-F (1).jpg")
bar = cv2.imread("C:\\Coding_Test\\Python\\ImageStitch\\images\\zirconSmall\\1\\WJE068-F (2).jpg")
bar2 = cv2.imread("C:\\Coding_Test\\Python\\ImageStitch\\images\\zirconSmall\\1\\WJE068-F (3).jpg")
result = stitcher.stitch((foo, bar))

cv2.imwrite("D:\\result.jpg", result[1])