from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("images/test/left_01.png")
imageB = cv2.imread("images/test/right_01.png")
# imageA = cv2.imread("images/dendriticCrystal/1/1-001.jpg")
# imageB = cv2.imread("images/dendriticCrystal/1/1-002.jpg")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True, m_method="SURF", m_direction="horizontal")

# 显示所有图片
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.imwrite("ImageCol.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()