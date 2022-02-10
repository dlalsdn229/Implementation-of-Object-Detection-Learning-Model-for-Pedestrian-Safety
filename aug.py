import numpy as np
import cv2


# 원본 이미지
img_source = cv2.imread('3.jpg')
cv2.imshow("original", img_source)

cv2.waitKey(0)


# 이미지 이동
height, width = img_source.shape[:2]
M = np.float32([[1, 0, 0], [0, 1, 10]]) # 이미지를 오른쪽으로 100, 아래로 25 이동시킵니다.
img_translation = cv2.warpAffine(img_source, M, (width,height))
cv2.imshow("translation", img_translation)

cv2.waitKey(0)

cv2.destroyAllWindows()