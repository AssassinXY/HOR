import cv2
import numpy as np  

img = cv2.imread("/data/AssassionXY/Deepfake-Detection/0000.png", 0)
 
img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img, 50, 150)//单层
 
#cv2.imshow('Canny', canny)
cv2.imwrite('1.png', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
