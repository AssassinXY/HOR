import io
import os
import json
import numpy as np
import cv2
from sys import getsizeof
from PIL import Image
from PIL import ImageFile

filePath='/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/c40/face_images128'
list1=os.listdir(filePath)

for i in list1:
    filePath2=filePath+'/'+i
    lits2=os.listdir(filePath2)
    ii=1
    for j in lits2:
        img = cv2.imread(filePath2+'/'+j)
        #print(img.shape)
        cropped = img[15:16, 0:128]  # 裁剪坐标为[y0:y1, x0:x1]
        if ii==1:
            line=cropped
            ii=0
        else:
            line=np.concatenate((line,cropped),axis=0)
    
    cv2.imwrite("/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/c40/line/"+i+".jpg", line)