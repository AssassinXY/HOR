import numpy as np
import cv2
import glob
import os
import json

dst_root = '/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/flow'
list1 = glob.glob(dst_root+'/flow3/*')
for video_path in sorted(list1):
    if not os.path.exists(video_path.replace("flow3","RGBFolw3")):
        os.makedirs(video_path.replace("flow3","RGBFolw3"))
    print(video_path.replace("flow3","RGBFolw3"))
    file_list =  glob.glob(video_path+'/*')
    for file in sorted(file_list):
        flow = np.load(file)
        filepath=file.replace("flow/flow3","c40/face_images")[0:-4]+".png"
        str_list = list(filepath)
        str_list.insert(-7,"0")
        rgbpath=''.join(str_list)
        rgb  = cv2.imread(rgbpath)
        flowedge=np.zeros([rgb.shape[0], rgb.shape[1], 6])
        flowedge[:,:,0:3]=rgb
        for i in flow:
                y=flow[i]
        flowedge[:,:,3:6]=y
        np.savez(file.replace("flow3","RGBFolw3"), pic_pre_optical=flowedge)    
        


