import numpy as np
import cv2
import glob
import os
import json
root = '/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/c40/videos'
dst_root = '/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/flow'
list = glob.glob(root +'/*.mp4')








for video_path in sorted(list):
    if os.path.exists(dst_root + '/onlyflow/' + str(video_path)[-7:-4]):
            os.rename(dst_root + '/onlyflow/' + str(video_path)[-7:-4],dst_root + '/onlyflow/' + str(video_path)[-11:-4])
    if os.path.exists(dst_root + '/onlyedge/' + str(video_path)[-7:-4]):
            os.rename(dst_root + '/onlyedge/' + str(video_path)[-7:-4],dst_root + '/onlyedge/' + str(video_path)[-11:-4])
    if os.path.exists(dst_root + '/edgeflow/' + str(video_path)[-7:-4]):
            os.rename(dst_root + '/edgeflow/' + str(video_path)[-7:-4],dst_root + '/edgeflow/' + str(video_path)[-11:-4])




    # if os.path.exists(dst_root + '/onlyflow/' + str(video_path)[-7:-4]):
    #         for file_name in os.listdir(dst_root + '/onlyflow/'):
    #             if 'npz' in file_name:
    #                 os.rename(dst_root + '/onlyflow/'+file_name,dst_root + '/onlyflow/' +'/'+file_name[0:3]+'/'+file_name[3:])
    # if os.path.exists(dst_root + '/onlyedge/' + str(video_path)[-7:-4]):
    #         for file_name in os.listdir(dst_root + '/onlyedge/'):
    #             if 'npz' in file_name:
    #                 os.rename(dst_root + '/onlyedge/'+file_name,dst_root + '/onlyedge/' +'/'+file_name[0:3]+'/'+file_name[3:])
    if os.path.exists(dst_root + '/edgeflow/' + str(video_path)[-11:-4]):
        for file_name in os.listdir(dst_root + '/edgeflow/' + str(video_path)[-11:-4]):
            os.rename(dst_root + '/edgeflow/'+ str(video_path)[-11:-4]+'/'+file_name,dst_root + '/edgeflow/'+ str(video_path)[-11:-4]+'/0'+file_name)

            
