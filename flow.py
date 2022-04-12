
import numpy as np
import cv2
import glob
import os
import json
root = '/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/c40/videos'
dst_root = '/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/flow'
list = glob.glob(root +'/*.mp4')



def _extract_face(img_path, face, face_images_dir):
    if not face:
        return False
    img = cv2.imread(str(img_path))
    x, y, w, h = face
    cropped_face = img[y : y + h, x : x + w]  # noqa E203
    cv2.imwrite(str(face_images_dir / img_path.name), cropped_face)
    return True

















for video_path in sorted(list):

    # if  int(str(video_path)[-3:-0])=124:
    #     continue
    with open(root[0:-6]+'bounding_boxes/'+str(video_path)[-11:-4]+'.json', 'r') as f:

        faces = json.load(f)
  
    # 先统计每个视频的帧数
    frame_count = 0

    # 统计该视频帧数
    video_cap = cv2.VideoCapture(video_path)
    while(1):
        ret, frame = video_cap.read()
        if ret is False:
            break
        # all_frames.append(frame)
        frame_count = frame_count + 1
    print('total frame : ', frame_count)

    cap = cv2.VideoCapture(video_path)

    # 获取第一帧
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    print('frame.shape : ', type(frame1))
    # 遍历每一行的第1列
    hsv[..., 1] = 255
    i = 3
    jj=len(faces)
    
    while(i < frame_count - 1 + 3 - 10):

        # 不断重复读取下一帧， ret代表是否读取到该帧，为TRUE或者FALSE
        ret, frame2 = cap.read()
        # cv2.imshow('test', frame2)
        if ret==False:
            i+=1
            continue

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        




        # 返回一个两通道的光流向量，实际上是每个点的像素位移值
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        #边缘
        img = cv2.GaussianBlur(prvs,(3,3),0)
        canny = cv2.Canny(img, 50, 150)
        
        face = faces[str(i).zfill(4)]
        if face==[]:
            continue
        y1, x1, y2, x2 = face[0][0],face[0][1],face[0][2],face[0][3]
        #根据josn以及帧次序进行对脸部位置划分
        #print("YES")
        ####################################
        pic_next_optical = np.zeros([img.shape[0], img.shape[1], 5])
        pic_pre_optical = np.zeros([img.shape[0], img.shape[1], 5])
        pic_both_optical = np.zeros([img.shape[0], img.shape[1], 7])
        flow = np.zeros([img.shape[0], img.shape[1], 2])




        # 创建要存储的文件夹 
        
        
        if not os.path.exists(dst_root + '/onlyflow/' + str(video_path)[-11:-4]):
            os.makedirs(dst_root + '/onlyflow/' + str(video_path)[-11:-4])
        if not os.path.exists(dst_root + '/onlyedge/' + str(video_path)[-11:-4]):
            os.makedirs(dst_root + '/onlyedge/' + str(video_path)[-11:-4])
        if not os.path.exists(dst_root + '/edgeflow/' + str(video_path)[-11:-4]):
            os.makedirs(dst_root + '/edgeflow/' + str(video_path)[-11:-4])
        if not os.path.exists(dst_root + '/flow3/' + str(video_path)[-11:-4]):
            os.makedirs(dst_root + '/flow3/' + str(video_path)[-11:-4])

        # 添加光流信息的图片
        # 处理第一帧图片   frame_00003_rgb
        # if i == 0:
            # pic_next_optical[:, :, 0:3] = frame1
            # pic_next_optical[:, :, 3:5] = flow
            # #np.savez(dst_root + '/添加后一帧光流信息/' + str(video_path)[-7:-4] + '/' + 'frame_' + str(i).zfill(3) + '_rgb.npz', pic_next_optical=pic_next_optical)
            # pic_pre_optical[:, :, 0:3] = frame1
            #np.savez(dst_root + '/添加前一帧光流信息/' + str(video_path)[-7:-4] + '/' + 'frame_' + str(i).zfill(3) + '_rgb.npz', pic_pre_optical=pic_pre_optical)

        # 处理视频中间帧
        if i !=0 and i!=(frame_count - 2):
            pic_next_optical[:, :, 0:3] = frame2
            pic_next_optical[:, :, 3:5] = flow
            #np.savez(dst_root + '/添加后一帧光流信息/' + str(video_path)[-7:-4] + '/' + 'frame_' + str(i).zfill(3) + '_rgb.npz', pic_next_optical=pic_next_optical)
            pic_pre_optical[:, :, 0:3] = frame2
            pic_pre_optical[:, :, 3:5] = flow
            #np.savez(dst_root + '/添加前一帧光流信息/' + str(video_path)[-7:-4] + '/' + 'frame_' + str(i).zfill(3) + '_rgb.npz', pic_pre_optical=pic_pre_optical)
            flow[:, :, 0:2] = flow
            taflow=flow[y1 : y2, x2 : x1, :]#裁剪人脸
            canny=canny[y1 : y2, x2 : x1]
            flowedge=np.zeros([y2-y1, x1-x2, 3])
            flow3=np.zeros([y2-y1, x1-x2, 3])
            flowedge[:,:,0:2] = taflow
            flow3[:,:,0:2] = taflow
            canny3=canny.reshape((y2-y1, x1-x2, 1))
            flowedge[:,:,2:3] = canny3
            flow3[:,:,2:3] = taflow[:,:,0:1]
            #np.savez(dst_root + '/onlyflow/' + str(video_path)[-11:-4] + '/'  + str(i).zfill(3) + '.npz', pic_pre_optical=taflow)
            #cv2.imwrite(dst_root + '/onlyedge/' + str(video_path)[-11:-4] + '/'  + str(i).zfill(3) + '.png',canny)
            #np.savez(dst_root + '/edgeflow/' + str(video_path)[-11:-4] + '/'  + str(i).zfill(3) + '.npz', pic_pre_optical=flowedge)
            np.savez(dst_root + '/flow3/' + str(video_path)[-11:-4] + '/'  + str(i).zfill(3) + '.npz', pic_pre_optical=flow3)





        # 处理最后一帧图片
        # if i == (frame_count - 2):
        #     pic_next_optical[:, :, 0:3] = frame2
        #     np.savez(dst_root + '/添加后一帧光流信息/' + str(video_path)[-7:-4] + '/' + 'frame_' + str(i).zfill(3) + '_rgb.npz', pic_next_optical=pic_next_optical)
        #     pic_pre_optical[:, :, 0:3] = frame2
        #     pic_pre_optical[:, :, 3:5] = flow
            
        #     np.savez(dst_root + '/添加前一帧光流信息/' + str(video_path)[-7:-4] + '/' + 'frame_' + str(i).zfill(3) + '_rgb.npz', pic_pre_optical=pic_pre_optical)

        '''
        pic_pre_optical[:, :, 3:5] =
        pic_pre_optical[:, :, 3:5] =
        pic_both_optical[:, :, 3:5] =
        pic_both_optical[:, :, 3:5] =
        #'''

        # 笛卡尔坐标转换为极坐标，获得极轴和极角
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        #cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff

        prvs = next

        # print('frame : ', i)
        # i是本视频下的帧数
        i += 1


    cap.release()
    cv2.destroyAllWindows()





