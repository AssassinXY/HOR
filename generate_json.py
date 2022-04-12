# -*- coding:utf-8 -*-
import io
import os
import json
import numpy as np
from sys import getsizeof
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def loadFont(filename):
     f = open(filename, encoding='utf-8')
     setting = json.load(f)
     return setting
def folder(path):

    files = os.listdir(path)
    num_png = len(files)
    return num_png
def intis(str,list1,list2,list3):
    for i in range(0,len(list1)):
        if str==list1[i][0]:
         return 1
    for i in range(0,len(list2)):
        if str==list2[i][0]:
         return 2
    for i in range(0,len(list3)):
        if str==list3[i][0]:
         return 3
    return False
        

t=loadFont("train.json")
v=loadFont("val.json")
test=loadFont("test.json")

# v=loadFont("val.json")
# f=open("train_FF_FST_128_300_521.txt",'w')#360
# f2=open("train_FF_FST_Flow_128_300_521.txt",'w')
# y=open("val_FF_FST_128_300_521.txt",'w')#70
# y2=open("val_FF_FST_Flow_128_300_521.txt",'w')
# z=open("test_FF_FST_128_300_521.txt",'w')
# z2=open("test_FF_FST_Flow_128_300_521.txt",'w')


zonet=open("train_F2F_6rgbflow3.txt",'w')
zonev=open("val_F2F_Flow_6rgbflow3.txt",'w')

filePath='/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/c40/face_images'
list1=os.listdir(filePath)
# print(list1)
# i=0
# # for name in os.listdir(filePath):
#
# path=filePath+"/"+'train'
# list2=os.listdir(path)
# i = 0
# for bodyname in list2:
#     picpath=path+"/"+bodyname
#     list3=os.listdir(picpath)
#     for frname in list3:
#         frpath=picpath+"/"+frname
#         list4=os.listdir(frpath)
#         for picname in list4:
#             if frname=='real':
#                 string = frpath  + "/" + picname + " 1" + "\n"
#             else:
#                 string = frpath + "/" + picname + " 0" + "\n"
#
#             f.write(string)
#
# print("1")



for i in  range(0,300):
    num=intis(list1[i][0:3],t,v,test)
    print(i)
    if num==False:
        continue
    for j in range(0,270):
        string0="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceShifter/c40/face_images/"+list1[i]
        string1="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Deepfakes/c40/face_images/"+list1[i]#+"/"+str(j).zfill(4)+".png"+" 0"+"\n"
        string12="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Deepfakes/flow/flow3/"+list1[i]

        string2="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/c40/face_images/"+list1[i]
        string22="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/flow/flow3/"+list1[i]

        string3="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/c40/face_images/"+list1[i]
        string32="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/flow/flow3/"+list1[i]

        string4="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/c40/face_images/"+list1[i]
        string42="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/flow/flow3/"+list1[i]


        string5 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/c40/face_images/" +list1[i][0:3]#+"/"+str(j).zfill(4)+ ".png" + " 1" + "\n"
        string52="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/flow/flow3/"+list1[i][0:3]

        # if os.path.exists(string0+"/"+str(j).zfill(4)+".png"):
        #     folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceShifter/c40/face_images128/"+list1[i]
        #     if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
        #         os.makedirs(folder)
        #     if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):
        #         im = Image.open(string0+"/"+str(j).zfill(4)+".png")
        #         region = im.resize((128,128)) ##重新设定大小
        #         region.save(folder+"/"+str(j).zfill(4)+".png")
            
        #     if num==1:

        #         f.write(str(folder)+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
        #     elif num==2:
        #         y.write(str(folder)+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
        #     elif num==3:
        #         z.write(str(folder)+"/"+str(j).zfill(4)+".png"+" 0"+"\n")



        if False and os.path.exists(string1+"/"+str(j).zfill(4)+ ".png") and os.path.exists(string12+"/"+str(j).zfill(3)+ ".npz"):
            folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Deepfakes/c40/face_images128/"+list1[i][0:3]
            folder2 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Deepfakes/flow/edgeflow128/"+list1[i][0:3]
            folder3 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Deepfakes/flow/RGBFolw3/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string1+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            # if not os.path.exists(folder2):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder2)
            # if not os.path.exists(str(folder2)+"/"+str(j).zfill(4)+".npz"):  
            #     img = np.load(string12+"/"+str(j).zfill(4)+".npz")
            #     for pi in img:
            #         yy=img[pi]
            #     yy.resize((128, 128,3)) ##重新设定大小
            #     np.savez(folder2+"/"+str(j).zfill(4)+".npz", pic_pre_optical=yy)
            if not os.path.exists(folder3):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder3)
            if not os.path.exists(str(folder3)+"/"+str(j).zfill(4)+".npz"):  
                im = Image.open(string1+"/"+str(j).zfill(4)+".png")
                region = im.resize((128, 128)) ##重新设定大小
                region = np.array(region)
                img = np.load(string12+"/"+str(j).zfill(3)+".npz")
                for pi in img:
                    yy=img[pi]
                yy.resize((128, 128,3)) ##重新设定大小
                zone = np.zeros([128, 128, 6])
                zone[:,:,0:3]=region
                zone[:,:,3:6]=yy
                np.savez(folder3+"/"+str(j).zfill(4)+".npz", pic_pre_optical=zone)
            if num==1:
                #f.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #f2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonet.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            elif num==2:
                #y.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #y2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonev.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            #elif num==3:
                #z.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #z2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")





        if True and os.path.exists(string2+"/"+str(j).zfill(4)+ ".png") and os.path.exists(string22+"/"+str(j).zfill(3)+ ".npz"):
            # folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/c40/face_images128/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string2+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/c40/face_images128/"+list1[i][0:3]
            folder2 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/flow/edgeflow128/"+list1[i][0:3]
            folder3 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Face2Face/flow/RGBFolw3/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string2+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            # if not os.path.exists(folder2):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder2)
            # if not os.path.exists(str(folder2)+"/"+str(j).zfill(4)+".npz"):  
            #     img = np.load(string22+"/"+str(j).zfill(4)+".npz")
            #     for pi in img:
            #         yy=img[pi]
            #     yy.resize((128, 128,3)) ##重新设定大小
            #     np.savez(folder2+"/"+str(j).zfill(4)+".npz", pic_pre_optical=yy)
            if not os.path.exists(folder3):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder3)
            if not os.path.exists(str(folder3)+"/"+str(j).zfill(4)+".npz"):  
                im = Image.open(string2+"/"+str(j).zfill(4)+".png")
                region = im.resize((128, 128)) ##重新设定大小
                region = np.array(region)
                img = np.load(string22+"/"+str(j).zfill(3)+".npz")
                for pi in img:
                    yy=img[pi]
                yy.resize((128, 128,3)) ##重新设定大小
                zone = np.zeros([128, 128, 6])
                zone[:,:,0:3]=region
                zone[:,:,3:6]=yy
                np.savez(folder3+"/"+str(j).zfill(4)+".npz", pic_pre_optical=zone)
            if num==1:
                #f.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #f2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonet.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            elif num==2:
                #y.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #y2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonev.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            #elif num==3:
                #z.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #z2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")





        if False and os.path.exists(string3+"/"+str(j).zfill(4)+ ".png")and os.path.exists(string32+"/"+str(j).zfill(3)+ ".npz"):
            # folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/c40/face_images128/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string3+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/c40/face_images128/"+list1[i][0:3]
            folder2 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/flow/edgeflow128/"+list1[i][0:3]
            folder3 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/flow/RGBFolw3/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string3+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            # if not os.path.exists(folder2):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder2)
            # if not os.path.exists(str(folder2)+"/"+str(j).zfill(4)+".npz"):  
            #     img = np.load(string32+"/"+str(j).zfill(4)+".npz")
            #     for pi in img:
            #         yy=img[pi]
            #     yy.resize((128, 128,3)) ##重新设定大小
            #     np.savez(folder2+"/"+str(j).zfill(4)+".npz", pic_pre_optical=yy)
            if not os.path.exists(folder3):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder3)
            if not os.path.exists(str(folder3)+"/"+str(j).zfill(4)+".npz"):  
                im = Image.open(string3+"/"+str(j).zfill(4)+".png")
                region = im.resize((128, 128)) ##重新设定大小
                region = np.array(region)
                img = np.load(string32+"/"+str(j).zfill(3)+".npz")
                for pi in img:
                    yy=img[pi]
                yy.resize((128, 128,3)) ##重新设定大小
                zone = np.zeros([128, 128, 6])
                zone[:,:,0:3]=region
                zone[:,:,3:6]=yy
                np.savez(folder3+"/"+str(j).zfill(4)+".npz", pic_pre_optical=zone)
            if num==1:
                #f.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #f2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonet.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            elif num==2:
                #y.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #y2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonev.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            #elif num==3:
                #z.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #z2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")



        if False and os.path.exists(string4+"/"+str(j).zfill(4)+ ".png")and os.path.exists(string42+"/"+str(j).zfill(3)+ ".npz"):
            # folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/c40/face_images128/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string4+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/c40/face_images128/"+list1[i][0:3]
            folder2 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/flow/edgeflow128/"+list1[i][0:3]
            folder3 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/flow/RGBFolw3/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string4+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            # if not os.path.exists(folder2):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder2)
            # if not os.path.exists(str(folder2)+"/"+str(j).zfill(4)+".npz"):  
            #     img = np.load(string42+"/"+str(j).zfill(3)+".npz")
            #     for pi in img:
            #         yy=img[pi]
            #     yy.resize((128, 128,3)) ##重新设定大小
            #     np.savez(folder2+"/"+str(j).zfill(4)+".npz", pic_pre_optical=yy)
            if not os.path.exists(folder3):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder3)
            if not os.path.exists(str(folder3)+"/"+str(j).zfill(4)+".npz"):  
                im = Image.open(string4+"/"+str(j).zfill(4)+".png")
                region = im.resize((128, 128)) ##重新设定大小
                region = np.array(region)
                img = np.load(string42+"/"+str(j).zfill(3)+".npz")
                for pi in img:
                    yy=img[pi]
                yy.resize((128, 128,3)) ##重新设定大小
                zone = np.zeros([128, 128, 6])
                zone[:,:,0:3]=region
                zone[:,:,3:6]=yy
                np.savez(folder3+"/"+str(j).zfill(4)+".npz", pic_pre_optical=zone)
            if num==1:
                #f.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #f2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonet.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            elif num==2:
                #y.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #y2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
                zonev.write(folder3+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")
            #elif num==3:
                #z.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")
                #z2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 0"+"\n")





        if j%4==3 and os.path.exists(string5+"/"+str(j).zfill(4)+ ".png")and os.path.exists(string52+"/"+str(j).zfill(3)+ ".npz"):
            # folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/c40/face_images128/"+list1[i][0:3]
            # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)
            # if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
            #     im = Image.open(string5+"/"+str(j).zfill(4)+".png")
            #     region = im.resize((128, 128)) ##重新设定大小
            #     region.save(folder+"/"+str(j).zfill(4)+".png")
            folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/c40/face_images128/"+list1[i][0:3]
            folder2 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/flow/edgeflow128/"+list1[i][0:3]
            folder3 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/flow/RGBFolw3/"+list1[i][0:3]
            if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder)
            if not os.path.exists(str(folder)+"/"+str(j).zfill(4)+".png"):  
                im = Image.open(string5+"/"+str(j).zfill(4)+".png")
                region = im.resize((128, 128)) ##重新设定大小
                region.save(folder+"/"+str(j).zfill(4)+".png")
            if not os.path.exists(folder2):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder2)
            if not os.path.exists(str(folder2)+"/"+str(j).zfill(4)+".npz"):  
                img = np.load(string52+"/"+str(j).zfill(3)+".npz")
                for pi in img:
                    yy=img[pi]
                yy.resize((128, 128,3)) ##重新设定大小
                np.savez(folder2+"/"+str(j).zfill(4)+".npz", pic_pre_optical=yy)
            if not os.path.exists(folder3):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder3)
            if not os.path.exists(str(folder3)+"/"+str(j).zfill(4)+".npz"):  
                im = Image.open(string5+"/"+str(j).zfill(4)+".png")
                region = im.resize((128, 128)) ##重新设定大小
                region = np.array(region)
                img = np.load(string52+"/"+str(j).zfill(3)+".npz")
                for pi in img:
                    yy=img[pi]
                yy.resize((128, 128,3)) ##重新设定大小
                zone = np.zeros([128, 128, 6])
                zone[:,:,0:3]=region
                zone[:,:,3:6]=yy
                np.savez(folder3+"/"+str(j).zfill(4)+".npz", pic_pre_optical=zone)
            if num==1:
                #f.write(folder+"/"+str(j).zfill(4)+".png"+" 1"+"\n")
                #f2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 1"+"\n")
                zonet.write(folder3+"/"+str(j).zfill(4)+".npz"+" 1"+"\n")
            elif num==2:
                #y.write(folder+"/"+str(j).zfill(4)+".png"+" 1"+"\n")
                #y2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 1"+"\n")
                zonev.write(folder3+"/"+str(j).zfill(4)+".npz"+" 1"+"\n")
            #elif num==3:
                #z.write(folder+"/"+str(j).zfill(4)+".png"+" 1"+"\n")
                #z2.write(folder2+"/"+str(j).zfill(4)+".npz"+" 1"+"\n")


# for i in  range(0,40):
#     s = []
#     while (len(s) < 5):
#         a = random.randint(0, 700)
#         if a not in s:
#             s.append(a)
#     for j in s:
#         string="/mnt/disk2/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/UADFV/fake/images/"+str(i).zfill(4)+"_fake/"+str(j).zfill(4)+".png"+" 0"+"\n"
#         f.write(string)



# for i in  range(0,300):
#     # s=[]
#     # while (len(s)<2):
#     #     a= random.randint(0,folder("/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/Deepfakes/c40/images/"+ list1[i])-1)
#     #     if a not in s:
#     #         s.append(a)
#     # k=os.path.getsize("/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/FaceSwap/c40/face_images/"+list1[i]+"/0010"+".png")
#     # if k>153600:
#     #     continue
#     for j in range(0,75):
#         string1="/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/c40/face_images/"+list1[i]#+"/"+str(j).zfill(4)+".png"+" 0"+"\n"
#         string2 = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/c40/face_images/" +list1[i][0:3]#+"/"+str(j).zfill(4)+ ".png" + " 1" + "\n"
        

#         if os.path.exists(string1+"/"+str(j).zfill(4)+".png"):
#             folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/NeuralTextures/c40/face_images128/"+list1[i]
#             if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
#                 os.makedirs(folder)
#             im = Image.open(string1+"/"+str(j).zfill(4)+".png")
#             region = im.resize((128, 128)) ##重新设定大小
#             region.save(folder+"/"+str(j).zfill(4)+".png")
#             y.write(folder+"/"+str(j).zfill(4)+".png"+" 0"+"\n")


#         if j%4==0 and os.path.exists(string2+"/"+str(j).zfill(4)+ ".png"):
#             folder = "/data/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/original_sequences/c40/face_images128/"+list1[i][0:3]
#             if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
#                 os.makedirs(folder)
#             im = Image.open(string2+"/"+str(j).zfill(4)+".png")
#             region = im.resize((128, 128)) ##重新设定大小
#             region.save(folder+"/"+str(j).zfill(4)+".png")
#             y.write(folder+"/"+str(j).zfill(4)+".png"+" 1"+"\n")  
# for i in  range(0,40):
#     s = []
#     while (len(s) < 2):
#         a = random.randint(0, 700)
#         if a not in s:
#             s.append(a)
#     for j in s:
#         string="/mnt/disk2/AssassionXY/Deepfake-Detection/FaceForensics/dataset/manipulated_sequences/UADFV/fake/images/"+str(i).zfill(4)+"_fake/"+str(j).zfill(4)+".png"+" 0"+"\n"
#         y.write(string)

#f.close()
#y.close()
#z.close()
zonev.close()
zonet.close()
