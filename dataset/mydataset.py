"""
Author: Honggu Liu

"""

from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        print (os.getcwd())
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        
        if 'npz' in fn:
            img = np.load(fn)
            for i in img:
                y=img[i]
            if y.shape[2]==6:
                img1=Image.fromarray(np.uint8(y[:,:,0:3])).convert('RGB')
                img2=Image.fromarray(np.uint8(y[:,:,3:6])).convert('RGB')
                if self.transform is not None:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)
                return img1,img2,label
            else:
                img=Image.fromarray(np.uint8(y)).convert('RGB')


        else:
            img = Image.open(fn).convert('RGB')
            i=np.array(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

