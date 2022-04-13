import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
import torch.nn.functional as F
def main():
	args = parse.parse_args()
	test_list = args.test_list
	batch_size = args.batch_size
	model_path = args.model_path
	torch.backends.cudnn.benchmark=True
	test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
	test_dataset_size = len(test_dataset)
	corrects = 0
	acc = 0
	#model = torchvision.models.densenet121(num_classes=2)
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model.eval()
	distacnerecord=[]
	distacnerecord0=[]
	dis_corrects=0
	train_dis_corrects = 0.0
	train_corrects = 0.0
	with torch.no_grad():
		for (image1,image2, labels) in test_loader:
			# image = image.cuda()
			# labels = labels.cuda()
			# outputs = model(image)
			# _, preds = torch.max(outputs.data, 1)
			# corrects += torch.sum(preds == labels.data).to(torch.float32)
			# print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
			image1 = image1.cuda()
			image2 = image2.cuda()
			labels = labels.cuda()
			image=[]
			image.append(image1)
			image.append(image2)
			
			outputs,output1,output2,output3 = model.forward(image)#tensor (32,3,299,299)
			_, preds = torch.max(outputs.data, 1)
			euclidean_distance = F.pairwise_distance(output1, output3)
			for i in range(0,len(labels)):
				if labels[i]==1:
					distacnerecord.append(str(float(euclidean_distance[i]))+'\n')
					if float(euclidean_distance[i])>0.2:#0.37
						dis_corrects+=1
				else:
					distacnerecord0.append(str(float(euclidean_distance[i]))+'\n')
					if float(euclidean_distance[i])<0.2:
						dis_corrects+=1	
			train_dis_corrects += dis_corrects
			iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
			train_corrects += iter_corrects
		epoch_acc = train_corrects / test_dataset_size
		epoch_dis_acc = train_dis_corrects / test_dataset_size
		print('Acc: {:.4f} dis_Acc: {:.4f}'.format(epoch_acc,epoch_dis_acc))
		# acc = corrects / test_dataset_size
		# print('Test Acc: {:.4f}'.format(acc))
def train():
	main()


if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=12)
	parse.add_argument('--test_list', '-tl', type=str, default='/data/AssassionXY/Deepfake-Detection/data_list/train.txt')
	parse.add_argument('--model_path', '-mp', type=str, default='/data/output/fs_xception_F2F/14_fs_F2F.pkl')
	train()
	print('Hello world!!!')