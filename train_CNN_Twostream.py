import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import dlib
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import torch.nn.functional as F


from network.models import model_selection
from network.mesonet import Meso4, MesoInception4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset


class ContrastiveLoss(torch.nn.Module):
		def __init__(self,margin=0.99):
			super(ContrastiveLoss, self).__init__()
			self.margin = margin
	
		def forward(self, output1, output2, label):
			euclidean_distance = F.pairwise_distance(output1, output2)
			loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)+
	                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
	
			return loss_contrastive

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
def main():
	args = parse.parse_args()
	name = args.name
	continue_train = args.continue_train
	train_list = args.train_list
	val_list = args.val_list
	epoches = args.epoches
	batch_size = args.batch_size
	model_name = args.model_name
	model_path = args.model_path
	output_path = os.path.join('output', name)
	trypath = './output/try'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	torch.backends.cudnn.benchmark=True
	train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
	val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'])
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
	train_dataset_size = len(train_dataset)
	val_dataset_size = len(val_dataset)
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	if continue_train:
		model.load_state_dict(torch.load(model_path))
	model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	criterion2 = ContrastiveLoss()
	criterion3 = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	model = nn.DataParallel(model)
	best_model_wts = model.state_dict()
	best_acc = 0.0
	best_dis_acc = 0.0
	iteration = 0
	for epoch in range(epoches):
		print('Epoch {}/{}'.format(epoch+1, epoches))
		print('-'*10)
		model.train()
		train_loss = 0.0
		train_corrects = 0.0
		train_dis_corrects = 0.0
		val_loss = 0.0
		val_corrects = 0.0
		val_dis_corrects = 0.0
		one_preds = np.array([])
		auc_labels = np.array([])
		distacnerecord=[]
		distacnerecord0=[]

		for (image1,image2, labels) in train_loader:
			iter_loss = 0.0
			iter_corrects = 0.0
			dis_corrects=0


			image1 = image1.cuda()
			image2 = image2.cuda()
			labels = labels.cuda()
			optimizer.zero_grad()
			image=[]
			image.append(image1)
			image.append(image2)
			
			outputs,output1,output2,output3 = model.forward(image)#tensor (32,3,299,299)
			_, preds = torch.max(outputs.data, 1)#output
			######
			one_preds=np.append(one_preds,outputs.data[:,1].cpu().numpy())
			auc_labels= np.append(auc_labels,labels.cpu().numpy())
			

			######
			
			losscontrast = criterion2(output1,output3,labels)
			loss = criterion(outputs, labels)
			loss2 = criterion3(output2,labels)
			(loss+losscontrast+loss2).backward()# 混合损失
			#(loss).backward()#只有对比损失
			optimizer.step()
			euclidean_distance = F.pairwise_distance(output1, output3)
			for i in range(0,len(labels)):
				if labels[i]==1:
					distacnerecord.append(str(float(euclidean_distance[i]))+'\n')
					if float(euclidean_distance[i])>0.37:
						dis_corrects+=1
				else:
					distacnerecord0.append(str(float(euclidean_distance[i]))+'\n')
					if float(euclidean_distance[i])<0.37:
						dis_corrects+=1
				
			
			train_dis_corrects += dis_corrects



			iter_loss = loss.data.item()
			train_loss += iter_loss
			iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
			train_corrects += iter_corrects
			iteration += 1
			if not (iteration % 100):
				print('iteration {} train loss: {:.4f} Acc: {:.4f} dis_Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size,dis_corrects / batch_size))
			
		epoch_loss = train_loss / train_dataset_size
		epoch_acc = train_corrects / train_dataset_size
		epoch_dis_acc = train_dis_corrects / train_dataset_size
		print('epoch train loss: {:.4f} Acc: {:.4f} dis_Acc: {:.4f}'.format(epoch_loss, epoch_acc,epoch_dis_acc))

		########
		
		fpr, tpr, thresholds = roc_curve(auc_labels, one_preds, pos_label=1)
		print("-----train sklearn:",auc(fpr, tpr))
		# roc= auc_calculate(auc_labels, one_preds)
		# print('ROC :{:.4f}'.format(roc))

		########

		distacnerecordv=[]
		distacnerecordv0=[]

		model.eval()
		val_preds = np.array([])
		val_labels = np.array([])
		with torch.no_grad():
			for (image1, image2,labels) in val_loader:
				image1 = image1.cuda()
				labels = labels.cuda()

				image2 = image2.cuda()
				image=[]
				image.append(image1)
				image.append(image2)
				outputs,output1,output2,output3 = model(image)
				_, preds = torch.max(outputs.data, 1)
				######
				val_preds=np.append(one_preds,outputs.data[:,1].cpu().numpy())
				val_labels= np.append(auc_labels,labels.cpu().numpy())
				######
				euclidean_distance = F.pairwise_distance(output1, output3)
				for i in range(0,len(labels)):
					if labels[i]==1:
						distacnerecordv.append(str(float(euclidean_distance[i]))+'\n')
						if float(euclidean_distance[i])>0.6:
							val_dis_corrects+=1
					else:
						distacnerecordv0.append(str(float(euclidean_distance[i]))+'\n')
						if float(euclidean_distance[i])<0.6:
							val_dis_corrects+=1
				
			
				
				
				######
				loss = criterion(outputs, labels)
				val_loss += loss.data.item()
				val_corrects += torch.sum(preds == labels.data).to(torch.float32)
			epoch_loss = val_loss / val_dataset_size
			epoch_acc = val_corrects / val_dataset_size
			epoch_dis_acc = val_dis_corrects / val_dataset_size
			print('epoch val loss: {:.4f} Acc: {:.4f} dis_Acc: {:.4f}'.format(epoch_loss, epoch_acc,epoch_dis_acc))
			#########
			
			fpr, tpr, thresholds = roc_curve(val_labels,val_preds, pos_label=1)
			print("-----val sklearn:",auc(fpr, tpr))
			#########
			if epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()
			if epoch_dis_acc > best_dis_acc:
				best_dis_acc = epoch_dis_acc
		scheduler.step()
		#if not (epoch % 40):
		torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
	print('Best val Acc: {:.4f} dis_Acc: {:.4f} '.format(best_acc,best_dis_acc))
	f=open("k1.txt","w")
	f.writelines(distacnerecord)
	f.close()
	f=open("k0.txt","w")
	f.writelines(distacnerecord0)
	f.close()
	f=open("kval1.txt","w")
	f.writelines(distacnerecordv)
	f.close()
	f=open("kval0.txt","w")
	f.writelines(distacnerecordv0)
	f.close()
	model.load_state_dict(best_model_wts)
	torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))

def cudatensor2array(tensor):
	len=tensor.shape
	print(len[0])
	ay = np.zeros(shape=len)
	for i in range(len[0]):
		ay[i]=tensor[i]
	return ay	



def auc_calculate(labels,preds,n_bins=100):
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if nth_bin==100:
            nth_bin=99
        if labels[i]==1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)


if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--name', '-n', type=str, default='fs_xception_F2F')
	parse.add_argument('--train_list', '-tl' , type=str, default = '/data/AssassionXY/Deepfake-Detection/data_list/train_F2F_6rgbflow3.txt')
	parse.add_argument('--val_list', '-vl' , type=str, default = '/data/AssassionXY/Deepfake-Detection/data_list/val_F2F_Flow_6rgbflow3.txt')
	#print("DF1 30")
	parse.add_argument('--batch_size', '-bz', type=int, default=12)
	parse.add_argument('--epoches', '-e', type=int, default='15')
	parse.add_argument('--model_name', '-mn', type=str, default='fs_F2F.pkl')
	parse.add_argument('--continue_train', type=bool, default=False)
	parse.add_argument('--model_path', '-mp', type=str, default='./Deepfake-Detection/output/1_df_c0_299.pkl')
	main()