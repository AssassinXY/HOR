"""

Author: Andreas Rössler
"""
import os
import argparse


import torch
#import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from network.xception import xception, xception_concat, xception_lstm,xception_twostream
# from network.classifiers import Meso4
import math
import torchvision


def return_pytorch04_xception(pretrained=False):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            '/data/model/xception-b5690688.pth')
        list1={}
        for name, weights in state_dict.items():
            if 'block' in name:
                if 'block1.' in name:
                    name2=list(name)
                    name2.insert(5,'2')
                    name2= ''.join(name2)
                    list1[name2] =state_dict[name]
                else:
                    name2=list(name)
                    name2.insert(6,'2')
                    name2= ''.join(name2)
                    list1[name2] =state_dict[name]
            elif 'conv' in name:
                name2=list(name)
                name2.insert(5,'2')
                name2= ''.join(name2)
                list1[name2] =state_dict[name]
            elif 'bn' in name:
                name2=list(name)
                name2.insert(3,'2')
                name2= ''.join(name2)
                list1[name2] =state_dict[name]
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
                
        for i in list1:
            if 'pointwise' in i:
                list1[i] = list1[i].unsqueeze(-1).unsqueeze(-1)
            state_dict[i]=list1[i]
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc



    modelLS = xception_twostream()

    pretrained_dict = model.state_dict()
    model_dict = modelLS.state_dict()
    # 将pretrained_dict里不属于modelBX的键剔除掉
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    modelLS.load_state_dict(model_dict)
    return modelLS
    


    return model

def return_pytorch04_xception_lstm(pretrained=False):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            '/data/model/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    modelLS = xception_lstm()

    pretrained_dict = model.state_dict()
    model_dict = modelLS.state_dict()
    # 将pretrained_dict里不属于modelBX的键剔除掉
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    modelLS.load_state_dict(model_dict)
    return modelLS

class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.5):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception(pretrained=True)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!如果不载入是False载入是True
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            num_ftrs2 = self.model.last_linear2.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
                self.model.last_linear2 = nn.Linear(num_ftrs2, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes) #real linear
                )
                self.model.last_linear2 = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs2, num_out_classes) #real linear
                )
        elif modelchoice == 'xception_lstm':
            self.model = return_pytorch04_xception_lstm(pretrained=True)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!如果不载入是False载入是True
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            num_ftrs2 = self.model.last_linear2.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
                self.model.last_linear2 = nn.Linear(num_ftrs2, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes) #real linear
                )
                self.model.last_linear2 = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs2, num_out_classes) #real linear
                )
        elif modelchoice == 'xception_concat':
            self.model = xception_concat()
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes)
    #    , 299, \True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes)
    #    , \224, True, ['image'], None
    elif modelname == 'xception_concat':
        return TransferModel(modelchoice='xception_concat',
                             num_out_classes=num_out_classes)
    elif modelname == 'resnet50':
        return TransferModel(modelchoice='resnet50', dropout=dropout,
                             num_out_classes=num_out_classes)
    elif modelname == 'xception_lstm':
        return TransferModel(modelchoice='xception_lstm', dropout=dropout,
                             num_out_classes=num_out_classes)                         
    else:
        raise NotImplementedError(modelname)


if __name__ == '__main__':
    model, image_size, *_ = model_selection('resnet18', num_out_classes=2)
    print(model)
    model = model.cuda()
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s))
