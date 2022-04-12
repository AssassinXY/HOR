"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from .bam import *

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()
        #self.bam = BAM(out_filters)
        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            #rep.append(self.bam)
            rep.append(nn.MaxPool2d(3,strides,1))

        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception_LSTM(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_LSTM, self).__init__()
        self.num_classes = num_classes

        #self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        #do bam here 
        # self.bam1 = BAM(32)
        # self.bam2 = BAM(64)
        # self.bam3 = BAM(128)


        #-------------------LSTM----------------------
        self.rnn = nn.LSTM(input_size=2048, hidden_size=20, num_layers=2,bidirectional=True)#(input_size,hidden_size,num_layers)
        self.input = torch.randn(5, 64, 10)#(seq_len, batch, input_size)#(32,2048,10,10)
        self.h0 = torch.randn(4, 32, 20).cuda() #(num_layers,batch,output_size)
        self.c0 = torch.randn(4, 32, 20).cuda() #(num_layers,batch,output_size)
        

        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)########last_linner second parameter is num_classes####################2048---->>>>>>>100

        self.fc2 = nn.Linear(40, num_classes)
        
        self.softmax = nn.Softmax()

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input) #(32,3,299, 299)
        x = self.bn1(x)
        x = self.relu(x)

        #######
        #x=self.bam1(x)#(32,32,149,149)
        #print("BAM is using")
        #x=self.bam2(x)
        #x=self.bam3(x)
        #######

        x = self.conv2(x) #(32,64, 147, 147)
        x = self.bn2(x)#(32,64, 147, 147)
        x = self.relu(x)#(32,64, 147, 147)
        x = self.block1(x)#(32,128, 74, 74)
        x = self.block2(x)#(32,256, 147, 147)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x) #(1024, 299, 299)

        x = self.conv3(x) #(1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)#（1536，10，10）
        
        x = self.conv4(x) #(2048, 299, 299)
        x = self.bn4(x)#(32,2048,10,10)
        y = x
        x = x.permute(2,3,0,1).clone()
        x=x.view(-1,32,2048).clone()#(10.10.32.2048)->(100.32.2048)

        x, (hn, cn) = self.rnn(x, (self.h0, self.c0))#(100.32.2048)->(100.32.40)
        x = x.permute(1,2,0).clone()#->(32,40,100)
        x=x.view(32,40,10,10).clone()#->(32.40.10.10)
        return x#（100，32，40）

    def logits(self, features):#[32,2048,10,10]
        x = self.relu(features)
        # x=self.bam4(x)#[32,2]
        # print("using bam in front of pool")
        x = F.adaptive_avg_pool2d(x, (1, 1)) #  ->(32.40.1.1)
        x = x.view(x.size(0), -1)#->(32.40)
        x = self.last_linear2(x)#->(32.2)
        x= self.softmax(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        #self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        #do bam here 
        # self.bam1 = BAM(32)
        # self.bam2 = BAM(64)
        # self.bam3 = BAM(128)



        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)########last_linner second parameter is num_classes
        self.softmax = nn.Softmax()




        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input) #(32, 299, 299)
        x = self.bn1(x)
        x = self.relu(x)

        #######
        #x=self.bam1(x)#(32,32,149,149)
        #print("BAM is using")
        #x=self.bam2(x)
        #x=self.bam3(x)
        #######

        x = self.conv2(x) #(32,64, 147, 147)
        x = self.bn2(x)#(32,64, 147, 147)
        x = self.relu(x)#(32,64, 147, 147)
        x = self.block1(x)#(32,128, 74, 74)
        x = self.block2(x)#(32,256, 147, 147)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x) #(1024, 299, 299)

        x = self.conv3(x) #(1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)#（1536，10，10）
        
        x = self.conv4(x) #(2048, 10,10)
        x = self.bn4(x)#(32,2048,10,10)


        

        return x

    def logits(self, features):#[32,2048,10,10]
        x = self.relu(features)
        # x=self.bam4(x)#[32,2]
        # print("using bam in front of pool")
        x = F.adaptive_avg_pool2d(x, (1, 1)) #[32,2048,10,10]-->[32,2048,1,1]
        x = x.view(x.size(0), -1)#1front    #[32,2048,1,1]-->[32,2048]
        x = self.last_linear(x)#[32,2048]-->[32,2]
        x= self.softmax(x)
        return x#[32,2]

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class Xception_Twostream(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_Twostream, self).__init__()
        self.num_classes = num_classes

        #self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        #do bam here 
        self.bam1 = BAM(32)
        self.bam12 = BAM(32)
        self.bam2 = BAM(64)
        self.bam22 = BAM(64)
        #-------------------LSTM----------------------
        self.rnn = nn.LSTM(input_size=2048, hidden_size=20, num_layers=2,bidirectional=True)#(input_size,hidden_size,num_layers)
        self.input = torch.randn(5, 24, 10)#(seq_len, batch, input_size)#(32,2048,10,10)
        self.h0 = torch.randn(4, 12, 20).cuda() #(num_layers,batch,output_size)
        self.c0 = torch.randn(4, 12, 20).cuda() #(num_layers,batch,output_size)

        self.fc2 = nn.Linear(40, num_classes)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)########last_linner second parameter is num_classes
        self.softmax = nn.Softmax()

        ####################################two-stream##################################
        self.conv12 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn12 = nn.BatchNorm2d(32)
        

        self.conv22 = nn.Conv2d(32,64,3,bias=False)
        self.bn22 = nn.BatchNorm2d(64)

        #do relu here

        self.block21=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block22=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block32=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block42=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block52=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block62=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block72=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block82=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block92=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block120=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block121=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block122=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv32 = SeparableConv2d(1024,1536,3,1,1)
        self.bn32 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv42 = SeparableConv2d(1536,2048,3,1,1)
        self.bn42 = nn.BatchNorm2d(2048)




        ####################################two-stream##################################


        

    def features(self, input1,input2):
        
        x = self.conv1(input1) #(32, 299, 299)
        #x=self.bam1(x)
        x = self.bn1(x)
        x = self.relu(x)

        #######
        x=self.bam1(x)#(32,32,149,149)
        #print("BAM is using")
        #x=self.bam2(x)
        #x=self.bam3(x)
        #######

        x = self.conv2(x) #(32,64, 147, 147)
        #x = self.bam2(x)
        x = self.bn2(x)#(32,64, 147, 147)
        x = self.relu(x)#(32,64, 147, 147)
        #x = self.bam2(x)
        x = self.block1(x)#(32,128, 74, 74)   64->128
        x = self.block2(x)#(32,256, 147, 147)  128->256
        x = self.block3(x)#256->728
        x = self.block4(x)#728->728
        x = self.block5(x)#728->728
        x = self.block6(x)#728->728
        x = self.block7(x)#728->728
        x = self.block8(x)#728->728
        x = self.block9(x)#728->728
        x = self.block10(x)#728->728
        x = self.block11(x)#728->728
        x = self.block12(x) #(1024, 299, 299) 728->1024

        x = self.conv3(x) #(1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)#（1536，10，10）
        #x = self.bam2(x)
        x = self.conv4(x) #(2048, 299, 299)
        cx=x
        x = self.bn4(x)


        y = self.conv12(input2) #(32, 299, 299)
        #y=self.bam12(y)#0.8204
        y = self.bn12(y)
        y = self.relu(y)
        #____________
        y=self.bam12(y)

        #_____________
        y = self.conv22(y) #(32,64, 147, 147)
        #y = self.bam22(y)#acc 0.82 dis acc0.80
        y = self.bn22(y)#(32,64, 147, 147)
        y = self.relu(y)#(32,64, 147, 147)
        #y = self.bam22(y)#没效果
        y = self.block21(y)#(32,128, 74, 74)
        y = self.block22(y)#(32,256, 147, 147)
        y = self.block32(y)
        y = self.block42(y)
        y = self.block52(y)
        y = self.block62(y)
        y = self.block72(y)
        y = self.block82(y)
        y = self.block92(y)
        y = self.block120(y)
        y = self.block121(y)
        y = self.block122(y) #(1024, 299, 299)

        y = self.conv32(y) #(1536, 299, 299)
        y = self.bn32(y)
        y = self.relu(y)#（1536，10，10）
        #y = self.bam22(y)

        y = self.conv42(y)
        cy=y
        y = self.bn42(y)
        y = y.permute(2,3,0,1).clone()
        y=y.view(-1,12,2048).clone()#(10.10.32.2048)->(100.32.2048) 第2个是bc

        y, (hn,cn) = self.rnn(y, (self.h0,self.c0))#(100.32.2048)->(100.32.40)
        y = y.permute(1,2,0).clone()#->(32,40,100)
        y=y.view(12,40,10,10).clone()#->(32.40.10.10)第一个是bc
        





        
        


        

        return x,y,cx,cy

    def logits(self, features1,features2):#[32,2048,10,10]
        x = self.relu(features1)
        y = self.relu(features2)
        # x=self.bam4(x)#[32,2]
        # print("using bam in front of pool")

        # y = F.adaptive_avg_pool2d(y, (1, 1)) 
        # y = y.view(y.size(0), -1)#1front
        # y = self.last_linear(y)
        # oy=y
        # y = self.softmax(y)

        #___________________
        y = F.adaptive_avg_pool2d(y, (1, 1)) #  ->(32.40.1.1)
        y = y.view(y.size(0), -1)#->(32.40)
        y = self.last_linear2(y)#->(32.2)
        oy=y
        y= self.softmax(y)
        #___________________




        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.size(0), -1)#1front
        x = self.last_linear(x)
        ox=x
        x= self.softmax(x)
        return x,ox,y,oy

    def forward(self, input):
        input1 = input[0]
        input2 = input[1]
        x,y,cx,cy = self.features(input1,input2)
        x,ox,oy,y = self.logits(x,y)
        return x,ox,y,oy

class Xception_Twostream_basic(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_Twostream_basic, self).__init__()
        self.num_classes = num_classes

        #self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        #do bam here 
        # self.bam1 = BAM(32)
        # self.bam2 = BAM(64)
        # self.bam3 = BAM(128)
        


        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)########last_linner second parameter is num_classes
        self.softmax = nn.Softmax()

        ####################################two-stream##################################
        self.conv12 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn12 = nn.BatchNorm2d(32)
        

        self.conv22 = nn.Conv2d(32,64,3,bias=False)
        self.bn22 = nn.BatchNorm2d(64)

        #do relu here

        self.block21=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block22=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block32=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block42=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block52=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block62=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block72=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block82=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block92=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block120=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block121=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block122=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv32 = SeparableConv2d(1024,1536,3,1,1)
        self.bn32 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv42 = SeparableConv2d(1536,2048,3,1,1)
        self.bn42 = nn.BatchNorm2d(2048)




        ####################################two-stream##################################


        

    def features(self, input1,input2):
        x = self.conv1(input1) #(32, 299, 299)
        x = self.bn1(x)
        x = self.relu(x)

        #######
        #x=self.bam1(x)#(32,32,149,149)
        #print("BAM is using")
        #x=self.bam2(x)
        #x=self.bam3(x)
        #######

        x = self.conv2(x) #(32,64, 147, 147)
        x = self.bn2(x)#(32,64, 147, 147)
        x = self.relu(x)#(32,64, 147, 147)
        x = self.block1(x)#(32,128, 74, 74)   64->128
        x = self.block2(x)#(32,256, 147, 147)  128->256
        x = self.block3(x)#256->728
        x = self.block4(x)#728->728
        x = self.block5(x)#728->728
        x = self.block6(x)#728->728
        x = self.block7(x)#728->728
        x = self.block8(x)#728->728
        x = self.block9(x)#728->728
        x = self.block10(x)#728->728
        x = self.block11(x)#728->728
        x = self.block12(x) #(1024, 299, 299) 728->1024

        x = self.conv3(x) #(1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)#（1536，10，10）
        x = self.conv4(x) #(2048, 299, 299)
        cx=x
        x = self.bn4(x)


        y = self.conv12(input2) #(32, 299, 299)
        y = self.bn12(y)
        y = self.relu(y)
        y = self.conv22(y) #(32,64, 147, 147)
        y = self.bn22(y)#(32,64, 147, 147)
        y = self.relu(y)#(32,64, 147, 147)
        y = self.block21(y)#(32,128, 74, 74)
        y = self.block22(y)#(32,256, 147, 147)
        y = self.block32(y)
        y = self.block42(y)
        y = self.block52(y)
        y = self.block62(y)
        y = self.block72(y)
        y = self.block82(y)
        y = self.block92(y)
        y = self.block120(y)
        y = self.block121(y)
        y = self.block122(y) #(1024, 299, 299)

        y = self.conv32(y) #(1536, 299, 299)
        y = self.bn32(y)
        y = self.relu(y)#（1536，10，10）

        y = self.conv42(y)
        cy=y
        y = self.bn42(y)






        
        


        

        return x,y,cx,cy

    def logits(self, features1,features2):#[32,2048,10,10]
        x = self.relu(features1)
        y = self.relu(features2)
        # x=self.bam4(x)#[32,2]
        # print("using bam in front of pool")

        y = F.adaptive_avg_pool2d(y, (1, 1)) 
        y = y.view(y.size(0), -1)#1front
        y = self.last_linear(y)
        oy=y
        y = self.softmax(y)




        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.size(0), -1)#1front
        x = self.last_linear(x)
        ox=x
        x= self.softmax(x)
        return x,ox,y,oy

    def forward(self, input):
        input1 = input[0]
        input2 = input[1]
        x,y,cx,cy = self.features(input1,input2)
        x,ox,oy,y = self.logits(x,y)
        return x,ox,y,oy


class Xception_concat(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_concat, self).__init__()
        self.num_classes = num_classes

        #self.conv1 = nn.Conv2d(15,32,3,2,0,bias=False)
        self.conv1 = nn.Conv2d(3,32,3,2,0,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        #do bam here 
        self.bam1 = BAM(32)
        # self.bam2 = BAM(64)
        # self.bam3 = BAM(128)



        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)########last_linner second parameter is num_classes
        self.softmax = nn.Softmax()

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input) #(32, 299, 299)
        x = self.bn1(x)
        x = self.relu(x)

        #######
        x=self.bam1(x)#(32,32,149,149)
        #print("BAM is using")
        #x=self.bam2(x)
        #x=self.bam3(x)
        #######

        x = self.conv2(x) #(32,64, 147, 147)
        x = self.bn2(x)#(32,64, 147, 147)
        x = self.relu(x)#(32,64, 147, 147)
        x = self.block1(x)#(32,128, 74, 74)
        x = self.block2(x)#(32,256, 147, 147)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x) #(1024, 299, 299)

        x = self.conv3(x) #(1536, 299, 299)
        x = self.bn3(x)
        x = self.relu(x)#（1536，10，10）
        
        x = self.conv4(x) #(2048, 299, 299)
        x = self.bn4(x)
        return x

    def logits(self, features):#[32,2048,10,10]
        x = self.relu(features)
        # x=self.bam4(x)#[32,2]
        # print("using bam in front of pool")
        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.size(0), -1)#1front
        x = self.last_linear(x)
        x= self.softmax(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception_Twostream_basic(num_classes=num_classes)
    
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model

def xception_concat(num_classes=1000):
    model = Xception_concat(num_classes=num_classes)
    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model

def xception_lstm(num_classes=1000):
    model = Xception_LSTM(num_classes=num_classes)
    # TODO: ugly
    model.last_linear = model.fc
    model.last_linear2= model.fc2
    del model.fc
    del model.fc2
    return model

def xception_twostream(num_classes=1000):
    model = Xception_Twostream(num_classes=num_classes)
    # TODO: ugly
    model.last_linear = model.fc
    model.last_linear2= model.fc2
    del model.fc
    del model.fc2
    return model
