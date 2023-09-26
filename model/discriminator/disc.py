import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


def cnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0, first_layer = False):

   if first_layer:
       return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
   else:
       return nn.Sequential(
           nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )

def tcnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0, first_layer = False):
   if first_layer:
       return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding)

   else:
       return nn.Sequential(
           nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )

class Discriminator(nn.Module):
    def __init__(self,in_dim,df_dim,apply_sigmoid =True,):#input : 256x256
        super(Discriminator,self).__init__()
        self.conv1 = cnn_block(in_dim*2,df_dim,4,2,1, first_layer=True) # 128x128
        self.conv2 = cnn_block(df_dim,df_dim*2,4,2,1)# 64x64
        self.conv3 = cnn_block(df_dim*2,df_dim*4,4,2,1)# 32 x 32
        self.conv4 = cnn_block(df_dim*4,df_dim*8,4,1,1)# 31 x 31
        self.conv5 = cnn_block(df_dim*8,1,4,1,1, first_layer=True)# 30 x 30
       
        self.sigmoid = nn.Sigmoid() if apply_sigmoid==True else nn.Identity()
        
        
    def forward(self, x, y):
        O = torch.cat([x,y],dim=1)
        O = F.leaky_relu(self.conv1(O),0.2)
        O = F.leaky_relu(self.conv2(O),0.2)
        O = F.leaky_relu(self.conv3(O),0.2)
        O = F.leaky_relu(self.conv4(O),0.2)
        O = self.conv5(O)
        return self.sigmoid(O)