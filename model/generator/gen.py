import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import LayerNorm2d
from model.archs.patch_wiener import TransAttention

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class cnn_block(nn.Module):
    def __init__(self, in_channels,out_channels,num_heads,kernel_size,stride=1,padding=0,DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = in_channels * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=padding, stride=stride, groups=dw_channel,
                               bias=True)  
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #Downsampling layer
        self.convd = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride,bias=True)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * out_channels
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(out_channels)
        self.norm3 = LayerNorm2d(out_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.ones((1,out_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1,out_channels, 1, 1)), requires_grad=True)
        #Transpose Attention
        self.attn  = TransAttention(out_channels,num_heads=num_heads,bias = False)

    def forward(self, inp):
        x = inp #cin
        x_branch = self.convd(x)# cout
        x = self.norm1(x) #cin

        x = self.conv1(x) #cin->dw
        x = self.conv2(x) #dw->dw
        x = self.sg(x) # dw->dw/2
        x = x * self.sca(x)#dw/2->dw/2
        x = self.conv3(x)#dw/2->out_channels

        x = self.dropout1(x) #cout

        y = x_branch+ x * self.beta #cout

        x = self.conv4(self.norm2(y))#cout->ffn
        x = self.sg(x)#ffn->ffn/2
        x = self.conv5(x)#ffn/2->cout

        x = self.dropout2(x)#cout
        y = y + x * self.gamma
        y = y+self.attn(self.norm3(y))
        return  y
class tcnn_block(nn.Module):
    def __init__(self, in_channels,out_channels,num_heads,kernel_size,stride=1,padding=0,DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = in_channels * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=padding, stride=stride, groups=dw_channel,
                               bias=True)  
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #Upsampling layer
        self.convu = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride,bias=True)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * out_channels
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(out_channels)
        self.norm3 = LayerNorm2d(out_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1,out_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1,out_channels, 1, 1)), requires_grad=True)
         #Transpose Attention
        self.attn  = TransAttention(out_channels,num_heads=num_heads,bias = False)

    def forward(self, inp):
        x = inp
        x_branch = self.convu(inp)
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = x_branch+ x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = y+self.attn(self.norm3(y))
        return y

"""SHIVAGAN : Our Model"
Generator used has been adapted from NAFNet Paper with downsampling layer added to 
cnn_block as well as t_cnn_block"""
class Generator(nn.Module):
 
 def __init__(self,in_dim,out_dim,num_heads=[1,2,4,8],gf_dim = 64):#input : 256x256
   super(Generator,self).__init__()
   self.e1 = cnn_block(in_dim,gf_dim,num_heads[0],4,2,1,)
   self.e2 = cnn_block(gf_dim,gf_dim*2,num_heads[0],4,2,1,)
   self.e3 = cnn_block(gf_dim*2,gf_dim*4,num_heads[1],4,2,1,)
   self.e4 = cnn_block(gf_dim*4,gf_dim*8,num_heads[1],4,2,1,)
   self.e5 = cnn_block(gf_dim*8,gf_dim*8,num_heads[2],4,2,1,)
   self.e6 = cnn_block(gf_dim*8,gf_dim*8,num_heads[2],4,2,1,)
   self.e7 = cnn_block(gf_dim*8,gf_dim*8,num_heads[3],4,2,1,)
   self.e8 = cnn_block(gf_dim*8,gf_dim*8,num_heads[3],4,2,1)

   self.d1 = tcnn_block(gf_dim*8,gf_dim*8,num_heads[3],4,2,1,)
   self.d2 = tcnn_block(gf_dim*8*2,gf_dim*8,num_heads[3],4,2,1,)
   self.d3 = tcnn_block(gf_dim*8*2,gf_dim*8,num_heads[2],4,2,1,)
   self.d4 = tcnn_block(gf_dim*8*2,gf_dim*8,num_heads[2],4,2,1)
   self.d5 = tcnn_block(gf_dim*8*2,gf_dim*4,num_heads[1],4,2,1)
   self.d6 = tcnn_block(gf_dim*4*2,gf_dim*2,num_heads[1],4,2,1)
   self.d7 = tcnn_block(gf_dim*2*2,gf_dim*1,num_heads[0],4,2,1)
   self.d8 = tcnn_block(gf_dim*1*2,out_dim,num_heads[0],4,2,1,)#256x256
   #self.end = nn.Conv2d(out_dim,out_dim,1,1,0)
 # Input Modulated Bias Term 
   self.bias1 = Biaser(in_dim,gf_dim,out_size=128)
   #self.bias2 = Biaser(in_dim,gf_dim*2,out_size=64)
   self.bias3 = Biaser(in_dim,gf_dim*4,out_size=32)
   #self.bias4 = Biaser(in_dim,gf_dim*8,out_size=16)
   self.bias5 = Biaser(in_dim,gf_dim*8,out_size=8)
   #self.bias6 = Biaser(in_dim,gf_dim*8,out_size=4)
   self.bias7 = Biaser(in_dim,gf_dim*8,out_size=2)
   #self.bias8 = Biaser(in_dim,gf_dim*8,out_size=1)

   



 def forward(self,x,w):
    B,C,H,W = x.shape
    e1 = self.e1(x) +self.bias1(w)
    e2 = self.e2(e1)#+self.bias2(w)
    e3 = self.e3(e2)+self.bias3(w)
    e4 = self.e4(e3)#+self.bias4(w)
    e5 = self.e5(e4)+self.bias5(w)
    e6 = self.e6(e5)#+self.bias6(w)
    e7 = self.e7(e6)+self.bias7(w)
    e8 = self.e8(e7)#+self.bias8(w)

    d1 = torch.cat([self.d1((e8)),e7],1)
    d2 = torch.cat([self.d2(d1),e6],1)
    d3 = torch.cat([self.d3(d2),e5],1)
    d4 = torch.cat([self.d4(d3),e4],1) 
    d5 = torch.cat([self.d5(d4),e3],1)
    d6 = torch.cat([self.d6(d5),e2],1)
    d7 = torch.cat([self.d7(d6),e1],1)
    d8 = self.d8(d7)
    return d8[:,:,:H,:W]

class Biaser(nn.Module):
    def __init__(self,in_channels,out_channels,out_size,with_attn = False,with_merge = True):
        super().__init__()
        hidden_dim = in_channels if with_merge==False else 3
        self.norm = LayerNorm2d(in_channels)if with_attn==True else nn.Identity()
        self.attn = TransAttention(in_channels,1,bias=False) if with_attn==True else nn.Identity()
        self.merge = nn.Conv2d(in_channels= in_channels,out_channels=hidden_dim,kernel_size = 1,padding = 0,stride = 1,groups=1) if with_merge==True else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(out_size)
        
        self.conv = nn.Conv2d(in_channels=hidden_dim,out_channels = out_channels,kernel_size =3,padding = 1,stride = 1)
        expand_dim = 2 * out_channels
        self.conv_1  = nn.Conv2d(in_channels= out_channels,out_channels=expand_dim,kernel_size = 1,padding = 0,stride = 1)
        self.sg = SimpleGate()
        self.conv_2  = nn.Conv2d(in_channels= expand_dim//2,out_channels=out_channels,kernel_size = 1,padding = 0,stride = 1)
        
        

    def forward(self,x):
        #Dimension Reduce
        x = self.merge(self.pool(self.attn(self.norm(x))))
        x = self.conv(x)
        x = self.conv_1(x)
        x = self.sg(x)
        x = self.conv_2(x)
        return x

