import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import LayerNorm2d,SimpleGate
class Projection2d(nn.Module):
    def __init__(self,dim,w_size,in_dim = 3,h_dim =6,npatches =16):
        super().__init__()
        self.projection =  nn.Sequential(LayerNorm2d(in_dim*npatches),
                                        nn.Conv2d(in_channels=in_dim*npatches,out_channels =h_dim*npatches,
                                                kernel_size=4,stride=2,padding=1,groups=npatches),
                                        nn.AvgPool2d(w_size),
                                        LayerNorm2d(h_dim*npatches),
                                        nn.Conv2d(in_channels=h_dim*npatches,out_channels =dim,
                                                kernel_size=4,stride=2,padding=1,groups=npatches)
                                        )
    def  forward(self,x):
        return self.projection(x)

def first_cnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0,groups=1,first_layer = False):
   if first_layer:
       return nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,groups=groups)
   else:
       return nn.Sequential(
           nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,groups=groups),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )

def first_tcnn_block(in_channels,out_channels,kernel_size,stride=1,padding=0,output_padding=0,groups=1,first_layer = False):
   if first_layer:
       return nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding,groups=groups)

   else:
       return nn.Sequential(
           nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,output_padding=output_padding),
           nn.BatchNorm2d(out_channels,momentum=0.1,eps=1e-5),
           )
class cnn_block(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=0,groups=1,DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = in_channels * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=padding, stride=stride, groups=dw_channel,#groups->changed from dw_chn to groups*dw_chn
                               bias=True)  
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)#groups->changed from 1 to groups
        #Downsampling layer
        self.convd = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride,groups=groups)
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

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1,out_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1,out_channels, 1, 1)), requires_grad=True)

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
        
        return  y + x * self.gamma
class tcnn_block(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,padding=0,groups=1,DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = in_channels * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)#Same
        self.conv2 = nn.ConvTranspose2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=padding, stride=stride, groups=dw_channel,#Same
                               bias=True)  
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)#Same
        #Upsampling layer
        self.convu = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride,groups=groups,bias=False)
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

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1,out_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1,out_channels, 1, 1)), requires_grad=True)

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
        
        return  y + x * self.gamma


class naf_patchGen(nn.Module):
    def __init__(self,in_dim,out_dim,psfs,npatches =16,gf_dim =2):#input : 64x64
        super(naf_patchGen,self).__init__()
        
        self.e1 = cnn_block(in_dim*npatches,gf_dim*npatches,4,2,1,groups=npatches) #32x32
        self.e2 = cnn_block(gf_dim*npatches,gf_dim*npatches*2,4,2,1,groups=npatches)#16x16
        self.e3 = cnn_block(gf_dim*2*npatches,gf_dim*npatches*4,4,2,1,groups=npatches)#8x8
        self.e4 = cnn_block(gf_dim*4*npatches,gf_dim*npatches*8,4,2,1,groups=npatches)#4x4
        self.e5 = cnn_block(gf_dim*8*npatches,gf_dim*npatches*8,4,2,1,groups=npatches)#2x2
        self.e6 = cnn_block(gf_dim*8*npatches,gf_dim*npatches*8,4,2,1,groups=npatches)#1x1

        self.d3 = tcnn_block(gf_dim*8*npatches,gf_dim*8*npatches,4,2,1,groups=npatches)       
        self.d4 = tcnn_block(gf_dim*8*2*npatches,gf_dim*8*npatches,4,2,1,groups=npatches)      
        self.d5 = tcnn_block(gf_dim*8*2*npatches,gf_dim*4*npatches,4,2,1,groups=npatches)        
        self.d6 = tcnn_block(gf_dim*4*2*npatches,gf_dim*2*npatches,4,2,1,groups=npatches)
        self.d7 = tcnn_block(gf_dim*2*2*npatches,gf_dim*1*npatches,4,2,1,groups=npatches)
        self.d8 = tcnn_block(gf_dim*1*2*npatches,out_dim*npatches,4,2,1,groups=npatches)#64

    def forward(self,x):
        B,C,H,W = x.shape
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d3 = torch.cat([self.d3(e6),e5],1)
        d4 = torch.cat([self.d4(d3),e4],1)
        d5 = torch.cat([self.d5(d4),e3],1)
        d6 = torch.cat([self.d6(d5),e2],1)
        d7 = torch.cat([self.d7(d6),e1],1)
        d8 = self.d8(d7)

        return d8[:,:,:H,:W]

class p2p_Discriminator(nn.Module):
 def __init__(self,in_dim,df_dim,instance_norm=False):#input : 256x256
   super(p2p_Discriminator,self).__init__()
   self.conv1 = cnn_block(in_dim*2,df_dim,4,2,1, first_layer=True) # 128x128
   self.conv2 = cnn_block(df_dim,df_dim*2,4,2,1)# 64x64
   self.conv3 = cnn_block(df_dim*2,df_dim*4,4,2,1)# 32 x 32
   self.conv4 = cnn_block(df_dim*4,df_dim*8,4,1,1)# 31 x 31
   self.conv5 = cnn_block(df_dim*8,1,4,1,1, first_layer=True)# 30 x 30

   self.sigmoid = nn.Sigmoid()
 def forward(self, x, y):
   O = torch.cat([x,y],dim=1)
   O = F.leaky_relu(self.conv1(O),0.2)
   O = F.leaky_relu(self.conv2(O),0.2)
   O = F.leaky_relu(self.conv3(O),0.2)
   O = F.leaky_relu(self.conv4(O),0.2)
   O = self.conv5(O)

   return self.sigmoid(O)

"""6/5 Inspiration from NAFNET"""
"""Model: from 'Simple Baselines for Image Restoration' """
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class naf_patchGen_32(nn.Module):
    def __init__(self,in_dim,out_dim,npatches =16,gf_dim =2,instance_norm=False):#input : 32x32
        super(naf_patchGen_32,self).__init__()
        #self.expand_dim =  cnn_block(in_dim*npatches,gf_dim*npatches,4,2,1,groups=npatches)
        #self.down_layers = nn.ModuleList([cnn_block(in_dim*npatches*(2**i),
                                        #gf_dim*npatches*(2**(i+1)),4,2,1,groups=npatches) for i in range(6)])

        self.e1 = cnn_block(in_dim*npatches,gf_dim*npatches,4,2,1,groups=npatches//4) #16x16
        self.e2 = cnn_block(gf_dim*npatches,gf_dim*npatches*2,4,2,1,groups=npatches//4)#8x8
        self.e3 = cnn_block(gf_dim*2*npatches,gf_dim*npatches*4,4,2,1,groups=npatches//4)#4x4
        self.e4 = cnn_block(gf_dim*4*npatches,gf_dim*npatches*8,4,2,1,groups=npatches//4)#2x2
        self.e5 = cnn_block(gf_dim*8*npatches,gf_dim*npatches*8,4,2,1,groups=npatches//4)#1x1
        

        self.d4 = tcnn_block(gf_dim*8*npatches,gf_dim*8*npatches,4,2,1,groups=npatches//4)#2x2
        self.d5 = tcnn_block(gf_dim*8*2*npatches,gf_dim*4*npatches,4,2,1,groups=npatches//4)#4x4
        self.d6 = tcnn_block(gf_dim*4*2*npatches,gf_dim*2*npatches,4,2,1,groups=npatches//4)#8x8
        self.d7 = tcnn_block(gf_dim*2*2*npatches,gf_dim*1*npatches,4,2,1,groups=npatches//4)#16x16
        self.d8 = tcnn_block(gf_dim*1*2*npatches,out_dim*npatches,4,2,1,groups=npatches//4)#32x32

    def forward(self,x):
        B,C,H,W = x.shape
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        d4 = torch.cat([self.d4(e5),e4],1)
        d5 = torch.cat([self.d5(d4),e3],1)
        d6 = torch.cat([self.d6(d5),e2],1)
        d7 = torch.cat([self.d7(d6),e1],1)
        d8 = self.d8(d7)

        return d8[:,:,:H,:W]






class patch_NAFNet(nn.Module):

    def __init__(self,in_channels,n_blocks=1):
        super().__init__()
            
        self.model = nn.Sequential(*[NAFBlock(in_channels) for _ in range(n_blocks)])
        
            
    def forward(self,inp):
        return self.model(inp)