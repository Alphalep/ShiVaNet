import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from utils import LayerNorm2d
from einops import rearrange,reduce,repeat

class patch_wiener_with_SCA(nn.Module):
    """Performs patch simple channel attention by breaking the image into non-
    overlapping patches and then calculating simple channel attention as shown in NAFNet

    Args:
        in_channels (int): _description_
        initial_psfs(torch.tensor): 
        initial_Ks(torch.tensor):
        patch_count(int)

    Returns:
        _type_: _description_
    """
    def __init__(self,in_channels,psfs,Ks,patch_count,toggle_patched = True,patch_size=64,step_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.step_size = step_size
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=patch_count*in_channels, out_channels=patch_count*in_channels, kernel_size=1, padding=0, stride=1,
                      groups=patch_count, bias=True))
        self.lnorm = LayerNorm2d(in_channels*patch_count)
        self.wiener_deconv =  wiener_deconv(psfs=psfs,Ks=Ks,patch_shape=patch_size,toggle_patched=True)
        self.alpha = nn.Parameter(torch.zeros(1,patch_count*in_channels,1,1),requires_grad=True)
    def forward(self,inp):
        patched_inp = extract_patches_2d(inp,
                                        kernel_size=self.patch_size,
                                        padding=0,
                                        stride=self.step_size,
                                        dilation=1)
        x = self.lnorm(patched_inp)
        wiener_out = self.wiener_deconv(patched_inp)
        x =x*self.sca(x)
        return wiener_out+x*self.alpha


            
class wiener_deconv(nn.Module):
    """
    Performs Wiener Deconvolutions on fixed patches in the frequency domain for each psf
    Input : initial_PSF's are of Shape (C,Y,X)
            initial_K has shape (C,1,1) for each psf.
    """

    def __init__(self, psfs,Ks,patch_shape =64,toggle_patched=False):
        super(wiener_deconv, self).__init__()
        self.psfs = psfs
        self.Ks = Ks
        self.transform = transforms.CenterCrop(patch_shape) if toggle_patched==True else nn.Identity()
    def forward(self,y):
        #Preprocessing image tensor to be of same same shape as psf 
        h,w = y.shape[2:4]
        #PSF 
        h_psf, w_psf = self.psfs.shape[1:3]
        h_pad = h_psf - h
        w_pad = w_psf - w
        padding  = (int(np.ceil(h_pad / 2)), int(np.floor(h_pad / 2)),
                    int(np.ceil(w_pad / 2)), int(np.floor(w_pad / 2))
                    )
        y = F.pad(y,padding,"constant",0).type(torch.complex64)
        # Temporarily transpose y since we cannot specify axes for fft2d
        Y = torch.fft.fft2(y) 
        H = torch.fft.fft2(self.psfs)
        
        X=(torch.conj(H)*Y)/ (torch.square(torch.abs(H))+ 100*torch.exp(self.Ks))# Added a RElU function to force only positive values
        x=torch.abs((torch.fft.ifftshift(torch.fft.ifft2(X), dim=(-2, -1))))
        return self.transform(x)

class ensemble(nn.Module):
    def __init__(self, wiener_model, unet_model):
        super(ensemble, self).__init__()
        self.wiener_model = wiener_model
        self.unet_model = unet_model
    def forward(self, x):
        wiener_output = self.wiener_model(x)
        final_output = self.unet_model(wiener_output)
        return final_output

class multi_wiener_with_SCA_and_TransAttention(nn.Module):
    """Performs patch simple channel attention by breaking the image into non-
    overlapping patches and then calculating simple channel attention as shown in NAFNet

    Args:
        in_channels (int): _description_
        initial_psfs(torch.tensor): 
        initial_Ks(torch.tensor):
        patch_count(int)

    Returns:
        _type_: _description_
    """
    def __init__(self,in_channels,psfs,Ks,patch_count,num_heads,toggle_patched = False,toggle_attention_module=False):
        super().__init__()
        self.patch_count = patch_count
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=patch_count*in_channels, out_channels=patch_count*in_channels, kernel_size=1, padding=0, stride=1,
                      groups=patch_count, bias=True))
        self.lnorm1 = LayerNorm2d(in_channels*patch_count)
        self.wiener_deconv =  wiener_deconv(psfs=psfs,Ks=Ks,toggle_patched=toggle_patched)
        self.alpha = nn.Parameter(torch.zeros(1,patch_count*in_channels,1,1),requires_grad=True)
        #self.beta = nn.Parameter(torch.zeros(1,patch_count*in_channels,1,1),requires_grad=True)
        if toggle_attention_module:
            self.attn = nn.Sequential( LayerNorm2d(in_channels*patch_count),
                                      TransAttention(patch_count*in_channels,num_heads=num_heads,bias=False))
        else:
            self.attn = nn.Identity()
    def forward(self,inp):
        x = self.lnorm1(inp)
        x = self.wiener_deconv(x)
        x = x*self.sca(x)
        x = x+inp*self.alpha
        x = x+self.attn(x)
        return x

#----------------------------------------
#   TRANS-CHANNEL ATTENTION
#-----------------------------------------
class TransAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TransAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
