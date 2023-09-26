import torch.nn.functional as F
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import os
from os import listdir
from PIL import Image
from numpy.fft import fft2,fftshift,ifft2,ifftshift 
import matplotlib.pyplot as plt
from empatches import EMPatches

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Space Vraiant Blur 
class svBlur(object):
    """Class that defines the SvBlur Object.
    """
    def __init__(self,psfs,windows,image_size,step_size,device):
        """Function that initializes the svBlur function
        Keyword arguments:
        psfs--Psf filters tensor: Tensor of batch of Convolution kernels [M,1,H,W]
        windows-- window tensors
        step_size = integer stride of the window filter
        device = Activate in cuda
        """
        self.psfs = torch.from_numpy(psfs).unsqueeze(1)
        self.step_size =  step_size
        self.windows = torch.from_numpy(windows)
        self.device = device
        self.transform = transforms.CenterCrop(image_size-2*step_size)

    def __call__(self,imgs):
        """Function that generates a space variant image convolution.
        Keyword arguments:
        imgs--Tensor of a batch of images ->[N,C,H,W]
        """
        #imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        batch_size = imgs.size(0)
        chn_nos = imgs.size(1)
        psf_nos = self.psfs.size(0)
        pad = (int(np.ceil(self.psfs.size(2)/ 2)), int(np.floor(self.psfs.size(2) / 2)),
                    int(np.ceil(self.psfs.size(2) / 2)), int(np.floor(self.psfs.size(2) / 2))
                    )
        in_channels = chn_nos*psf_nos
        groups = in_channels
        img_tensor = imgs.unsqueeze(2).expand(-1,-1,psf_nos,-1,-1).to(self.device)
        mask_tensor = self.windows.unsqueeze(0).unsqueeze(0).expand(batch_size,chn_nos,-1,-1,-1).to(self.device)
        product = img_tensor * mask_tensor
        input = product.view(batch_size,in_channels,product.size(3),product.size(4))

        #Convolution: patched_imgs->[N,C,H,W], psfs = [M,1,H,W] 
        output = F.conv2d(input.to(self.device),self.psfs.expand(chn_nos,-1,-1,-1,-1).reshape(in_channels,1,self.psfs.size(2),-1).to(self.device),padding = "same",groups = groups)
        output = torch.sum(output.reshape(-1,imgs.size(1),self.psfs.size(0),output.size(3),output.size(2)),dim = 2,keepdim=False)                            
        print('Output Image',output.shape)
        return self.transform(output)
class svBlur_new(object):
    """Class that defines the SvBlur Object.
    """
    def __init__(self,psfs,windows,image_size,step_size,device):
        """Function that initializes the svBlur function
        Keyword arguments:
        psfs--Psf filters tensor: Tensor of batch of Convolution kernels [M,1,H,W]
        windows-- window tensors
        step_size = integer stride of the window filter
        device = Activate in cuda
        """
        self.psfs = torch.from_numpy(psfs).unsqueeze(1).type(torch.float32)
        self.step_size =  step_size
        self.windows  = torch.from_numpy(windows).type(torch.float32)
        self.device = device
        self.transform = transforms.CenterCrop(image_size-2*step_size)

    def __call__(self,imgs):
        """Function that generates a space variant image convolution.
        Keyword arguments:
        imgs--Tensor of a batch of images ->[N,C,H,W]
        """
        #imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        batch_size = imgs.size(0)
        chn_nos = imgs.size(1)
        psf_nos = self.psfs.size(0)
        pad = (int(np.ceil(self.psfs.size(2)/ 2)), int(np.floor(self.psfs.size(2) / 2)),
                    int(np.ceil(self.psfs.size(2) / 2)), int(np.floor(self.psfs.size(2) / 2))
                    )
        in_channels = chn_nos*psf_nos
        groups = in_channels
        img_tensor = imgs.unsqueeze(2).expand(-1,-1,psf_nos,-1,-1).to(self.device).type(torch.float32)
        mask_tensor = self.windows.unsqueeze(0).unsqueeze(0).expand(batch_size,chn_nos,-1,-1,-1).to(self.device)
        product = img_tensor * mask_tensor
        norm_product = torch.ones_like(img_tensor)*mask_tensor
        input = product.view(batch_size,in_channels,product.size(3),product.size(4))
        norm_input = norm_product.view(batch_size,in_channels,norm_product.size(3),norm_product.size(4))
        #Convolution: patched_imgs->[N,C,H,W], psfs = [M,1,H,W] 
        output = F.conv2d(input.to(self.device),self.psfs.expand(chn_nos,-1,-1,-1,-1).reshape(in_channels,1,self.psfs.size(2),-1).to(self.device),padding = "same",groups = groups)
        output = torch.sum(output.reshape(-1,imgs.size(1),self.psfs.size(0),output.size(3),output.size(2)),dim = 2,keepdim=False)
        norm_out = torch.sum(norm_input.reshape(-1,imgs.size(1),self.psfs.size(0),output.size(3),output.size(2)),dim = 2,keepdim=False) 
        out = torch.div(output,norm_out)
        print('Output Image',out.shape)
        return self.transform(out)

def star_field(img_size,step_size,radius):
    img = np.zeros(img_size)
    for i in range(step_size,img.shape[0],step_size):
        for j in range(step_size,img.shape[1],step_size):
            cv2.circle(img,center =(i,j),radius = radius,color = 255,thickness=-1)
    return img

def get_img_strip(tensr):
    # shape: [bs,1,h,w]
    bs, _ , h, w = tensr.shape
    tensr2np = (tensr.cpu().detach().numpy().clip(0,1)*255).astype(np.uint8)    
    canvas = np.ones((h, w*bs), dtype = np.uint8)
    for i in range(tensr.shape[0]):
        patch_to_paste = tensr2np[i, 0, :, :]
        canvas[:, i*w: (i+1)*w] = patch_to_paste
    return canvas
#Seidel Aberration Function
def seidel (p0,q0,x,y,coeffs):
    beta = np.arctan2(q0,p0)
    h2 = np.sqrt(p0**2 + q0**2)
    #rotation of grid
    xr = x*np.cos(beta)+y*np.sin(beta)
    yr = -x*np.sin(beta) + y*np.cos(beta)

    #Seidel Aberration function

    rho2 = xr**2 + yr**2

    W = coeffs[0]*rho2 + coeffs[1]*rho2**2 + coeffs[2]*h2*rho2*xr + coeffs[3]*h2**2*xr**2 + coeffs[4]*h2**2*rho2 + coeffs[5]*h2**3*xr
    return W

def circular(N,dim = [512,512],center=[256,256]):
    x = np.linspace(1,dim[0],dim[0])
    y = x
    X = x-center[0]
    Y = y-center[1]
    P,Q = np.meshgrid(X/N,Y/N)
    out = P**2+Q**2
    out = out<=1
    return out.astype(float)
def circ(X):
    out = X<=1
    return out.astype(float)
def generate_psf(u0,v0,coeffs,Fu,Fv,lam,fnum,k):
    #wavefront 
    W = seidel(u0,v0,-2*lam*fnum*Fu,-2*lam*fnum*Fv,coeffs)
    #coherent tarnsfer function
    H = circ(np.sqrt(Fu**2+Fv**2)*2*lam*fnum)*np.exp(-1j*k*W)
    #PSF
    psf = np.abs(ifftshift(ifft2(H)))**2
    psf = psf/np.sum(psf)
    return np.flip(psf,-1)

def bartlett2d(N):
     return(np.outer(np.bartlett(N),np.bartlett(N)))

def generate_psf_array(image_size,patch_size,step_size,psf_size,aberr_coeff):
    """Image Plane"""
    Ig = np.zeros((image_size,image_size))
    M = psf_size#                         # Sample No    :Patch Size Value
    L = 10**-3                      # Image Plane side Length
    du = L/M                        #Sample Interval
    u = np.arange(-L/2,L/2,du)      # u-coordinates
    v = u    # v - coordinates
    """Exit pupil plane"""
    lam = 0.55 *10**-6              # Wavelength
    k = 2*np.pi/lam                 #Wavenumber
    Dxp = 20*10**-3                 #Exit pupil size
    wxp = Dxp/2
    zxp = 100*10**-3                #Exit pupil distance
    fnum = zxp/(2*wxp)              #Exit-pupil f number

    twof0 = 1/(lam*fnum)            # cutoff frequency
    fN = 1/(2*du)                   #Nyquist frequency
    fu = np.arange(-1/(2*du),1/(2*du),1/L) # Image freq coordinates
    fu = fftshift(fu)                     # Avoiding shift in loop
    Fu,Fv = np.meshgrid(fu,fu)

    """Aberration Coefficients"""
    W_d = aberr_coeff[0]*lam  
    W_sa = aberr_coeff[1]*lam
    W_coma=aberr_coeff[2]* lam
    W_astig = aberr_coeff[3]*lam 
    W_field = aberr_coeff[4]*lam
    W_distort  = aberr_coeff[5]*lam 
    coeffs = [W_d,W_sa,W_coma,W_astig,W_field,W_distort]
    "Generates PSF Array for different areas"
    emp = EMPatches()
    #overlap = np.round(1 - step_size/patch_size,decimals=1)
    overlap = 1 - step_size/patch_size

    img_patches, indices = emp.extract_patches(Ig, patchsize=patch_size, overlap=overlap)
    psf_arr = [] 
    M_diameter = Ig.shape[0]
    for idx in tqdm(range(len(indices))):
        #PSF_ARRAYS
        a,b,c,d = (np.array(indices[idx])-M_diameter/2)/(M_diameter/2)
        u_psf = (a+b)/2
        v_psf = (c+d)/2
        #print("u_psf,v_psf",u_psf,v_psf)
        psf_arr.append(np.flip(generate_psf(v_psf,u_psf,coeffs,Fu,Fv,lam,fnum,k),-1))
        
    return np.array(psf_arr)

def generate_mask(img_shape,kernel_size,step_size,mask_type = "Bartlett"):
    emp = EMPatches()
    #overlap = np.round(1 - step_size/patch_size,decimals=1)
    overlap = 1 - step_size/kernel_size
    #print("Overlap of Masks",overlap)
    img_patches, indices = emp.extract_patches(np.zeros((img_shape,img_shape)), patchsize=kernel_size, overlap=overlap)
    #print("Indices",indices)
    window_stack =[]
    assert (kernel_size%step_size==0)
    for idx in tqdm(range(len(indices))):
        mask = np.zeros((img_shape,img_shape))
        ind = indices[idx]
        if mask_type == "box":
            mask[ind[0]:ind[1],ind[2]:ind[3]] = 1
        elif mask_type == "Bartlett":
            mask[ind[0]:ind[1],ind[2]:ind[3]] = bartlett2d(kernel_size)
        window_stack.append(mask)
    window_stack = np.array(window_stack)
    return window_stack