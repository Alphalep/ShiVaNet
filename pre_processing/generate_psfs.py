import torch 
import torch.nn.functional as F
from empatches import EMPatches
import numpy as np
from scipy.signal import fftconvolve,convolve2d
import cv2
from numpy.fft import fftshift,ifftshift,ifft2
from helper import bartlett2d,generate_psf,seidel,circ
import matplotlib.pyplot as plt

#---------------------------------------
# Generate PSf function definition
#---------------------------------------
def generate_psf(u0,v0,coeffs,Fu,Fv,lam,fnum):
    #wavefront 
    W = seidel(u0,v0,-2*lam*fnum*Fu,-2*lam*fnum*Fv,coeffs)
    #coherent tarnsfer function
    H = circ(np.sqrt(Fu**2+Fv**2)*2*lam*fnum)*np.exp(-1j*k*W)
    #PSF
    psf = np.abs(ifftshift(ifft2(H)))**2
    psf = psf/np.sum(psf)
    return np.flip(psf,-1)

"""Image Plane"""
M = 256#                         # Sample No    :Patch Size Value
L = 10**-3                      # Image Plane side Length
du = L/M                        #Sample Interval
u = np.arange(-L/2,L/2,du)      # u-coordinates
v = u    # v - coordinates
"""Constants for Image Patchification"""
step_size = int(M/2)
overlap = 0.5
image_size =512

img = np.ones((512,512))
print(img.shape)
Ig = img/np.max(img)
Ig = np.pad(Ig,step_size)
"""Exit pupil plane"""
lam = 0.55 *10**-6              # Wavelength
k = 2*np.pi/lam                 #Wavenumber
Dxp = 20*10**-3                 #Exit pupil size
wxp = Dxp/2
zxp = 100*10**-3                #Exit pupil distance
fnum = zxp/(2*wxp)              #Exit-pupil f number

twof0 = 1/(lam*fnum)            # cutoff frequency
fN = 1/(2*du)                   #Nyquist frequency


"""Aberration Coefficients"""
W_d = 0*lam  
W_sa = 10.5*lam
W_coma= 10* lam
W_astig = 10.5*lam 
W_field = 10*lam
W_distort  = 10*lam 
coeffs = [W_d,W_sa,W_coma,W_astig,W_field,W_distort]

fu = np.arange(-1/(2*du),1/(2*du),1/L) # Image freq coordinates
fu = fftshift(fu)                     # Avoiding shift in loop
Fu,Fv = np.meshgrid(fu,fu)
print(Fu.shape)
#Assert that patch_size is divisible by the image size
emp = EMPatches()
img_patches, indices = emp.extract_patches(Ig, patchsize=M, overlap=overlap)
final = np.zeros_like(Ig)
psf_arr = [] 
M_diameter = Ig.shape[0]
for idx in range(len(indices)):
    p,q,r,s = indices[idx]
    a,b,c,d = (np.array(indices[idx])-M_diameter/2)/(M_diameter/2)
    u_psf = (a+b)/2
    v_psf = (c+d)/2
    #print("u_psf,v_psf",u_psf,v_psf)
    psf_patch = generate_psf(v_psf,u_psf,coeffs,Fu,Fv,lam,fnum)
    #print(a,b,c,d)
    input = np.zeros_like(Ig)
    input[p:q,r:s] =  bartlett2d(M)*Ig[p:q,r:s]
    #general observation of fftconvolution of image patch with its corresponding psf
    out = cv2.filter2D(input,-1,psf_patch, borderType=cv2.BORDER_REFLECT)
    #out = convolve2d(input,psf_patch,mode="same",boundary="fill")
    final = final+out
    #Dump the psfs for future use
    psf_arr.append(psf_patch)
psf_stack = np.stack(psf_arr,axis=2)
psf_stack = psf_stack.transpose(2,0,1)
print(psf_stack.shape)
window_arr=[]
for n in range(len(psf_arr)):  
    windows = np.zeros((Ig.shape[0],Ig.shape[1]),float)
    ind = indices[n]
    windows[ind[0]:ind[1],ind[2]:ind[3]] = bartlett2d(M)
    window_arr.append(windows) 
windows = np.array(window_arr)
print(windows.shape)
np.savez("/cis/phd/ag3671/arnab/COPSV-GAN/pre_processing/psf_info_new",psf_arr = psf_arr,windows = windows)
