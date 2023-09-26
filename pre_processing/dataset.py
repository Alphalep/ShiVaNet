import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets,models,transforms
from torchvision.utils import make_grid
import os
from os import listdir
from PIL import Image
#Forward Model with Seidel Aberrations
#Load the PSF arrays 

class Urban100(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.tensor_transform = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor(),transforms.Grayscale()])

    def __len__(self):
        return(len(listdir(self.root_dir))) 


    def __getitem__(self,idx):
        file_name = os.listdir(self.root_dir)[idx]
        file_path = os.path.join(self.root_dir,file_name)
        image = Image.open(file_path).convert('L')#Removed .convert('L) to turn into color images
        img_hr = self.tensor_transform(image)
        img_tensor  = torch.unsqueeze(img_hr,0)
        if self.transform is not None:
            img_blur = self.transform(img_tensor)
            
        return (img_hr,img_blur,file_name)
   
   
class Custom_Dataset(Dataset):
    def __init__(self,root_dir,blur_dir,transform=None):
        self.root_dir  = root_dir
        self.blur_dir  = blur_dir
        self.transform = transform

    def __len__(self):
        return(len(listdir(self.root_dir))) 


    def __getitem__(self,idx):
        file_name = os.listdir(self.root_dir)[idx]

        file_path_true = os.path.join(self.root_dir,file_name)
        file_path_blur = os.path.join(self.blur_dir,file_name)
        image_true = Image.open(file_path_true) #Removed .convert('L) to turn into color images
        image_blur = Image.open(file_path_blur)#Removed .convert('L) to turn into color images
        if self.transform is not None:
            img_hr = self.transform(image_true)
            img_blur = self.transform(image_blur)
            
        return (img_hr,img_blur)

class gBlur_Dataset(Dataset):
    def __init__(self,root_dir,blur_size = 257,transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.blur = transforms.GaussianBlur(blur_size)

    def __len__(self):
        return(len(listdir(self.root_dir))) 


    def __getitem__(self,idx):
        file_name = os.listdir(self.root_dir)[idx]

        file_path_true = os.path.join(self.root_dir,file_name)
        image_true = Image.open(file_path_true)#Removed .convert('L) to turn into color images
    
        if self.transform is not None:
            img_hr = self.transform(image_true)
            img_blur = self.blur(img_hr)
            
        return (img_hr,img_blur)


