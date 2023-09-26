import os, sys
import argparse
sys.path.insert(0,'/cis/phd/ag3671/arnab/SHIVAGAN')
import lightning as L
import numpy as np
#from einops import rearrange,reduce,repeat
#import wandb
#import torch.nn as nn
#import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

#Custom Modules
from model.archs.shiva_base import ShiVaGAN
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#Pre-processing
from pre_processing.patches_extract import *
from pre_processing.dataset import Custom_Dataset
#Pytorch-Lightning fucntions
from lightning.pytorch.callbacks import ModelSummary,Callback
from lightning.pytorch.loggers import WandbLogger
from utils.functions import generate_psf_array


#PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 4 if torch.cuda.is_available() else 16
NUM_WORKERS = int(os.cpu_count() / 2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str,default ="/cis/phd/ag3671/arnab/datasets/GoPro/zemax_02f/train",
                        help='Local address to Training Dataset')
    parser.add_argument('--val_dataset', type=str,default ="/cis/phd/ag3671/arnab/datasets/GoPro/zemax_02f/test",
                        help='Local address to Training Dataset')
    parser.add_argument('--test_dataset', type=str,default ="/cis/phd/ag3671/arnab/datasets/GoPro/zemax_02f/test",
                        help='Local address to Training Dataset')
    parser.add_argument('--resume', type=str,default = None,
                        help='Resume from Checkpoint:Options: "allow", "must", "never", "auto" or None.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Training Epochs')
    parser.add_argument('--image_shape', type=int, default=256,
                        help='Square Image dimension')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch dimension')
    parser.add_argument('--psf_size', type=int, default=256,
                        help='Support of the PSF function')
    parser.add_argument('--step_size', type=int, default=64,
                        help='Step Size dimension')                    
    parser.add_argument('--num_channels', type=int, default=3,
                        help='number of channels')
    parser.add_argument('--gf_dim', type=int, default=64,
                        help='number of layer')

    args = parser.parse_args()
    

    wblogger = WandbLogger(project="SHIVA_BASIC_DESIGN",
                            log_model="True",
                            resume=args.resume,
                            )
    class LogPredictionsCallback(Callback):
    
        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx):
            """Called when the validation batch ends."""
 
            # `outputs` comes from `LightningModule.validation_step`
            # which corresponds to our model predictions in this case
        
            # Let's log 20 sample image predictions from first batch
            if batch_idx == 0:
                n = 6
                _,x = batch
                gen_images = pl_module(x)
                images = [img for img in gen_images[:n]]
                # Option 1: log images with `WandbLogger.log_image`
                wblogger.log_image(key='sample_images', images=images)
        
    log_predictions_callback = LogPredictionsCallback()
   
    root_test =os.path.join(args.test_dataset,"gt")
    blur_test =os.path.join(args.test_dataset,"blur")
    #Pre-processing and loading Data
    transform = transforms.Compose([    #transforms.Grayscale(1),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5))])
    test_dataset = Custom_Dataset(root_dir= root_test,blur_dir = blur_test,transform=transform)

    #Loaders
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE,num_workers =NUM_WORKERS,shuffle = not True)
    #Model Initialization
    padding = 0
    psf_count = int(np.floor(((args.image_shape +2*padding -(args.patch_size-1)-1)/args.step_size) +1))**2
   
    model =ShiVaGAN(
                   image_shape=args.image_shape,
                   psf_size =  args.psf_size,
                   in_channels  = args.num_channels,
                   psf_count = psf_count,
                   patch_size= args.patch_size,
                   step_size = args.step_size)

    
    model = ShiVaGAN.load_from_checkpoint(checkpoint_path="arnab/SHIVAGAN/train/ZEMAX_02f/cmpq3xlz/checkpoints/epoch=199-step=62600.ckpt")
    trainer = L.Trainer(logger = wblogger,
                        log_every_n_steps=1,
                        callbacks=[ModelSummary(max_depth=-1),
                        log_predictions_callback],
                        devices=[0])

    trainer.test(model = model,dataloaders=test_loader,verbose=True)
if __name__ == "__main__":
    main()