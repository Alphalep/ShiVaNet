"""Incorporates patched Weiner with sca attention and nafnet for patched input
   New upper U net as well which uses NAf blocks to augment training"""
### Novel Architecture which uses grouped convolutions 
###for each patch in the image and generates different filter values for each patch.

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from pytorch_msssim import ssim, ms_ssim,MS_SSIM
from torchmetrics.functional import peak_signal_noise_ratio

#TorchMetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio 
from torchmetrics.image import StructuralSimilarityIndexMeasure


#Patch Wiener
from model.archs.patch_wiener import wiener_deconv,patch_wiener_with_SCA,multi_wiener_with_SCA_and_TransAttention
from model.generator.gen import Generator
from model.generator.unext import ShiVaNext
from model.generator.patch_gen import naf_patchGen
#Custom Modules
from utils.functions import generate_mask, generate_psf_array
from pre_processing.patches_extract import *
#Loss Function definition

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class ShiVaGAN(L.LightningModule):
    def __init__(
        self,
        image_shape:int=256,
        psf_size: int = 256,
        in_channels: int = 3,
        psf_count : int = 64,
        patch_size: int = 32,
        step_size: int = 32,
        K: float = 1,
        C1: float = 0,
        C2: float = 1,
        gf_dim: int = 48,
        lr: float = 8e-4,
        weight_decay:float = 1e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        #**kwargs,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["wiener_model","generator","patched_generator"])
        self.automatic_optimization = False
        self.patch_size = patch_size
        self.step_size = step_size
        # ---------------------------------
        #  PSF and K GENERATION AND PROCESSING
        #----------------------------------
        aberration = [0,4.963,2.637,9.025,7.536,0.157] #zemax0.2f coefficients Seidel
        initial_psfs = torch.from_numpy(generate_psf_array(image_size=image_shape,patch_size=patch_size,
                                                           step_size =step_size,psf_size=psf_size,aberr_coeff=aberration)).type(torch.float32)
        
        h_psf, w_psf = initial_psfs.shape[1:3]
        self.initial_psfs= initial_psfs.unsqueeze(1).expand(-1,in_channels,-1,-1).contiguous().view(-1,h_psf,w_psf)
        #self.masks = nn.Parameter(torch.from_numpy(generate_mask(image_shape,patch_size,step_size,mask_type = "Box")).type(torch.float32).unsqueeze(1).expand(-1,in_channels,-1,-1).contiguous().view(-1,image_shape,image_shape).to("cuda"),requires_grad=False)
        initial_Ks =torch.ones((psf_count*in_channels,1,1),dtype =torch.float32)
        self.psfs = nn.Parameter(self.initial_psfs, requires_grad =True)
        self.Ks = nn.Parameter(initial_Ks, requires_grad =True)

        #-----------------------------------------------
        #          INITIATE MODEL CALL
        #------------------------------------------------
        #self.wiener = wiener_deconv(psfs=self.psfs,Ks=self.Ks,patch_shape=patch_size)
        # self.wiener = patch_wiener_with_SCA(in_channels=in_channels,
        #                                     psfs=self.psfs,
        #                                     Ks=self.Ks,
        #                                     patch_count=psf_count,
        #                                     patch_size=patch_size,
        #                                     step_size=step_size,
        #                                     toggle_patched=True)
        self.wiener = multi_wiener_with_SCA_and_TransAttention(in_channels=in_channels,
                                            psfs=self.psfs,
                                            Ks=self.Ks,
                                            patch_count=psf_count,
                                            num_heads = 1,
                                            toggle_patched=False,
                                            toggle_attention_module=True)
        self.generator  = Generator(in_dim = psf_count*in_channels,out_dim=3,gf_dim=gf_dim)
        self.patched_generator = naf_patchGen(in_dim=in_channels,out_dim=in_channels,psfs = self.psfs,npatches =psf_count,gf_dim =2)
        #self.merge_layer = nn.Conv2d(in_channels=psf_count*in_channels,out_channels=in_channels,kernel_size = 1,padding = 0,stride = 1,groups=1)
        #-----------------------------------------------
        #        INITIAITE LOSS FUNCTION CALL
        #------------------------------------------------
        #loss functions
        self.ms_ssim = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
        self.l1_loss = nn.L1Loss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.psnr = PeakSignalNoiseRatio()
        self.loss_psnr = PSNRLoss()

    def forward(self, z):
        B,C,H,W = z.shape
        #Input Type: 1.Patched 2.Expanded 
        #p_z = extract_patches_2d(z,kernel_size=self.patch_size,padding=0,stride=self.step_size,dilation=1)
        z0  = z.unsqueeze(1).expand(-1,self.hparams.psf_count,-1,-1,-1).contiguous().view(B,-1,H,W)
        #STAGE1
        w = self.wiener(z0)
        z1 = self.patched_generator(w+z0)
        #STAGE2
        #z = combine_patches_2d(p_z,kernel_size=self.patch_size,output_shape=z.shape,padding=0,stride=self.step_size,dilation=1)
        out = self.generator(z1+z0,w+z0)#w+z0 # Modified NAFNet with Downsampling layer and Transpose Attention Added
        #z = self.merge_layer(z)
        #z= torch.sum(z.contiguous().view(B,C,self.hparams.psf_count,H,W),dim=2,keepdim=False)
        return out+z

    
    def custom_loss(self,y_hat,y):
        return (self.hparams.C1*(1-self.ms_ssim(y_hat,y))+self.hparams.C2*self.l1_loss(y_hat,y))
    def psnr_loss(self,y_hat,y):
        return (self.loss_psnr(y_hat,y))

    def training_step(self, batch,batch_idx):
         
        imgs,noisy_imgs = batch

        optimizer_m = self.optimizers()

        # train model

        # generate images
        self.toggle_optimizer(optimizer_m)
        self.generated_imgs = self(noisy_imgs)

        # Custom Loss from Multi-Wiener Net
        #model_loss = self.hparams.K*self.custom_loss(self(noisy_imgs),imgs)
        model_loss =self.psnr_loss(self(noisy_imgs),imgs)

        #Log
        self.log("model_loss", model_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        optimizer_m.zero_grad()
        self.manual_backward(model_loss)
         # clip gradients
        self.clip_gradients(optimizer_m, gradient_clip_val=0.0001, gradient_clip_algorithm="norm")
        optimizer_m.step()
        
        self.untoggle_optimizer(optimizer_m)
        return model_loss
        
       

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = self.hparams.weight_decay
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_m = torch.optim.AdamW(self.parameters(), lr=lr, betas=(b1, b2),weight_decay = wd)
        return [opt_m], []

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

    def validation_step(self,batch,batch_idx):
        imgs,noisy_imgs = batch
        # adversarial loss is binary cross-entropy
        val_loss = self.custom_loss(self(noisy_imgs),imgs)
        psnr_value= self.psnr(self(noisy_imgs),imgs)
        self.log("val_loss",val_loss,on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("psnr_score",psnr_value,on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        lpips_score = self.lpips(torch.clamp(self(noisy_imgs),0,1),torch.clamp(imgs,0,1))
        self.log("lpips_score",lpips_score,on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return val_loss

    def test_step(self,batch,batch_idx):
        imgs,noisy_imgs = batch
        restored_imgs = self(noisy_imgs)
        grid = torchvision.utils.make_grid(restored_imgs) 
        grid_in = torchvision.utils.make_grid(noisy_imgs) 
        grid_gt = torchvision.utils.make_grid(imgs) 
        self.logger.log_image('generated_images', [grid,])
        self.logger.log_image('input_images', [grid_in,])
        self.logger.log_image('gt_images', [grid_gt,])

        #Calculate Mean PSNR value and Mean MS_SSIM value on the whole test dataset
        value_psnr = peak_signal_noise_ratio(restored_imgs,imgs)
        in_psnr  = peak_signal_noise_ratio(noisy_imgs,imgs)
        self.log("PSNR_output",value_psnr,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("PSNR_input",in_psnr,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        value_mssim = ms_ssim(restored_imgs,imgs)
        in_mssim = ms_ssim(noisy_imgs,imgs)
        self.log("MSSIM_OUT",value_mssim,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("MSSIM_IN",in_mssim,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        lpips_score = self.lpips(torch.clamp(self(noisy_imgs),0,1),torch.clamp(imgs,0,1))
        lpips_score_in = self.lpips(torch.clamp(noisy_imgs,0,1),torch.clamp(imgs,0,1))
        self.log("lpips_score_out",lpips_score,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lpips_score_in",lpips_score_in,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return restored_imgs
class lShiVaNext(L.LightningModule):
    def __init__(
        self,
        image_shape = 256,
        psf_size = 256,
        patch_size = 64,
        step_size = 64,
        inp_channels=3, 
        width = 48,
        enc_blk_nums = [4,4,4,8],
        middle_blk_nums = 4,
        heads = [1,2,4,8],
        dec_blk_nums =[8,4,4,4],
        tmax = 85,
        eta_min:float= 1e-7,
        lr: float = 1e-3,
        weight_decay:float = 1e-3,
        b1: float = 0.9,
        b2: float = 0.9,
        ):
        super().__init__()
        self.save_hyperparameters(ignore=["wiener_model","generator","patched_generator"])
        self.automatic_optimization = False
        self.patch_size = patch_size
        self.step_size = step_size
        padding = 0 
        self.psf_count = int(np.floor(((image_shape +2*padding -(patch_size-1)-1)/step_size) +1))**2
        # ---------------------------------
        #  PSF and K GENERATION AND PROCESSING
        #----------------------------------
        aberration = [0,4.963,2.637,9.025,7.536,0.157]
        initial_psfs = torch.from_numpy(generate_psf_array(image_size=image_shape,patch_size=patch_size,
                                                           step_size =step_size,psf_size=psf_size,aberr_coeff=aberration)).type(torch.float32)
        
        h_psf, w_psf = initial_psfs.shape[1:3]
        self.initial_psfs= initial_psfs.unsqueeze(1).expand(-1,inp_channels,-1,-1).contiguous().view(-1,h_psf,w_psf)
        #self.masks = nn.Parameter(torch.from_numpy(generate_mask(image_shape,patch_size,step_size,mask_type = "Box")).type(torch.float32).unsqueeze(1).expand(-1,in_channels,-1,-1).contiguous().view(-1,image_shape,image_shape).to("cuda"),requires_grad=False)
        initial_Ks =torch.ones((self.psf_count*inp_channels,1,1),dtype =torch.float32)
        self.psfs = nn.Parameter(self.initial_psfs, requires_grad =True)
        self.Ks = nn.Parameter(initial_Ks, requires_grad =True)

        #-----------------------------------------------
        #          INITIATE MODEL CALL
        #------------------------------------------------
        self.wiener = multi_wiener_with_SCA_and_TransAttention(in_channels=inp_channels,
                                            psfs=self.psfs,
                                            Ks=self.Ks,
                                            patch_count=self.psf_count,
                                            num_heads = 1,
                                            toggle_patched=False,
                                            toggle_attention_module=True)
        self.generator  = ShiVaNext(img_channel=inp_channels*self.psf_count, width=width, middle_blk_num=middle_blk_nums,heads=heads,bias=False,enc_blk_nums=enc_blk_nums, dec_blk_nums=dec_blk_nums)
                                    
        self.merge_layer = nn.Conv2d(in_channels=self.psf_count*inp_channels,out_channels=inp_channels,kernel_size = 1,padding = 0,stride = 1,groups=1)
        #-----------------------------------------------
        #        INITIAITE LOSS FUNCTION CALL
        #------------------------------------------------
        #loss functions
        self.ssim = StructuralSimilarityIndexMeasure()
        self.l1_loss = nn.L1Loss()
        self.loss_psnr = PSNRLoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.psnr = PeakSignalNoiseRatio()

    # def forward(self, z):
    #     return (self.generator(z)+z)

    # def forward(self, z):   #Original
    #         B,C,H,W = z.shape
    #         z0  = z.unsqueeze(1).expand(-1,self.psf_count,-1,-1,-1).contiguous().view(B,-1,H,W)
    #         w = self.merge_layer(self.wiener(z0)+z0)
    #         out = self.generator(w+z)+z
    #         return out

    def forward(self, z):   #Ablation1 
            B,C,H,W = z.shape
            z0  = z.unsqueeze(1).expand(-1,self.psf_count,-1,-1,-1).contiguous().view(B,-1,H,W)
            out = self.merge_layer(self.generator(self.wiener(z0))) #Refinement Generator 
            return out
    
    def custom_loss(self,y_hat,y):
        return (self.l1_loss(y_hat,y))

    def psnr_loss(self,y_hat,y):
        return (self.loss_psnr(y_hat,y))


    def training_step(self, batch,batch_idx):
         
        imgs,noisy_imgs = batch

        optimizer_m = self.optimizers()

        # train model

        # generate images
        self.toggle_optimizer(optimizer_m)
        
        # Custom Loss from Multi-Wiener Net
        #model_loss =self.custom_loss(self(noisy_imgs),imgs)
        model_loss =self.psnr_loss(self(noisy_imgs),imgs)
       
        self.log("model_loss", model_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        optimizer_m.zero_grad()
        self.manual_backward(model_loss)
         # clip gradients
        self.clip_gradients(optimizer_m, gradient_clip_val=0.01, gradient_clip_algorithm="norm")
        optimizer_m.step()

        self.untoggle_optimizer(optimizer_m)
        return model_loss
        
       

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = self.hparams.weight_decay
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        Tmax = self.hparams.tmax
        eta_min = self.hparams.eta_min
        opt_m = torch.optim.AdamW(self.parameters(), lr=lr, betas=(b1, b2),weight_decay = wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m,Tmax,eta_min)
        return {
            'optimizer': opt_m,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Adjust the scheduling frequency (epoch/batch)
                'monitor': 'val_loss'  # Metric to monitor for scheduling adjustments
            }
        }

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value

    def validation_step(self,batch,batch_idx):
        imgs,noisy_imgs = batch
        # adversarial loss is binary cross-entropy
        val_loss = self.psnr_loss(self(noisy_imgs),imgs)
        psnr_value= self.psnr(self(noisy_imgs),imgs)
        value_ssim = self.ssim(self(noisy_imgs),imgs)
        lpips_score = self.lpips(torch.clamp(self(noisy_imgs),0,1),torch.clamp(imgs,0,1))

        #self.log("val_loss",val_loss,on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("psnr_score",psnr_value,on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lpips_score",lpips_score,on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("SSIM",value_ssim,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return val_loss


    def test_step(self,batch,batch_idx):
        imgs,noisy_imgs = batch
        restored_imgs = self(noisy_imgs)
        grid = torchvision.utils.make_grid(restored_imgs) 
        self.logger.log_image('generated_images', [grid,])

        #Calculate Mean PSNR value and Mean MS_SSIM value on the whole test dataset
        value_psnr = peak_signal_noise_ratio(restored_imgs,imgs)
        in_psnr  = peak_signal_noise_ratio(noisy_imgs,imgs)
        self.log("PSNR_output",value_psnr,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("PSNR_input",in_psnr,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        value_ssim = self.ssim(restored_imgs,imgs)
        in_ssim = self.ssim(noisy_imgs,imgs)
        self.log("SSIM_OUT",value_ssim,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("SSIM_IN",in_ssim,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        lpips_score = self.lpips(torch.clamp(self(noisy_imgs),0,1),torch.clamp(imgs,0,1))
        lpips_score_in = self.lpips(torch.clamp(noisy_imgs,0,1),torch.clamp(imgs,0,1))
        self.log("lpips_score_out",lpips_score,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lpips_score_in",lpips_score_in,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return restored_imgs

