"""Incorporates patched Weiner with sca attention and nafnet for patched input
   New upper U net as well which uses NAf blocks to augment training"""
### Novel Architecture which uses grouped convolutions 
###for each patch in the image and generates different filter values for each patch.

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_msssim import ssim, ms_ssim,MS_SSIM
from torchmetrics.functional import peak_signal_noise_ratio
import numpy as np
#TorchMetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio 

from model.archs.patch_wiener import wiener_deconv,patch_wiener_with_SCA,multi_wiener_with_SCA_and_TransAttention
from model.generator.gen import Generator
from model.discriminator.disc import Discriminator
from model.generator.patch_gen import naf_patchGen
#Custom Modules
from utils.functions import generate_mask, generate_psf_array
from pre_processing.patches_extract import *

class ShiVaGAN(L.LightningModule):
    def __init__(
        self,
        image_shape:int=256,
        psf_size: int = 256,
        in_channels: int = 3,
        psf_count : int = 16,
        patch_size: int = 64,
        step_size: int = 64,
        K: float = 100,
        C1: float = 0,
        C2: float = 1,
        gf_dim: int = 36,
        lr: float = 6e-4,
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
        aberration = [5,0,0,0,0,0]
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
        # self.p_wiener = patch_wiener_with_SCA(in_channels=in_channels,
        #                                     psfs=self.psfs,
        #                                     Ks=self.Ks,
        #                                     patch_count=psf_count,
        #                                     patch_size=patch_size,
        #                                     step_size=step_size)
        
        self.discriminator  = Discriminator(in_dim=3,df_dim=64,apply_sigmoid=False)
        #self.epsilon = nn.Parameter(torch.ones(1,psf_count*in_channels,1,1),requires_grad=True)#psf_count*in_channels->in_channels
        self.wiener = multi_wiener_with_SCA_and_TransAttention(in_channels=in_channels,
                                            psfs=self.psfs,
                                            Ks=self.Ks,
                                            patch_count=psf_count,
                                            num_heads = 1,
                                            toggle_patched=False,
                                            toggle_attention_module=True)
        self.generator  = Generator(in_dim = psf_count*in_channels,out_dim=3,gf_dim=gf_dim)
        self.patched_generator = naf_patchGen(in_dim=in_channels,out_dim=in_channels,psfs = self.psfs,npatches =psf_count,gf_dim =2)
        ##self.merge_layer = nn.Conv2d(in_channels=psf_count*in_channels,out_channels=in_channels,kernel_size = 1,padding = 0,stride = 1,groups=1)
        #-----------------------------------------------
        #        INITIAITE LOSS FUNCTION CALL
        #------------------------------------------------
        #loss functions
        self.ms_ssim = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
        self.l1_loss = nn.L1Loss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.psnr = PeakSignalNoiseRatio()

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
        out = self.generator(z1+z0+w,w+z0) # Modified NAFNet with Downsampling layer and Transpose Attention Added
        #z = self.merge_layer(z)
        #z= torch.sum(z.contiguous().view(B,C,self.hparams.psf_count,H,W),dim=2,keepdim=False)
        return out+z
    
    def custom_loss(self,y_hat,y):
        return (self.hparams.C1*(1-self.ms_ssim(y_hat,y))+self.hparams.C2*self.l1_loss(y_hat,y))

    def cal_gradient_penalty(self,input_data,real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0):
        """Calculate the gradient penalty loss, used in WGAN-GP paper 
        https://arxiv.org/abs/1704.00028

        Arguments:
        input_data(tensor_array)    -- input_images
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

        Returns the gradient penalty loss
        """
        if lambda_gp > 0.0:
            if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
                interpolatesv = real_data
            elif type == 'fake':
                interpolatesv = fake_data
            elif type == 'mixed':
                alpha = torch.rand(real_data.shape[0], 1, device=self.device)
                alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
                interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
            else:
                raise NotImplementedError('{} not implemented'.format(type))
            interpolatesv.requires_grad_(True)
            disc_interpolates = self.discriminator(interpolatesv,input_data)
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
            gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
            gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
            return gradient_penalty, gradients
        else:
            return 0.0, None
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch,batch_idx):
         
        imgs,noisy_imgs = batch

        optimizer_m,optimizer_d= self.optimizers()
        #lambda_gp = 10
        # train model

        # generate images
        self.toggle_optimizer(optimizer_m)
        # self.generated_imgs = self(noisy_imgs)
        """On for Adverserial training<->Remove for Normal"""
        # d_result = self.discriminator(self(noisy_imgs),noisy_imgs)
        # valid = torch.ones_like(d_result)
        # valid = valid.type_as(d_result)
        "---------------------------------------------END"
        # Custom Loss from Multi-Wiener Net
        l1_loss = self.hparams.K*self.custom_loss(self(noisy_imgs),imgs)
        #adv_loss = self.adversarial_loss(d_result,valid)
        g_loss = -torch.mean(self.discriminator(self(noisy_imgs),noisy_imgs))
        model_loss = l1_loss + g_loss
        self.log("model_loss", model_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        optimizer_m.zero_grad()
        self.manual_backward(model_loss)
         # clip gradients
        self.clip_gradients(optimizer_m, gradient_clip_val=0.0001, gradient_clip_algorithm="norm")
        optimizer_m.step()
        
        self.untoggle_optimizer(optimizer_m)

        #Discriminator Step
        #self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        # d_real = self.discriminator(imgs,noisy_imgs)
        # valid = torch.ones_like(d_real)
        # valid = valid.type_as(d_real)

        # real_loss = self.adversarial_loss(d_real, valid)

        # # how well can it label as fake?
        # d_fake = self.discriminator(self(noisy_imgs).detach(),noisy_imgs)
        # fake_class = torch.zeros_like(d_fake)
        # fake_class = fake_class.type_as(d_fake)

        # fake_loss = self.adversarial_loss(d_fake, fake_class)

        # #discriminator loss is the average of these
        # d_loss = (real_loss + fake_loss) / 2
        # self.log("d_loss", d_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # optimizer_d.zero_grad()
        # self.manual_backward(d_loss)
        # optimizer_d.step()
        # self.untoggle_optimizer(optimizer_d)

        self.toggle_optimizer(optimizer_d)

        fake_imgs = self(noisy_imgs)
        #Validation on Real sample pairs
        real_valid = self.discriminator(imgs,noisy_imgs)
        #validity on Fake sample pairs
        fake_valid = self.discriminator(fake_imgs,noisy_imgs)
        #Compute Gradient Penalty
        gradient_penalty = self.cal_gradient_penalty(noisy_imgs.detach(),imgs.detach(),fake_imgs.detach())
        #Adverserial Loss
        d_loss = -torch.mean(real_valid) + torch.mean(fake_valid) + gradient_penalty[0]
        self.log("Discriminator_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        return model_loss
        
       

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = self.hparams.weight_decay
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_m = torch.optim.AdamW(self.parameters(), lr=lr, betas=(b1, b2),weight_decay = wd)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(b1, b2),weight_decay = wd)
        return [opt_m,opt_d], []

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
        self.logger.log_image('generated_images', [grid,])

        #Calculate Mean PSNR value and Mean MS_SSIM value on the whole test dataset
        value_psnr = peak_signal_noise_ratio(restored_imgs,imgs)
        self.log("PSNR_value",value_psnr,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        value_mssim = ms_ssim(restored_imgs,imgs)
        self.log("MSSIM_value",value_mssim,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        lpips_score = self.lpips(torch.clamp(self(noisy_imgs),0,1),torch.clamp(imgs,0,1))
        self.log("lpips_score",lpips_score,on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return restored_imgs