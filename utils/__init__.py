
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.modules.conv as conv
import numpy as np
from matplotlib import pyplot as plt
import os
import numpy.random as random

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)
            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
            zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out

#Patchify Image batch : [N,C,H,W]-> [N,C,M,H,W] M:No of Patches
class ImagePatchify(nn.Module):
    def __init__(self,patch_size,stride,channels,batch_patching):
        super(ImagePatchify, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.channels = channels
        self.bp = batch_patching
        
    def forward(self,images):
        batch_patches,batch_indices = self.bp.patch_batch(images)
        voxel = torch.empty([len(batch_patches),len(batch_patches[0]),self.patch_size,self.patch_size,self.channels])
        for i,imgs in enumerate(batch_patches):
                voxel[i]= torch.stack(imgs)
        return (voxel.permute(0,4,1,2,3),batch_indices)


class MergePatches(nn.Module):
    def __init__(self,batch_patching):
        super(MergePatches, self).__init__()
        self.bp = batch_patching

    def forward(self,voxel_tensor,batch_indices):
        voxel_tensor = voxel_tensor.permute(0,2,3,4,1)
        voxel_list = voxel_tensor.tolist()
        batch_patches =[]
        for voxel in voxel_list:
            patches=[]
            for patch in voxel:
               patches.append(np.asarray(patch))
            batch_patches.append(patches)  
        merged_batch = self.bp.merge_batch(batch_patches,batch_indices,mode='avg')
        m_imgs = torch.from_numpy(merged_batch).permute(0,3,1,2)
        return m_imgs

def mergePatches(tensor,img_size):
    """
    tensor->3D Tensor in the form (N x C x P x H x W)
    img_size : Original_image_size
    """
    return 0


#---------------------------------------
# SPACE VARIANT BLURRING OPEARTOR
# --------------------------------------
      
class svBlur(object):
    """Class that defines the SvBlur Object.
    """
    def __init__(self,psfs,windows,step_size,device):
        """Function that initializes the svBlur function
        Keyword arguments:
        psfs--Psf filters tensor
        windows-- window tensors
        step_size = integer stride of the window filter
        device = Activate in cuda
        """
        self.psfs = torch.from_numpy(psfs)
        self.step_size =  step_size
        self.windows = torch.from_numpy(windows)
        self.device = device

    def __call__(self,imgs):
        """Function that generates a space variant image convolution.
        Keyword arguments:
        imgs--Tensor of a batch of images ->[N,C,H,W]
        """
        imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        #Convolution: patched_imgs->[N,C,H,W], psfs = [M,1,H,W] 
        output = torch.sum(F.conv2d(imgs.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * #expanded image = psf#
                                    self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device), #expanded windows= psf#
                                    self.psfs.unsqueeze(1).to(self.device),padding ="same",
                                    groups = self.psfs.size(0)),dim=1,keepdim=True)
                                    
        print('Output Image',output.shape)
        return output[:,:,self.step_size:output.size(2)-self.step_size,self.step_size:output.size(3)-self.step_size] 

# def plot_losses(running_train_loss,running_disc_loss, train_epoch_loss, val_epoch_loss, epoch):
#     fig = plt.figure(figsize=(16,16))
#     fig.suptitle('loss trends', fontsize=20)
#     ax1 = fig.add_subplot(221)
#     ax2 = fig.add_subplot(222)
#     ax3 = fig.add_subplot(223)
#     ax4 = fig.add_subplot(224)

#     ax1.title.set_text('epoch train loss VS #epochs')
#     ax1.set_xlabel('#epochs')
#     ax1.set_ylabel('mean train loss per epoch')
#     ax1.plot(train_epoch_loss)
    
#     ax2.title.set_text('epoch val loss VS #epochs')
#     ax2.set_xlabel('#epochs')
#     ax2.set_ylabel('epoch val loss')
#     ax2.plot(val_epoch_loss)
 
#     ax3.title.set_text('Average train loss VS #batches')
#     ax3.set_xlabel('#batches')
#     ax3.set_ylabel('batch train loss')
#     ax3.plot(running_train_loss)

#     ax4.title.set_text('Discriminator loss VS #batches')
#     ax4.set_xlabel('#batches')
#     ax4.set_ylabel('Discriminator Loss')
#     ax4.plot(running_disc_loss)
    
#     #plt.savefig(os.path.join(cfg.losses_dir,'losses_{}.png'.format(str(epoch + 1).zfill(2))))

def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def extract_patches_2ds(x, kernel_size, padding=0, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)

    channels = x.shape[1]

    x = torch.nn.functional.pad(x, padding)
    # (B, C, H, W)
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])
    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1])
    # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
    return x

def extract_patches_2d(x, kernel_size, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out
        
    channels = x.shape[1]
    h_dim_in = x.shape[2]
    w_dim_in = x.shape[3]
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])

    # (B, C, H, W)
    x = torch.nn.functional.unfold(x, kernel_size, padding=padding, stride=stride, dilation=dilation)
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_out * w_dim_out)
    x = x.view(-1, channels, kernel_size[0], kernel_size[1], h_dim_out, w_dim_out)
    # (B, C, kernel_size[0], kernel_size[1], h_dim_out, w_dim_out)
    x = x.permute(0,1,4,5,2,3)
    # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])
    x = x.contiguous().view(-1,channels*h_dim_out*w_dim_out, kernel_size[0], kernel_size[1])
    # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
    return x


def combine_patches_2d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = output_shape[1]
    h_dim_out, w_dim_out = output_shape[2:]
    h_dim_in = get_dim_blocks(h_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_in = get_dim_blocks(w_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])

    # (B * h_dim_in * w_dim_in, C, kernel_size[0], kernel_size[1])
    x = x.view(-1, channels, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    # (B, C, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])
    x = x.permute(0,1,4,5,2,3)
    # (B, C, kernel_size[0], kernel_size[1], h_dim_in, w_dim_in)
    x = x.contiguous().view(-1, channels * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    norm_x = torch.ones_like(x)
    x = torch.nn.functional.fold(x, (h_dim_out, w_dim_out), kernel_size=(kernel_size[0], kernel_size[1]), padding=padding, stride=stride, dilation=dilation)
    norm = torch.nn.functional.fold(norm_x, (h_dim_out, w_dim_out), kernel_size=(kernel_size[0], kernel_size[1]), padding=padding, stride=stride, dilation=dilation)
    # (B, C, H, W)

    return torch.div(x,norm)


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
            
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


if __name__ == "__main__":
    #main()
    #a = torch.arange(1, 65, dtype=torch.float).view(2,2,4,4)
    a = torch.randn([8,3,512,512]).to("cuda")

    print(a.shape)
    #print(a)
    b = extract_patches_2d(a,64, padding=0, stride=64, dilation=1)
    # b = extract_patches_2ds(a, 2, padding=1, stride=2)
    print(b.shape)
    #print(b)
    c = combine_patches_2d(b,kernel_size = 64,output_shape=(8,3,512,512), padding=0, stride=64, dilation=1)
    print(c.shape)
    print(torch.all(a==c))