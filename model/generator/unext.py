import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from utils import LayerNorm, GRN
from model.archs.patch_wiener import TransAttention
from timm.models.layers import trunc_normal_, DropPath



class FusionModule(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim,bias,drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Simplified Channel Attention
        self.norm1 = LayerNorm(dim, eps=1e-6,data_format="channels_first")
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        #Projection 
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        input = x
        sca = self.sca(self.norm1(x))
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        x = self.project_out(x*sca)
        return x
class UNextBlock(nn.Module):
    """ Fundamental Module of the U-Net Architecture of ShiVaNext

    Args:
        dim (int): number of channels in the image
        heads: Multi-Headed Attention: No of heads per channel
        bias (bool): bias parameter is also counted in the projection output

    Returns:
       Tensor: Feature descriptor
    """
    def __init__(self,dim,heads,bias):
        super().__init__()
        self.norm1 = LayerNorm(dim,eps=1e-6,data_format="channels_first")
        self.norm2 = LayerNorm(dim,eps=1e-6,data_format="channels_first")
        self.net = FusionModule(dim,bias)
        self.attn = TransAttention(dim,num_heads = heads,bias=bias)
    def forward(self,x):
        x = x + self.net(x)
        x = x + self.attn(self.norm2(x))
        return x

class ShiVaNext(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1,bias = True,heads=[],enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for idx,num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[UNextBlock(chan,heads[idx],bias) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[UNextBlock(chan,heads[3],bias) for _ in range(middle_blk_num)]
            )

        for idx,num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[UNextBlock(chan,heads[-idx-1],bias) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
