import torch
from torch import nn
import torch.nn.functional as F

class conv_1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size = (1,1))
    
    def forward(self, input):
        return self.conv(input)

class up_conv(nn.Module):
    def __init__(self, in_c, out_c, scale = 4):
        super().__init__()
        if scale == 4:
            self.upconv_bilinear = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_c, out_c, kernel_size=(3,3), padding=1),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_c, out_c, kernel_size=(3,3), padding=1)
            )
            
        else:
            self.upconv_bilinear = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_c, out_c, kernel_size=(3,3), padding=1)
            )

    def forward(self, input): return self.upconv_bilinear(input)
        
        


