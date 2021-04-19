import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import timm

#1x1 Conv layer
class conv_1x1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size = (1,1))
    
    def forward(self, input):
        return self.conv(input)


# Upsampling Conv block
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


#efficient net based unet
class eff_unet(nn.Module):
    """
    unet with efficient net backbone
    """
    def __init__(self, num_classes):
        super().__init__()
        self.eff_unet = timm.create_model('efficientnet_b0' , pretrained = True)
        self.enc_list = list(self.eff_unet.children())

        self.enc_1 = nn.Sequential(*self.enc_list[:3]) 
        self.enc_2_1 = nn.Sequential(*self.enc_list[3][:3])
        self.enc_2_2 = nn.Sequential(*self.enc_list[3][3:5])
        self.enc_2_3 = nn.Sequential(*self.enc_list[3][5:])
        
        self.conv_1x1 = conv_1x1(320,320)

        self.upconv_2_3 = up_conv(320+320, 128, scale=2)  
        self.upconv_2_2 = up_conv(128+112, 128, scale=2)
        self.upconv_2_1 = up_conv(128+40, 64, scale=4)
        self.upconv_1 = up_conv(64+32, 64, scale=2)

        self.last_conv = nn.Conv2d(64+3, 1, kernel_size=(3,3), padding=1)

        self.adapool = nn.AdaptiveAvgPool2d(output_size=(14, 14))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(14*14, num_classes)


    def forward(self, input):
        out_1 = self.enc_1(input)
        out_2_1 = self.enc_2_1(out_1)
        out_2_2 = self.enc_2_2(out_2_1)
        out_2_3 = self.enc_2_3(out_2_2)

        out_conv_1x1 = self.conv_1x1(out_2_3)
        
        up_2_3 = torch.cat([out_conv_1x1, out_2_3], 1)
        up_conv_2_3 = self.upconv_2_3(up_2_3)
        up_2_2 = torch.cat([up_conv_2_3, out_2_2], 1)
        up_conv_2_2 = self.upconv_2_2(up_2_2)
        up_2_1 = torch.cat([up_conv_2_2, out_2_1], 1)
        up_conv_2_1 = self.upconv_2_1(up_2_1)
        up_1 = torch.cat([up_conv_2_1, out_1], 1)
        up_conv_1 = self.upconv_1(up_1)   
        up_0 = torch.cat([up_conv_1, input], 1)

        pred_map = self.last_conv(up_0)
        pred_map_adapool = self.adapool(pred_map)
        pred_map_flatten = self.flatten(pred_map_adapool)
        pred_class = self.fc(pred_map_flatten)

        return pred_map, pred_class

if __name__ == "__main__":
    tens = torch.randn((1,3,224,224))
    m = eff_unet(2)
    pred = m(tens)

    print(pred)