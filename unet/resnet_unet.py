import torch
from torch import nn
from torchvision import models
from unet_parts import *

class Unet(nn.Module):
    """
    Simple-Unet architecture with resnet34 encoder
    """
    def __init__(self, n_out):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.encoder_layers = list(self.resnet.children())[:-2]
        self.enc_b1 = nn.Sequential(*self.encoder_layers[:4])  #>>torch.Size([1, 64, 56, 56])
        self.enc_b2 = nn.Sequential(*self.encoder_layers[4:6]) #>>torch.Size([1, 128, 28, 28])
        self.enc_b3 = nn.Sequential(*self.encoder_layers[6:8]) #>>torch.Size([1, 512, 7, 7])
        
        self.conv_1x1 = conv_1x1(512,512) #>>torch.Size([1, 512, 7, 7])

        self.upconv_1 = up_conv(512,128, scale=4)  #>>torch.Size([1, 128, 28, 28])
        self.upconv_2 = up_conv(128+128, 128, scale=2)#>>torch.Size([1, 128, 56, 56])
        self.upconv_3 = up_conv(128+64, 64, scale=4)

        self.last_conv = nn.Conv2d(64+3, 3, kernel_size=(3,3), padding=1)

    def forward(self, input):
        out1 = self.enc_b1(input)
        out2 = self.enc_b2(out1)
        out3 = self.enc_b3(out2)
        out4 = self.conv_1x1(out3)
        out5 = self.upconv_1(out4)

        out6 = torch.cat([out5,out2], 1)
        out7 = self.upconv_2(out6)

        out8 = torch.cat([out7, out1], 1)
        out9 = self.upconv_3(out8)

        out10 = torch.cat([input, out9], 1)
        out10 = self.last_conv(out10)
        return out10


if __name__ == "__main__":
    input = torch.randn((1,3,224,224))
    unet = Unet(3)
    out = unet(input)
    print(out.size())




