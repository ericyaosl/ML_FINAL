""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, pad=( diffY // 2, diffY - diffY // 2), mode='reflect')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = torch.clamp(x, min=0, max=1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, noise_dim=50, bilinear=False):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        merge_dim = 1 + 1024 // factor
        self.merge = nn.Conv1d(merge_dim, 1024, kernel_size=1)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x, noise):
        x1 = self.inc(x)
        # print('x1: ', x1.size())
        x2 = self.down1(x1)
        # print("x2: ", x2.size())
        x3 = self.down2(x2)
        # print("x3: ", x3.size())
        x4 = self.down3(x3)
        # print("x4: ", x4.size())
        x5 = self.down4(x4)
        # print("x5: ", x5.size())
        noise = noise.view(noise.size(0), 1, noise.size(1))
        # print("noise: ", noise.size())
        merge = torch.cat([x5, noise], dim=1)
        x5 = self.merge(merge)
        # print("x5_: ", x5.size())
        x = self.up1(x5, x4)
        # print("x4_: ", x.size())
        x = self.up2(x, x3)
        # print("x3_: ", x.size())
        x = self.up3(x, x2)
        # print("x2_: ", x.size())
        x = self.up4(x, x1)
        # print("x1_: ", x.size())
        logits = self.outc(x)

        return logits


class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, in_channels, mid_dim, out_channel):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.inc = DoubleConv(in_channels, mid_dim)
        self.down1 = Down(mid_dim, mid_dim*2)
        self.down2 = Down(mid_dim*2, mid_dim*2)
        self.down3 = Down(mid_dim*2, mid_dim*4)
        self.down4 = Down(mid_dim*4, mid_dim*4)
        self.down5 = Down(mid_dim * 4, mid_dim * 4)
        self.down6 = Down(mid_dim * 4, mid_dim * 8)
        self.down7 = Down(mid_dim * 8, mid_dim * 8)
        self.down8 = Down(mid_dim * 8, mid_dim * 16)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.fc = nn.Linear(mid_dim*16, out_channel)

    def forward(self, input):
        """description_method """
        x = self.inc(input)
        # print("x: ", x.size())
        x = self.down1(x)
        # print("x1: ", x.size())
        x = self.down2(x)
        # print("x2: ", x.size())
        x = self.down3(x)
        # print("x3: ", x.size())
        x = self.down4(x)
        # print("x4: ", x.size())
        x = self.down5(x)
        # print("x5: ", x.size())
        x = self.down6(x)
        # print("x6: ", x.size())
        x = self.down7(x)
        # print("x7: ", x.size())
        x = self.down8(x)
        # print("x8: ", x.size())
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print("fc_x: ", x.size())
        x = torch.sigmoid(x)

        return x



if __name__ == '__main__':

    input = torch.randn(64, 1, 800)

    unet = Generator(in_channels=1, out_channels=1)

    noise = torch.randn(64, 50)
    out = unet(input, noise)
    print("out: ", out.size())


    dis = Discriminator(in_channels=1, mid_dim=64, out_channel=1)
    out = dis(input)
    print("dis_out: ", out.size())

