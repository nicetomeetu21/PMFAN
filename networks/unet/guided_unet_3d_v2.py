""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedUNet(nn.Module):
    def __init__(self, local_size=(80, 320, 80), is_ret_global=False):
        super(GuidedUNet, self).__init__()
        self.global_unet = UNet1(is_out_features=True)
        # [64 128 256 512]
        self.local_unet = UNet2(is_in_features=True)
        self.local_size = local_size
        self.is_ret_global = is_ret_global

    def get_center_features(self, features, local_pos):
        pos_z, pos_x, pos_y = local_pos
        size_z, size_x, size_y = self.local_size
        center_features = []
        for feature in features:
            pos_z //= 2
            pos_x //= 2
            pos_y //= 2
            size_z //= 2
            size_x //= 2
            size_y //= 2
            center = feature[:, :, pos_z:pos_z + size_z, pos_x:pos_x + size_x, pos_y:pos_y + size_y]
            center_features.append(center)

            # print('local_pos local_size in guided unet',local_pos, self.local_size)

            # print('center_feature: ', z,x,y, pos_z, pos_x, pos_y, center.shape)
        return center_features

    def forward(self, global_img, local_img, local_pos):
        out1, global_features = self.global_unet(global_img)
        center_features = self.get_center_features(global_features, local_pos)
        out2 = self.local_unet(local_img, center_features)
        if self.is_ret_global:
            return out2, out1
        else:
            return out2

    def encode(self, global_img, local_img, local_pos):
        out1, global_features = self.global_unet(global_img)
        center_features = self.get_center_features(global_features, local_pos)
        feats = self.local_unet.encode(local_img, center_features)
        if self.is_ret_global:
            return feats, out1
        else:
            return feats

    def decode(self, feats):
        out2 = self.local_unet.decode(feats)
        return out2

class UNet1(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_out_features=False):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down_old(32, 64)
        self.down2 = Down_old(64, 128)
        self.down3 = Down_old(128, 256)
        self.up1 = Up(256, 384, 128, bilinear)
        self.up2 = Up(128, 192, 64, bilinear)
        self.up3 = Up(64, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.is_out_features = is_out_features

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # 256
        x5 = self.up1(x4, x3)  # 128
        x6 = self.up2(x5, x2)  # 64
        x7 = self.up3(x6, x1)  # 32
        logits = self.outc(x7)
        if self.is_out_features:
            return logits, [x7, x6, x5, x4]
        else:
            return logits

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # 256
        return [x1,x2,x3,x4]
    def decode(self, features):
        x1,x2,x3,x4 = features

        x5 = self.up1(x4, x3)  # 128
        x6 = self.up2(x5, x2)  # 64
        x7 = self.up3(x6, x1)  # 32
        logits = self.outc(x7)
        if self.is_out_features:
            return logits, [x7, x6, x5, x4]
        else:
            return logits

class UNet2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_in_features=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(64, 64, is_in_features)
        self.down2 = Down(128, 128, is_in_features)
        self.down3 = Down(256, 256, is_in_features)
        self.down4 = Down(512, 512, is_in_features)
        self.up1 = Up(512, 768, 256, bilinear)
        self.up2 = Up(256, 384, 128, bilinear)
        self.up3 = Up(128, 192, 64, bilinear)
        self.up4 = Up(64, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.is_in_features = is_in_features

    def encode(self, x, center_features):
        if self.is_in_features:
            x1 = self.inc(x)
            x2 = self.down1((x1, center_features[0]))
            x3 = self.down2((x2, center_features[1]))
            x4 = self.down3((x3, center_features[2]))
            x5 = self.down4((x4, center_features[3]))
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        return [x1,x2,x3,x4,x5]

    def decode(self, features):
        x1,x2,x3,x4,x5 = features
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits

    def forward(self, x, center_features):
        # for feature in features:
        #     print(feature.shape)
        x1 = self.inc(x)
        x2 = self.down1((x1, center_features[0]))
        x3 = self.down2((x2, center_features[1]))
        x4 = self.down3((x3, center_features[2]))
        x5 = self.down4((x4, center_features[3]))

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits

class UNet3_dec(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_in_features=False):
        super(UNet3_dec, self).__init__()
        self.up1 = Up(512, 768, 256, bilinear)
        self.up2 = Up(256, 384, 128, bilinear)
        self.up3 = Up(128, 192, 64, bilinear)
        self.up4 = Up(64, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def decode(self, features):
        x1,x2,x3,x4,x5 = features
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits


class UNet3_dec_scale_3_tp2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_in_features=False):
        super(UNet3_dec_scale_3_tp2, self).__init__()
        self.up1 = Up_single_tp2(256, 768, 256, bilinear)
        self.up2 = Up(256, 384, 128, bilinear)
        self.up3 = Up(128, 192, 64, bilinear)
        self.up4 = Up(64, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def decode(self, features):
        x1,x2,x3,x4,x5 = features
        x6 = self.up1(x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits
class UNet3_dec_scale_2_tp2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_in_features=False):
        super(UNet3_dec_scale_2_tp2, self).__init__()
        self.up2 = Up_single_tp2(128, 384, 128, bilinear)
        self.up3 = Up(128, 192, 64, bilinear)
        self.up4 = Up(64, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def decode(self, features):
        x1,x2,x3,x4,x5 = features
        x7 = self.up2(x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits

class UNet3_dec_scale_1_tp2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_in_features=False):
        super(UNet3_dec_scale_1_tp2, self).__init__()
        self.up3 = Up_single_tp2(64, 192, 64, bilinear)
        self.up4 = Up(64, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def decode(self, features):
        x1,x2,x3,x4,x5 = features
        # x6 = self.up1(x5, x4)
        # x7 = self.up2(x6, x3)
        x8 = self.up3(x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits
class UNet3_dec_scale_0_tp2(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_in_features=False):
        super(UNet3_dec_scale_0_tp2, self).__init__()
        self.up4 = Up_single_tp2(32, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def decode(self, features):
        x1,x2,x3,x4,x5 = features
        # x6 = self.up1(x5, x4)
        # x7 = self.up2(x6, x3)
        # x8 = self.up3(x7, x2)
        x9 = self.up4(x1)
        logits = self.outc(x9)
        return logits
class UNet3(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, is_in_features=False):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64, is_in_features)
        self.down2 = Down(64, 128, is_in_features)
        self.down3 = Down(128, 256, is_in_features)
        self.down4 = Down(256, 512, is_in_features)
        self.up1 = Up(512, 768, 256, bilinear)
        self.up2 = Up(256, 384, 128, bilinear)
        self.up3 = Up(128, 192, 64, bilinear)
        self.up4 = Up(64, 96, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.is_in_features = is_in_features

    def encode(self, x, center_features):
        if self.is_in_features:
            x1 = self.inc(x)
            x2 = self.down1((x1, center_features[0]))
            x3 = self.down2((x2, center_features[1]))
            x4 = self.down3((x3, center_features[2]))
            x5 = self.down4((x4, center_features[3]))
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        return [x1,x2,x3,x4,x5]

    def decode(self, features):
        x1,x2,x3,x4,x5 = features
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits

    def forward(self, x, center_features):
        # for feature in features:
        #     print(feature.shape)
        x1 = self.inc(x)
        x2 = self.down1((x1, center_features[0]))
        x3 = self.down2((x2, center_features[1]))
        x4 = self.down3((x3, center_features[2]))
        x5 = self.down4((x4, center_features[3]))

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_old(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d([2, 2, 2]),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, is_in_features=False):
        super().__init__()
        self.maxpool = nn.MaxPool3d([2, 2, 2])
        self.conv = DoubleConv(in_channels, out_channels)
        self.is_in_features = is_in_features

    def forward(self, inputs):
        if self.is_in_features:
            x, feature = inputs
            x = self.maxpool(x)
            x = self.conv(torch.cat([x, feature], dim=1))
        else:
            x = inputs
            x = self.maxpool(x)
            x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, T_channels, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(T_channels, T_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x2):
        x = self.up(x)
        diffZ = x2.size()[2] - x.size()[2]
        diffY = x2.size()[3] - x.size()[3]
        diffX = x2.size()[4] - x.size()[4]
        if diffX > 0 or diffY > 0 or diffZ > 0:
            x = F.pad(x,
                      [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class Up_single(nn.Module):
    def __init__(self, T_channels, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(T_channels, out_channels)

    def forward(self, x, x2):
        x = self.up(x)
        diffZ = x2.size()[2] - x.size()[2]
        diffY = x2.size()[3] - x.size()[3]
        diffX = x2.size()[4] - x.size()[4]
        if diffX > 0 or diffY > 0 or diffZ > 0:
            x = F.pad(x,
                      [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        return self.conv(x)
class Up_single_tp2(nn.Module):
    def __init__(self, T_channels, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = DoubleConv(T_channels, out_channels)

    def forward(self,x2):
        # x = self.up(x)
        # diffZ = x2.size()[2] - x.size()[2]
        # diffY = x2.size()[3] - x.size()[3]
        # diffX = x2.size()[4] - x.size()[4]
        # if diffX > 0 or diffY > 0 or diffZ > 0:
        #     x = F.pad(x,
        #               [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        return self.conv(x2)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
