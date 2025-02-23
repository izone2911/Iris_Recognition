import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()

        # Encoder path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # Decoder path
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)  # 1024 because of skip connection

        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)  # 512 because of skip connection

        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)  # 256 because of skip connection

        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)   # 128 because of skip connection

        # Final 1x1 convolution layer to get desired output channels
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # Output: [64, 240, 320]
        e2 = self.enc2(self.max_pool(e1))  # Output: [128, 120, 160]
        e3 = self.enc3(self.max_pool(e2))  # Output: [256, 60, 80]
        e4 = self.enc4(self.max_pool(e3))  # Output: [512, 30, 40]
        e5 = self.enc5(self.max_pool(e4))  # Output: [1024, 15, 20]

        # Decoder
        d4 = self.upconv4(e5)               # Output: [512, 30, 40]
        e4_cropped = self.center_crop(e4, d4)  # Crop e4 to match d4 size
        d4 = torch.cat((e4_cropped, d4), dim=1)     # Concatenate skip connection
        d4 = self.dec4(d4)                  # Output: [512, 30, 40]

        d3 = self.upconv3(d4)               # Output: [256, 60, 80]
        e3_cropped = self.center_crop(e3, d3)  # Crop e3 to match d3 size
        d3 = torch.cat((e3_cropped, d3), dim=1)     # Concatenate skip connection
        d3 = self.dec3(d3)                  # Output: [256, 60, 80]

        d2 = self.upconv2(d3)               # Output: [128, 120, 160]
        e2_cropped = self.center_crop(e2, d2)  # Crop e2 to match d2 size
        d2 = torch.cat((e2_cropped, d2), dim=1)     # Concatenate skip connection
        d2 = self.dec2(d2)                  # Output: [128, 120, 160]

        d1 = self.upconv1(d2)               # Output: [64, 240, 320]
        e1_cropped = self.center_crop(e1, d1)  # Crop e1 to match d1 size
        d1 = torch.cat((e1_cropped, d1), dim=1)     # Concatenate skip connection
        d1 = self.dec1(d1)                  # Output: [64, 240, 320]

        # Final segmentation map
        out = self.final_conv(d1)           # Output: [out_channels, 240, 320]
        return out

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def max_pool(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

    def center_crop(self, encoder_feature, target_feature):
        _, _, h, w = target_feature.size()
        enc_h, enc_w = encoder_feature.size(2), encoder_feature.size(3)
        crop_h = (enc_h - h) // 2
        crop_w = (enc_w - w) // 2
        return encoder_feature[:, :, crop_h:crop_h + h, crop_w:crop_w + w]


