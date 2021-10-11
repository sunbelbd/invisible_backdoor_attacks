import paddle
import paddle.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2D(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2D(out_channels),
        nn.ReLU(),
        nn.Conv2D(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2D(out_channels),
        nn.ReLU()
    )


class UNet(nn.Layer):

    def __init__(self, out_channel):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.AvgPool2D(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2D(64, out_channel, 1),
            nn.BatchNorm2D(out_channel),
        )
        
        self.out_layer = nn.Tanh()

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = paddle.concat([x, conv3], 1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = paddle.concat([x, conv2], 1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = paddle.concat([x, conv1], 1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = self.out_layer(out)

        return out
