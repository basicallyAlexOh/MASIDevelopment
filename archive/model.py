from torch import cat, randn
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation, padding=padding),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size, stride=stride, dilation=dilation, padding=padding),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, channels=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.blocks = nn.ModuleList(ConvBlock(channels[i], channels[i+1]) for i in range(len(channels)-1))
        self.pools = nn.ModuleList(nn.MaxPool3d(2, stride=2) for i in range(len(channels)-2))

    def forward(self, x):
        features = []
        for i, pool in enumerate(self.pools):
            x = self.blocks[i](x)
            features.append(x)
            x = pool(x)
        x = self.blocks[-1](x) # base
        return x, features

class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64), nclasses=6):
        super().__init__()
        self.decs = nn.ModuleList(ConvBlock(channels[i], channels[i+1]) for i in range(len(channels)-1))
        self.deconvs = nn.ModuleList(ConvTransposeBlock(channels[i], channels[i+1]) for i in range(len(channels)-1))
        self.one_conv = nn.Conv3d(64, nclasses, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, encoder_features):
        for i, deconv in enumerate(self.deconvs):
            encoder_feature = encoder_features[len(encoder_features)-(i+1)]
            x = cat([deconv(x), encoder_feature], dim=1) # feature dim = 2*channel
            x = self.decs[i](x) # in_ch=2*channel, out_ch=channel
        print(f"final conv {x.shape}")
        return self.sigmoid(self.one_conv(x))


class Unet3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self,x):
        x, features = self.encoder(x)
        print(f"x {x.shape}")
        x = self.decoder(x, features)
        return x

if __name__ == "__main__":
    # unet = Unet3d()
    # x = randn(1,1,192,192,128)
    # seg = unet(x)
    # print(f"seg {seg.shape}")
    encoder = Encoder()
    x = randn(1, 1, 192,192,128)
    x, features = encoder(x)
    for feature in features:
        print(feature.shape)
    # decoder = Decoder()
    # # base_x = randn(1, 1024, 16, 16, 17)
    # decoder(x, features)