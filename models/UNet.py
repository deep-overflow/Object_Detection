import torch
import torch.nn as nn

class UNetSegmentationModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.encoder_blocks = nn.ModuleList([
            UNetEncoderBlock(in_channels=3, out_channels=64, down=True),
            UNetEncoderBlock(in_channels=64, out_channels=128, down=True),
            UNetEncoderBlock(in_channels=128, out_channels=256, down=True),
            UNetEncoderBlock(in_channels=256, out_channels=512, down=False),
        ])

        self.decoder_blocks = nn.ModuleList([
            UNetDecoderBlock(in_channels=768, out_channels=256, up=True),
            UNetDecoderBlock(in_channels=384, out_channels=128, up=True),
            UNetDecoderBlock(in_channels=192, out_channels=64, up=True),
            UNetDecoderBlock(in_channels=64, out_channels=self.configs.n_classes, up=False)
        ])
    
    def forward(self, inputs):
        connect1 = self.encoder_blocks[0](inputs) # N -> N / 2
        connect2 = self.encoder_blocks[1](connect1) # N / 2 -> N / 4
        connect3 = self.encoder_blocks[2](connect2) # N / 4 -> N / 8
        encoding = self.encoder_blocks[3](connect3) # N / 8 -> N / 8

        encoding = torch.concat([encoding, connect3], dim=1)
        outputs = self.decoder_blocks[0](encoding) # N / 8 -> N / 4
        outputs = torch.concat([outputs, connect2], dim=1)
        outputs = self.decoder_blocks[1](outputs) # N / 4 -> N / 2
        outputs = torch.concat([outputs, connect1], dim=1)
        outputs = self.decoder_blocks[2](outputs) # N / 2 -> N
        outputs = self.decoder_blocks[3](outputs) # N -> N
        return outputs

class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2 if down else 1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, inputs):
        return self.block(inputs)

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2 if up else 1,
                padding=1,
                output_padding=1 if up else 0,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, inputs):
        return self.block(inputs)