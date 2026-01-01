from typing import Optional,List, Any
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self,x):
        return self.block(x)

class FID_classifier(nn.Module):
    def __init__(self, conv_block_channels:List[int] = (1,32,64), fid_emb=128, n_classes=10, image_size = 32):
        super().__init__()
        self.fid_emb = 128
        self.conv_block_channels = conv_block_channels
        self.n_classes = n_classes
        self.image_size = image_size
        blocks = [ConvBlock(self.conv_block_channels[i],self.conv_block_channels[i+1]) for i in range(len(self.conv_block_channels)-1)]
        self.conv_blocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_block_channels[-1] * (image_size // (2**(len(conv_block_channels)-1)))**2,self.fid_emb),
            nn.ReLU(),
            nn.Linear(fid_emb, n_classes)
        )

    def forward(self,x):
        x = self.conv_blocks(x)
        x = self.out(x)
        return x
    

