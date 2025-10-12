from types import List

import torch 
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self,embedding_dim:int = 128, trunk_dim:int = 32, max_period: float = 10000.0):
        assert embedding_dim % 2 == 0, "Use an even embedding dimension."
        self.embedding_dim = embedding_dim
        self.log_max_period = math.log(max_period)
        self.trunk_dim = trunk_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim,trunk_dim),
            nn.SiLU(),
            nn.Linear(trunk_dim,trunk_dim)
        )

    def _sinusoidal_time_embedding(self,t: torch.Tensor):
        """
        t: shape (B,) or (B, 1) — timesteps (float or int)
        d: even embedding dimension
        returns: (B, d) with interleaved [sin(w0 t), cos(w0 t), sin(w1 t), cos(w1 t), ...]
        """
        t = t.view(-1)                         
        B = t.shape[0]
        half = self.embedding_dim // 2

        i = torch.arange(half, device=t.device, dtype=torch.float32)  # [0..half-1]
        inv_freq = torch.exp(-self.log_max_period * (2.0 * i / d))   # (half,)

        angles = t[:, None] * inv_freq[None, :]                       # (B, half)

        # Interleave: stack sines & cosines then flatten the last two dims
        emb = torch.stack([angles.sin(), angles.cos()], dim=-1)       # (B, half, 2)
        emb = emb.flatten(start_dim=1)                                 # (B, d)
        return emb
    
    def forward(self,t:torch.Tensor):
        time_embeddings = self._sinusoidal_time_embedding(t)
        return self.mlp(time_embeddings)

class ConvDownblock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_trunk,
                 d_concat,
                 group_norm_size = 8):
        """
        Deciding to use stride in order to downsample instead of maxpooling
        """
        self.d_concat = d_concat
        self.group_norm_size = group_norm_size
        self.block = nn.Sequential( 
            nn.GroupNorm(self.group_norm_size,in_channels+self.d_concat),
            nn.Conv2d(in_channels+self.d_concat,out_channels,kernel_size=3,padding=1,stride=1,bias=False),
            nn.SiLU(),
            nn.GroupNorm(self.group_norm_size,out_channels),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.SiLU()
        )
        self.time_emb_mlp = nn.Linear(d_trunk,d_concat)

    def forward(self,x:torch.Tensor, time_emb:torch.Tensor) -> torch.Tensor:
        """
        Put time embeddings though MLP (B,d_trunk) -> (B,d_concat) 
        Then reshape -> (B,d_concat,1,1)
        Then expand -> (B,d_concat,H,W)
        """
        time_emb = self.time_emb_mlp(time_emb)[:,:,None,None] 
        time_emb = time_emb.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x,time_emb],dim=1)
        return self.block(x)
    
class ConvUpblock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_trunk,
                 d_concat,
                 group_norm_size = 8,
                 upsample_mode='nearest'):
        """
        Note input will be sized (B,2*in_channels + d_concat,H,W)
        """
        self.d_concat = d_concat
        self.group_norm_size = group_norm_size
        self.upsampler = nn.Upsample(2,mode = upsample_mode)
        self.block = nn.Sequential( 
            nn.GroupNorm(self.group_norm_size,2*in_channels+self.d_concat),
            nn.Conv2d(2*in_channels+self.d_concat,out_channels,kernel_size=3,padding=1,stride=1,bias=False),
            nn.SiLU(),
            nn.GroupNorm(self.group_norm_size,out_channels),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.SiLU()
        )
        self.time_emb_mlp = nn.Linear(d_trunk,d_concat)

    def forward(self,x:torch.Tensor, skip_features:torch.Tensor, time_emb:torch.Tensor) -> torch.Tensor:
        """
        Put time embeddings though MLP (B,d_trunk) -> (B,d_concat) 
        Then reshape -> (B,d_concat,1,1)
        Then expand -> (B,d_concat,H,W)
        """
        # Upsample x
        x = self.upsampler(x)
        # Get time embeddings
        time_emb = self.time_emb_mlp(time_emb)[:,:,None,None] 
        time_emb = time_emb.expand(-1, -1, x.size(2), x.size(3))
        # Concat x, skip features and time embeddings
        x = torch.cat([x,skip_features,time_emb],dim=1)
        # Pass through block
        return self.block(x)
    

class Encoder(nn.Module): 
    def __init__(self,channels:List[int],d_trunk,d_concat):
        """
        For each channel create DownConvblock with in_channels = channel[i] and out_channels = channels[i+1]
        """
        self.blocks = nn.ModuleList([
            ConvDownblock(
                in_channels, 
                out_channels,
                d_trunk,
                d_concat,
            )
            for in_channels,out_channels in zip(channels[:-1],channels[1:])
        ])

class Decoder(nn.Module):
    def __init__(self,channels:List[int]):
        pass

class Bottleneck(nn.Module): 
    pass


class Unet(nn.Module):
    def __init__(self,channels:List[int],device = None):
        self.encoder = None
        self.decoder = None
        self.bottleneck = None
        self.time_embedding_mlp = None

    def forward(self) -> torch.Tensor:
        pass