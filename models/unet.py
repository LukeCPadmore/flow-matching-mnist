from typing import Optional, Any
import torch 
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self,embedding_dim:int = 128, trunk_dim:int = 32, max_period: float = 10000.0):
        assert embedding_dim % 2 == 0, "Use an even embedding dimension."
        super().__init__() 
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
        t = t.squeeze()                      
        B = t.shape[0]
        half = self.embedding_dim // 2

        i = torch.arange(half, device=t.device, dtype=torch.float32)  # [0..half-1]
        inv_freq = torch.exp(-self.log_max_period * (i / half))  # (half,)

        angles = t[:, None] * inv_freq[None, :]                       # (B, half)

        # Interleave: stack sines & cosines then flatten the last two dims
        emb = torch.stack([angles.sin(), angles.cos()], dim=-1)       # (B, half, 2)
        emb = emb.flatten(start_dim=1)                                 # (B, d)
        return emb
    
    def forward(self,t:torch.Tensor):
        time_embeddings = self._sinusoidal_time_embedding(t)
        return self.mlp(time_embeddings)
    
class SimpleClassConditioning(nn.Module):
    def __init__(self,cls_dim, embedding_dim, trunk_dim):
        super().__init__()
        self.cond_emb = nn.Embedding(cls_dim,embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim,trunk_dim),
            nn.SiLU(),
            nn.Linear(trunk_dim,trunk_dim)
        )
    def forward(self,cls_idx):
        cls_embedding = self.cond_emb(cls_idx)
        return self.mlp(cls_embedding)

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
        super().__init__()
        self.d_concat = d_concat
        self.group_norm_size = group_norm_size
        self.conv =  nn.Sequential(
            nn.GroupNorm(self.group_norm_size,in_channels+self.d_concat),
            nn.Conv2d(in_channels+self.d_concat,out_channels,kernel_size=3,padding=1,stride=1,bias=False),
            nn.SiLU(),
            nn.GroupNorm(self.group_norm_size,out_channels),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.SiLU()
        )
        self.down = nn.Sequential(
            nn.GroupNorm(self.group_norm_size,out_channels),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.SiLU(),
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
        x_skip_features = self.conv(x)
        x = self.down(x_skip_features)
        return x, x_skip_features
    
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
        super().__init__()
        self.d_concat = d_concat
        self.group_norm_size = group_norm_size
        self.upsampler = nn.Upsample(scale_factor = 2 ,mode = upsample_mode)
        self.block = nn.Sequential( 
            nn.GroupNorm(self.group_norm_size,2*in_channels+self.d_concat),
            nn.Conv2d(2*in_channels+self.d_concat,out_channels,kernel_size=3,padding=1,stride=1,bias=False),
            nn.SiLU(),
            nn.GroupNorm(self.group_norm_size,out_channels),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.SiLU()
        )
        self.time_emb_mlp = nn.Linear(d_trunk,d_concat)

    def forward(self,x:torch.Tensor, time_emb:torch.Tensor, skip_features:torch.Tensor) -> torch.Tensor:
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
    def __init__(self,channels:list[int],d_trunk,d_concat,group_norm_size=8):
        """
        For each channel create DownConvblock with in_channels = channel[i] and out_channels = channels[i+1]
        """
        super().__init__()
        # Lift number of channel to make GroupNorm work
        self.channels = channels.copy()
        self.initial_conv = nn.Conv2d(in_channels = channels[0], out_channels = channels[1], kernel_size = 3, padding = 1, stride = 1) 
        self.channels[0] = self.channels[1]
        self.down_blocks = nn.ModuleList([
            ConvDownblock(
                in_channels, 
                out_channels,
                d_trunk,
                d_concat,
                group_norm_size=group_norm_size
            )
            for in_channels,out_channels in zip(self.channels[:-1],self.channels[1:])
        ])
        
    def forward(self,x,time_emb):
        """
        Loop through downblocks and save output tensors to skip_features to later pass to decoder
        """
        skip_features = [] 
        x = self.initial_conv(x)
        for block in self.down_blocks:
            x, x_skip_features = block(x,time_emb)
            skip_features.append(x_skip_features)
        return x, skip_features


class Decoder(nn.Module):
    def __init__(self,channels:list[int],d_trunk,d_concat,group_norm_size=8,upsample_mode='nearest'):
        """
        Expects list of channels with the same orders as te encoder
        """
        super().__init__()
        chls = channels.copy()
        self.out_channels = chls.pop(0)
        self.channels = chls[::-1]
        self.channels.append(self.channels[-1])
        self.up_blocks = nn.ModuleList([
            ConvUpblock(
                in_channels, 
                out_channels,
                d_trunk,
                d_concat,
                group_norm_size=group_norm_size, 
                upsample_mode=upsample_mode
            )
            for in_channels,out_channels in zip(self.channels[:-1],self.channels[1:])
        ])
        self.final_conv = nn.Conv2d(self.channels[-1],self.out_channels,kernel_size = 3,stride = 1, padding = 1)
    def forward(self, x, time_emb: torch.Tensor, skip_features: list[torch.Tensor]) -> torch.Tensor:
        for block in self.up_blocks:
            x = block(x,time_emb,skip_features.pop())
        return  self.final_conv(x)


class Bottleneck(nn.Module): 
    def __init__(self,channels,d_trunk,d_concat,group_norm_size=8):
        """
        Uses similar architecture to ConvDownblock but doesn't use stride = 2 to halve image size
        """
        super().__init__()
        self.channels = channels
        self.d_trunk = d_trunk
        self.d_concat = d_concat
        self.bottleneck = nn.Sequential( 
            nn.GroupNorm(group_norm_size,self.channels+self.d_concat),
            nn.Conv2d(self.channels+self.d_concat,self.channels,kernel_size=3,padding=1,stride=1,bias=False),
            nn.SiLU(),
            nn.GroupNorm(group_norm_size,self.channels),
            nn.Conv2d(self.channels,self.channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.SiLU()
        )
        self.time_emb_mlp = nn.Linear(d_trunk,d_concat, bias = False)

    def forward(self, x:torch.Tensor, time_emb:torch.Tensor) -> torch.Tensor:
        """
        Put time embeddings though MLP (B,d_trunk) -> (B,d_concat) 
        Then reshape -> (B,d_concat,1,1)
        Then expand -> (B,d_concat,H,W)
        """
        time_emb = self.time_emb_mlp(time_emb)[:,:,None,None] 
        time_emb = time_emb.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x,time_emb],dim=1)
        return self.bottleneck(x)
        
        
class UNet(nn.Module):
    def __init__(self,channels:list[int],d_trunk = 8, d_concat = 8, group_norm_size = 8, d_time = 128):
        super().__init__()
        self.channels = channels
        self.encoder = Encoder(channels,d_trunk, d_concat, group_norm_size)
        self.decoder = Decoder(channels,d_trunk, d_concat, group_norm_size)
        self.bottleneck = Bottleneck(self.channels[-1],d_trunk,2 * d_concat,group_norm_size)
        self.time_embedding_mlp = SinusoidalTimeEmbedding(d_time, d_trunk, 2 * d_concat)

    def forward(self,x,t,*args, **kwargs) -> torch.Tensor:
        time_emb = self.time_embedding_mlp(t)
        x, skip_features = self.encoder(x,time_emb)
        x = self.bottleneck(x,time_emb)
        x = self.decoder(x,time_emb,skip_features)
        return x

class CondUNet(nn.Module):
    def __init__(self,channels:list[int], cond_dim, d_trunk = 8, d_concat = 8, group_norm_size = 8, d_time = 128, d_cls_emb = 128):
        super().__init__()
        # 2 * trunk for individual ConvBlock MLP dimensions as coming from time embedding and class embedding
        self.channels = channels
        self.encoder = Encoder(channels,2 * d_trunk, d_concat,group_norm_size)
        self.decoder = Decoder(channels,2 * d_trunk, d_concat,group_norm_size)
        self.bottleneck = Bottleneck(self.channels[-1],2 * d_trunk, d_concat,group_norm_size)
        self.time_embedding_mlp = SinusoidalTimeEmbedding(d_time, d_trunk)
        self.simple_cond_mlp = SimpleClassConditioning(cond_dim, d_cls_emb , d_trunk)

    def forward(self,x,t,c,*args,**kwargs) -> torch.Tensor:
        cond_emb = self.simple_cond_mlp(c)
        time_emb = self.time_embedding_mlp(t)
        combined_emb = torch.cat([cond_emb,time_emb], dim = 1)
        x, skip_features = self.encoder(x,combined_emb)
        x = self.bottleneck(x,combined_emb)
        x = self.decoder(x,combined_emb,skip_features)
        return x

