import math
import typing

import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached

class Rotary2D(torch.nn.Module):
    def __init__(self, dim, base=10_000, grid = 16):
        super().__init__()
        half_dim = dim//2
        # prepare the half base for sin-cos modulation.
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)) 
        self.register_buffer('inv_freq', inv_freq)
        self.grid = grid
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(0, self.grid, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            freqs_x = freqs[:, None, :].expand(-1, self.grid, -1)
            freqs_y = freqs[None, :, :].expand(self.grid, -1, -1)
            # freqs_sync = freqs_x + freqs_y
            # emb = torch.cat((freqs_sync.flatten(0,1), freqs_sync.flatten(0,1)), dim =-1).to(x.device)
            emb = torch.concat([freqs_x.flatten(0,1), freqs_y.flatten(0,1)], dim = -1)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # # keep identity transform for v, only rotate q and k.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_old(qkv, cos, sin):
    # this function has been hanged as we are using 2D rope now.
    cos = cos[0,:,0,0,:cos.shape[-1]//2] # in 1d, [1,256,3,1,64] -> [256,32]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)

def apply_rotary_pos_emb(qkv, cos, sin, mode):
    if mode =='2d' :
        cos = cos[0,:,0,0,:] # in 2d, [1,256,3,1,32] -> [256,32]
        sin = sin[0,:,0,0,:]
    elif mode == '1d' :
        cos = cos[0,:,0,0,:cos.shape[-1]//2] # in 1d, [1,256,3,1,64] -> [256,32]
        sin = sin[0,:,0,0,:sin.shape[-1]//2]
    else:
        raise ValueError
    
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


#################################################################################
#                                  Layers                                       #
#################################################################################

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class LayerNorm(nn.Module):
    ## replaced by RMS Norm for stability.
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)

class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = 4 * in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        torch.nn.init.trunc_normal_(self.fc1.weight.data, mean=0.0, std=0.02)
        torch.nn.init.trunc_normal_(self.fc2.weight.data, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.fc1.bias.data)
        torch.nn.init.zeros_(self.fc2.bias.data)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes
        # add a init.
        # torch.nn.init.normal_(self.embedding_table.weight, std=.02)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
        

#################################################################################
#                                 Core d-DIT Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    "Removed adaLN"
    def __init__(self, dim, n_heads, dropout=0.0, position_enc='1d'):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = RMSNorm(dim)
        
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.position_enc = position_enc
        self.dropout = nn.Dropout(dropout)

        self.norm2 = RMSNorm(dim)
        
        self.mlp = Mlp(dim, dim, drop=dropout)


    def forward(self, x, rotary_cos_sin, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        # bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        # attention operation

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype), mode=self.position_enc)
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False)
        
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = x_skip + self.attn_out(x)

        # mlp operation

        x = x + self.mlp(self.norm2(x)) # note that mlp has dropout(0.0 for now)

        return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]

class EmbeddingWithFrozenMasks(nn.Module):
    def __init__(self, dim, vocab_dim, num_masks):
        super().__init__()
        self.num_masks = num_masks
        self.vocab_dim = vocab_dim
        
        # Trainable embeddings (excluding mask tokens)
        self.trainable_embedding = nn.Parameter(torch.empty((vocab_dim - num_masks, dim)))
        torch.nn.init.kaiming_uniform_(self.trainable_embedding, a=math.sqrt(5))
        
        # Frozen embeddings (mask tokens)
        frozen_embeds = torch.empty((num_masks, dim))
        torch.nn.init.kaiming_uniform_(frozen_embeds, a=math.sqrt(5))
        self.register_buffer("frozen_embedding", frozen_embeds)  # No gradients

    def forward(self, x):
        # Concatenate trainable and frozen embeddings
        embedding = torch.cat([self.trainable_embedding, self.frozen_embedding], dim=0)
        return embedding[x]

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels)
        torch.nn.init.trunc_normal_(self.linear.weight.data, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    "This version use condition tokens, instead of adaLN."
    def __init__(self, config, lm_vocab_size: int, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.token_len = config.model.length
        self.hidden_size = config.model.hidden_size
        self.lm_vocab_size = lm_vocab_size
        self.vocab_size = vocab_size
        self.mask_num = vocab_size - lm_vocab_size

        # print(f'Current state = {self.config.mode}.')
        if (self.config.mode != 'train' and self.config.mode != 'debug'):
            self.training = False
            print(f'Current state = {self.config.mode}, set to eval mode.')
        else:
            self.training = True
            print(f'Current state = {self.config.mode}.')

        self.vocab_embed = EmbeddingLayer(self.hidden_size, vocab_size)
        # self.vocab_embed = EmbeddingWithFrozenMasks(config.model.hidden_size, vocab_size, self.mask_num)
        self.label_embed = LabelEmbedder(1000, self.hidden_size)
        self.sigma_map = TimestepEmbedder(self.hidden_size)
        ## both are projected as same tokens.
        if self.config.rope=='2d':
            self.rotary_emb = Rotary2D(self.hidden_size // config.model.n_heads, grid = int((self.token_len)**0.5))
        else:
            self.rotary_emb = Rotary(self.hidden_size // config.model.n_heads)

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(DDiTBlock(
                self.hidden_size,
                config.model.n_heads,
                dropout=config.model.dropout,
                position_enc=self.config.rope))
        self.blocks = nn.ModuleList(blocks)
        # introducing repa:

        if (self.config.repa_loss.use_repa==True and self.config.mode!='eval'):
            self.projectors = nn.ModuleList([
                build_mlp(self.config.model.hidden_size, self.config.repa_loss.projector_dim, self.config.repa_loss.z_dim)])
        else:
            print('RepA disabled, skip loading.')
            self.projectors = None

        self.head = FinalLayer(self.hidden_size, vocab_size)
        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, labels, indices, sigma): # y, x, t
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            token_embeds = self.vocab_embed(indices)  # [bs, psz**2, hidden_size]
            y = self.label_embed(labels).unsqueeze(1)
            t = self.sigma_map(sigma).unsqueeze(1)
            x = torch.cat([token_embeds, t, y], dim=1)
            # c = t + y 
            rotary_cos_sin = self.rotary_emb(x) # tuple of 2*[1,256,3,1,64]
            N, _, D = x.shape
            T = self.token_len
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)
                if i+1==8 and self.training and self.config.repa_loss.use_repa==True:
                    zs = [projector(x[:,:T, ...].reshape(-1, D)).reshape(N, T, -1) for projector in self.projectors]
            if self.config.repa_loss.use_repa==False or self.training==False:
                zs = None
            x = self.head(x)
        return x[:, :self.token_len, ...], zs
