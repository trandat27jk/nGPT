from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.nn.utils.parametrize import register_parametrization

#parallel computing
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    heads: int = 12
    n_layer: int = 12
    n_embd: int = 768
    bias: bool = False
    parametrize: bool = True
    factor: int = 4
    dropout: float=0.0


class AttentionConfig:
    d: int = -1
    groups: int = 1
    norm_eps: int = 0
    eps: float = 1e-6
    init_scale = 1
    scale: int = 1


class FFConfig:
    d: int = -1
    groups: int = 1
    norm_eps: int = 0
    eps: float = 1e-6
    init_scale: int = 1
    scale: int = 1


def exist(v):
    return v is not None


def default(v, d):
    return v if exist(v) else d


def l2Norm(x, d=-1, groups=1, eps=1e-6, norm_eps=0):
    eps = default(eps, 1e-5 if x.dtype == torch.float16 else 1e-10)

    if groups > 1:
        x = x.chunk(groups, dim=d)
        x = torch.stack(x)

    if norm_eps == 0:
        x_norm = F.normalize(x, dim=d, p=2, eps=eps)

    if norm_eps != 0:
        norm = x.norm(dim=d, keepdim=True)
        d_norm = norm.detach().clamp(min=1 - norm_eps, max=1 + norm_eps)
        divisor = norm / d_norm
        x_norm = x / divisor.clamp(min=eps)

    if groups > 1:
        x_norm = torch.cat([*x_norm], dim=d)

    return x_norm


class L2Norm(nn.Module):
    def __init__(self, d=-1, groups=1, eps=1e-6, norm_eps=0):
        super().__init__()
        self.d = d
        self.groups = groups
        self.eps = eps
        self.norm_eps = norm_eps

    def forward(self, x):
        return l2Norm(
            x, d=self.d, groups=self.groups, eps=self.eps, norm_eps=self.norm_eps
        )


class LinearNormWeight(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        parametrize=False,
        groups=1,
        d=-1,
        eps=1e-6,
        norm_eps=0,
        bias=False,
    ):
        super().__init__()
        self.scale = groups**-1
        self.parametrize = parametrize
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.L2Norm = L2Norm(d, groups, eps, norm_eps)
        if parametrize:
            register_parametrization(self.linear, "weight", self.L2Norm)

        self.norm_weight_()

    @torch.no_grad()
    def norm_weight_(self):
        if self.parametrize:
            norm = self.weights
            original = self.linear.parametrizations.weight.original
            original.copy_(norm)
        else:
            self.weights.copy_(self.L2Norm(self.weights))

    @property
    def weights(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x) * self.scale


class Scale(nn.Module):
    def __init__(self, dim, init_scale=1, scale=1):
        super().__init__()
        self.params = nn.Parameter(torch.ones(dim) * scale)
        self.divide_scale = init_scale / scale

    def forward(self):
        return self.params * self.divide_scale


class Attention(nn.Module):
    def __init__(self, args: ModelConfig, args_attn: AttentionConfig):
        super().__init__()
        self.args = args
        self.to_q = LinearNormWeight(
            args.n_embd,
            args.n_embd,
            args.parametrize,
            args_attn.groups,
            args_attn.d,
            args_attn.eps,
            args_attn.norm_eps,
        )
        self.to_k = LinearNormWeight(
            args.n_embd,
            args.n_embd,
            args.parametrize,
            args_attn.groups,
            args_attn.d,
            args_attn.eps,
            args_attn.norm_eps,
        )
        self.to_v = LinearNormWeight(
            args.n_embd,
            args.n_embd,
            args.parametrize,
            args_attn.groups,
            args_attn.d,
            args_attn.eps,
            args_attn.norm_eps,
        )

        self.dim_head = args.n_embd // args.heads
        self.n_heads = args.heads
        self.softmax_scale = self.dim_head**0.5
        self.q_scale = Scale(args.n_embd, 1, args.n_embd ** (-0.5))
        self.k_scale = Scale(args.n_embd, 1, args.n_embd ** (-0.5))
        self.rotary_embed=RotaryEmbedding(self.dim_head)
        self.flash=hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.dropout=args.dropout
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(args.block_size, args.block_size).view(
                        1, 1, args.block_size, args.block_size
                    )
                ),)
            
        self.c_proj = LinearNormWeight(
            args.n_embd,
            args.n_embd,
            args.parametrize,
            args_attn.groups,
            args_attn.d,
            args_attn.eps,
            args_attn.norm_eps,
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
    
        q = q * rearrange(self.q_scale(), "(h d) -> h 1 d", h=self.n_heads)
        k = k * rearrange(self.q_scale(), "(h d) -> h 1 d", h=self.n_heads)
        if self.flash:
            attn=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            attn = q @ k.transpose(-1, -2)
    
            attn = attn * self.softmax_scale
    
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v)
        out = attn.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)


class FeedForward(nn.Module):
    def __init__(self, args: ModelConfig, args_ffn: FFConfig):
        super().__init__()
        hidden_dim = args.factor * args.n_embd
        self.w1 = LinearNormWeight(args.n_embd, hidden_dim)
        self.w2 = LinearNormWeight(hidden_dim, args.n_embd)
        self.w3 = LinearNormWeight(args.n_embd, hidden_dim)

        self.scale_u = Scale(
            hidden_dim, init_scale=args_ffn.init_scale, scale=args_ffn.scale
        )
        self.scale_v = Scale(
            hidden_dim, init_scale=args_ffn.init_scale, scale=args_ffn.scale
        )
        self.scale_ = hidden_dim**0.5

    def forward(self, x):
        u = self.w1(x)*self.scale_u()
        
        v = self.w3(x)*self.scale_v()

        v = v * self.scale_

        return self.w2(F.silu(v) * u)


class Lerp_Residual(nn.Module):
    def __init__(self, args: ModelConfig, index_layer, fc):
        super().__init__()
        self.fc = fc
        self.l2Norm = L2Norm(d=-1)
        self.scale = Scale(
            args.n_embd, init_scale=(0.05 / (index_layer+1)), scale=args.n_embd ** (-0.5)
        )

    def forward(self, x, **kwargs):
        connect_ = x
        out = self.l2Norm(self.fc(x, **kwargs))
        out = torch.lerp(connect_, out, self.scale())

        return self.l2Norm(out)


class nGPT(nn.Module):
    def __init__(
        self, args: ModelConfig, args_attn: AttentionConfig, args_ffn: FFConfig
    ):
        super().__init__()
        self.n_layer = args.n_layer
        self.n_attn_layeers = nn.ModuleList(
            [Attention(args, args_attn) for i in range(args.n_layer)]
        )
        self.n_ffn_layers = nn.ModuleList(
            [FeedForward(args, args_ffn) for i in range(args.n_layer)]
        )
        self.residual_attn = nn.ModuleList(
            [
                Lerp_Residual(args, i, self.n_attn_layeers[i])
                for i in range(args.n_layer)
            ]
        )
        self.residual_ffn = nn.ModuleList(
            [Lerp_Residual(args, i, self.n_ffn_layers[i]) for i in range(args.n_layer)]
        )
        self.to_logits = LinearNormWeight(args.n_embd, args.vocab_size)
        self.scale_logits=Scale(args.vocab_size,1,args.n_embd**-0.5)
        self.to_embedding=nn.Embedding(args.vocab_size,args.n_embd)
        self.block_size=args.block_size
    def forward(self, x,targets=None):
        
        x=self.to_embedding(x)
        B, T, C = x.size()
        for residual_attn, residual_ffn in zip(self.residual_attn, self.residual_ffn):
            x = residual_attn(x)
            x = residual_ffn(x)
        logits = (self.to_logits(x)*self.scale_logits())
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        else: 
            loss=None

        return loss,logits

    @torch.no_grad()
    def generate(self,idx,max_new_tokens,temperature=1.0,top_k=None):
        for i in range(max_new_tokens):
            idx_cond=idx if idx.size(1) <self.block_size else idx[:,-self.block_size:]
            _,logits=self(idx_cond)
            logits=logits[:,-1,:]/temperature
            if top_k is not None:
                v,_=torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits<v[:,[-1]]]=-float('Inf')
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx 