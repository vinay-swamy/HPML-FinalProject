# TransformerLayer adapted from:
#    https://github.com/facebookresearch/esm/blob/main/esm/modules.py
# FlashMHA adopted from:
#    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Note: RotaryEmbedding from FlashAttention is incompatible with ESM.
#  I took elements from both to get consistent results and optimize speed.
#  Both rotary embedding implementations are likely correct.


import torch
import torch.nn as nn
from esm.multihead_attention import MultiheadAttention  # noqa
from einops import rearrange
from rotary import RotaryEmbeddingESM
import math
import torch.nn.functional as F

import importlib
fa_is_installed = importlib.util.find_spec("flash_attn") is not None
lora_is_installed = importlib.util.find_spec("loralib") is not None
if lora_is_installed:
    import loralib as lora

#@torch.jit.script  # would require setting shape to static (or finite number of shapes)
def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))    


class SwiGLUB(nn.Module):
    """SwisGLU activation with trainable per-channel beta, combine with fc1
    Replaces the first linear layer of FFN.
    Beta allows swish function to interpolate between linear(0) and relu(inf)
    SWISH: A SELF-GATED ACTIVATION FUNCTION arXiv:1710.05941
    GLU Variants Improve Transformer  arXiv:2002.05202v1
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_out = dim_out
        self.beta = nn.Parameter(torch.ones(dim_in))
        self.linear = nn.Linear(dim_in, dim_out*2, bias=bias)
    
    def forward(self, x):
        x[..., :self.dim_out] *= self.beta  # gate
        x = self.linear(x)
        return F.silu(x[..., :self.dim_out]) * x[..., self.dim_out:]


class SwiGLU(nn.Module):
    """SwisGLU activation , combine with fc1
    Replaces the first linear layer of FFN.
    SWISH: A SELF-GATED ACTIVATION FUNCTION arXiv:1710.05941
    GLU Variants Improve Transformer  arXiv:2002.05202v1
    """
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out*2, bias=bias)
    
    def forward(self, x):
        x = self.linear(x)
        return F.silu(x[..., :self.dim_out]) * x[..., self.dim_out:]  # gate * x

    
if fa_is_installed:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
    from flash_attn.bert_padding import unpad_input

    class FlashMHASelfMaskKV(nn.Module):

        def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                     causal=False, use_rotary_emb=None, device=None, dtype=None,
                     lora_qv_rank=None, lora_alpha=1, **kwargs) -> None:
            assert batch_first
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.embed_dim = embed_dim
            self.causal = causal
            self.dropout_p = attention_dropout

            self.num_heads = num_heads
            assert self.embed_dim % num_heads == 0, f"emb {self.embed_dim} must be divisible by num_heads {num_heads}"
            self.head_dim = self.embed_dim // num_heads
            # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
            assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
            self.scaling = self.head_dim ** -0.5

            self.use_rotary_emb = use_rotary_emb
            if use_rotary_emb:
                self.rot_emb = RotaryEmbeddingESM(self.head_dim)

            if lora_qv_rank is not None:
                self.Wqkv = lora.MergedLinear(embed_dim, 3*embed_dim, r=lora_qv_rank, lora_alpha=lora_alpha,
                                              enable_lora=[True, False, True])
            else:
                self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)

            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        def forward(self, x, key_padding_mask=None, need_weights=False):
            """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
            key_padding_mask: bool tensor of shape (batch, seqlen)
            Credit: some elements adopted from OpenFold:
            https://github.com/aqlaboratory/openfold/blob/feed4ae22edf899b37bee49293fff902bdd64e2d/openfold/model/primitives.py#L660
            """
            qkv = self.Wqkv(x)
            dtype = qkv.dtype
            q, k, v = rearrange(qkv, 'b s (three h d) -> three b s h d',
                                three=3, h=self.num_heads)
            # q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).unbind(dim=2)
            b_size, s_size, _, _ = q.shape
            q_cu_seqlens = torch.arange(
                0, (b_size + 1) * s_size, step=s_size, dtype=torch.int32, device=q.device
            )

            if self.use_rotary_emb:
                q, k = self.rot_emb(q, k, seq_dimension=-3)
            q = rearrange(q.type(dtype), 'b s h d -> (b s) h d',
                          h=self.num_heads)
            q = q * self.scaling

            # [b s 2 h d]
            kv = torch.stack([k.type(dtype), v], dim=2)

            if key_padding_mask is not None:
                kv = rearrange(kv, 'b s two h d -> b s (two h d)',
                               two=2, h=self.num_heads)
                key_padding_mask = key_padding_mask.type(dtype)
                kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(
                    kv, key_padding_mask
                )
                kv_unpad = rearrange(kv_unpad, 'nnz (two h d) -> nnz two h d',
                                     two=2, h=self.num_heads)
            else:
                kv_unpad = rearrange(kv, 'b s two h d -> (b s) two h d',
                                     two=2, h=self.num_heads)
                kv_cu_seqlens = q_cu_seqlens
                kv_max_s = s_size

            # context = flash_attn_unpadded_kvpacked_func(  FA1
            context = flash_attn_varlen_kvpacked_func(
                q,
                kv_unpad,
                q_cu_seqlens,
                kv_cu_seqlens,
                s_size,
                kv_max_s,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=1.,  # apply on q above
            )
            context = rearrange(context, '(b s) h d -> b s (h d)',
                                b=b_size, h=self.num_heads)
            return self.out_proj(context), None

    class FlashMHACrossMaskKV(nn.Module):
        '''Cross attention by swapping Q
        Expect single tensor x; concat of protein pair along axis 0, i.e., [(2 b) s e]
        '''

        def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                     use_rotary_emb=None, device=None, dtype=None,
                     lora_qv_rank=None, lora_alpha=1, **kwargs) -> None:
            assert batch_first
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.embed_dim = embed_dim
            self.dropout_p = attention_dropout

            self.num_heads = num_heads
            assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
            self.head_dim = self.embed_dim // num_heads
            # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
            assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
            self.scaling = self.head_dim ** -0.5

            self.use_rotary_emb = use_rotary_emb
            if use_rotary_emb:
                self.rot_emb = RotaryEmbeddingESM(self.head_dim)

            if lora_qv_rank is not None:
                self.Wqkv = lora.MergedLinear(embed_dim, 3*embed_dim, r=lora_qv_rank,
                                              lora_alpha=lora_alpha, enable_lora=[True, False, True])
            else:
                self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        def forward(self, x, key_padding_mask=None, need_weights=False):
            """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
            key_padding_mask: bool tensor of shape (batch, seqlen)
            Credit: some elements adopted from OpenFold:
            https://github.com/aqlaboratory/openfold/blob/feed4ae22edf899b37bee49293fff902bdd64e2d/openfold/model/primitives.py#L660
            """
            qkv = self.Wqkv(x)  # [(2 b) s e]
            dtype = qkv.dtype
            qkv = rearrange(qkv, '(two b) s  e -> two b s e', two=2)
            # switches q1 and q2. Flip makes a copy  and reverses dimension
            qkv[...,:self.embed_dim] = qkv[...,:self.embed_dim].flip([0])  # flip makes copy

            q, k, v = rearrange(qkv, 'two b s (three h d) -> three (two b) s h d',
                                two=2, three=3, h=self.num_heads)
            # q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).unbind(dim=2)
            b_size, s_size, _, _ = q.shape
            q_cu_seqlens = torch.arange(
                0, (b_size + 1) * s_size, step=s_size, dtype=torch.int32, device=q.device
            )

            if self.use_rotary_emb:
                q, k = self.rot_emb(q, k, seq_dimension=-3)
            q = rearrange(q.type(dtype), 'b s h d -> (b s) h d',
                          h=self.num_heads)
            q = q * self.scaling

            # [b s 2 h d]
            kv = torch.stack([k.type(dtype), v], dim=2)

            if key_padding_mask is not None:
                kv = rearrange(kv, 'b s two h d -> b s (two h d)',
                               two=2, h=self.num_heads)
                key_padding_mask = key_padding_mask.type(dtype)
                kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(
                    kv, key_padding_mask
                )
                kv_unpad = rearrange(kv_unpad, 'nnz (two h d) -> nnz two h d',
                                     two=2, h=self.num_heads)
            else:
                kv_unpad = rearrange(kv, 'b s two h d -> (b s) two h d',
                                     two=2, h=self.num_heads)
                kv_cu_seqlens = torch.arange(
                    0, (b_size + 1) * k.shape[1], step=k.shape[1],
                    dtype=torch.int32, device=q.device
                )
                kv_max_s = k.shape[1]

            # context = flash_attn_unpadded_kvpacked_func(  FA1
            context = flash_attn_varlen_kvpacked_func(
                q,
                kv_unpad,
                q_cu_seqlens,
                kv_cu_seqlens,
                s_size,
                kv_max_s,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=1.,  # apply on q above
            )
            context = rearrange(context, '(b s) h d -> b s (h d)',
                                b=b_size, h=self.num_heads)
            return self.out_proj(context), None


    class FlashMHACrossQueryMaskKV(nn.Module):
        '''MHA attention with separate Q
        !NOTE! query doesn't have q_proj, implement outside if needed
        Expect tensor x; concat of protein pair along axis 0, i.e., [(2 b) s e]
        '''

        def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                     use_rotary_emb=None, device=None, dtype=None, lora_qv_rank=None,lora_alpha=1,
                     **kwargs) -> None:
            assert batch_first
            factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            self.embed_dim = embed_dim
            self.dropout_p = attention_dropout

            self.num_heads = num_heads
            assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
            self.head_dim = self.embed_dim // num_heads
            # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
            assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
            self.scaling = self.head_dim ** -0.5

            self.use_rotary_emb = use_rotary_emb
            if use_rotary_emb:
                self.rot_emb = RotaryEmbeddingESM(self.head_dim)
            if lora_qv_rank is not None:
                self.Wkv = lora.MergedLinear(embed_dim, 2*embed_dim, r=lora_qv_rank,
                                             lora_alpha=lora_alpha, enable_lora=[True, False, True])
            else:
                self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)

            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        def forward(self, x, q1, q2, key_padding_mask=None, need_weights=False):
            """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
            q1 corresponds to Q of first half of x, q2 second half
            key_padding_mask: bool tensor of shape (batch, seqlen)
            Credit: some elements adopted from OpenFold:
            https://github.com/aqlaboratory/openfold/blob/feed4ae22edf899b37bee49293fff902bdd64e2d/openfold/model/primitives.py#L660
            """
            kv = self.Wkv(x)  # [(2 b) s e]
            dtype = kv.dtype
            q = rearrange(torch.cat([q2, q1], dim=0), 'b s (h d) -> b s h d', h=self.num_heads)
            k, v = rearrange(kv, 'b s (two h d) -> two b s h d',
                                two=2, h=self.num_heads)
            b_size, s_size, _, _ = q.shape
            q_cu_seqlens = torch.arange(
                0, (b_size + 1) * s_size, step=s_size, dtype=torch.int32, device=q.device
            )

            if self.use_rotary_emb:
                q, k = self.rot_emb(q, k, seq_dimension=-3)
            q = rearrange(q.type(dtype), 'b s h d -> (b s) h d',
                          h=self.num_heads)
            q = q * self.scaling

            # [b s 2 h d]
            kv = torch.stack([k.type(dtype), v], dim=2)

            if key_padding_mask is not None:
                kv = rearrange(kv, 'b s two h d -> b s (two h d)',
                               two=2, h=self.num_heads)
                key_padding_mask = key_padding_mask.type(dtype)
                kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(
                    kv, key_padding_mask
                )
                kv_unpad = rearrange(kv_unpad, 'nnz (two h d) -> nnz two h d',
                                     two=2, h=self.num_heads)
            else:
                kv_unpad = rearrange(kv, 'b s two h d -> (b s) two h d',
                                     two=2, h=self.num_heads)
                kv_cu_seqlens = torch.arange(
                    0, (b_size + 1) * k.shape[1], step=k.shape[1],
                    dtype=torch.int32, device=q.device
                )
                kv_max_s = k.shape[1]

            # context = flash_attn_unpadded_kvpacked_func(
            context = flash_attn_varlen_kvpacked_func(
                q,
                kv_unpad,
                q_cu_seqlens,
                kv_cu_seqlens,
                s_size,
                kv_max_s,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=1.,  # apply on q above
            )
            context = rearrange(context, '(b s) h d -> b s (h d)',
                                b=b_size, h=self.num_heads)
            return self.out_proj(context), None


class CustomTorchMHASelf(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, use_rotary_emb=None, device=None, dtype=None,
                 lora_qv_rank=None,lora_alpha=1, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.dropout_p = attention_dropout

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        # assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
        assert (self.head_dim % 8 == 0) & (self.head_dim <= 128), 'heads divisible by 8'
        self.scaling = self.head_dim ** -0.5

        self.use_rotary_emb = use_rotary_emb
        if use_rotary_emb:
            self.rot_emb = RotaryEmbeddingESM(self.head_dim)

        if lora_qv_rank is not None:
            self.Wqkv = lora.MergedLinear(embed_dim, 3*embed_dim, r=lora_qv_rank,
                                          lora_alpha=lora_alpha, enable_lora=[True, False, True])
        else:
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        Credit: some elements adopted from OpenFold:
        https://github.com/aqlaboratory/openfold/blob/feed4ae22edf899b37bee49293fff902bdd64e2d/openfold/model/primitives.py#L660
        """
        qkv = self.Wqkv(x)
        dtype = qkv.dtype
        # bsz, num_heads, tgt_len, head_dim
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d',
                            three=3, h=self.num_heads)
        b_size, _, s_size, _ = q.shape

        if self.use_rotary_emb:
            q, k = self.rot_emb(q, k, seq_dimension=-2)
        # scaling happens in scaled_dot_product_attention
        # q = q * self.scaling

        if key_padding_mask is not None:
            key_padding_mask = rearrange(~key_padding_mask, 'b s -> b 1 1 s')

        context = F.scaled_dot_product_attention(q, k, v, attn_mask=key_padding_mask, dropout_p=0.0, is_causal=False)

        context = rearrange(context, 'b h s d -> b s (h d)',
                            b=b_size, h=self.num_heads)
        return self.out_proj(context), None


class TransformerLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        add_bias_kv=True,  # adds bias to the key and value sequences at dim=0
        # use_esm1b_layer_norm=False,
        use_rotary_embeddings: bool = False,
        activation_fn='esm-gelu',
        flash_attention=False,
        torch_attention=False,
        attention_dropout=0.0,  # =0 in ESM2
        attention_scale_d=False,  # MUP scale 1/d instead of 1/sqrt(d)
        deepnorm=False,  # deepnorm instead of pre-norm
        num_layers=None,
        swiglu_bias=False,
        lora_qv_rank=None,  # finetune qv with lora
        lora_qv_alpha=1,
        cross_attn=False,
    ):
        super().__init__()
        if activation_fn == 'esm-gelu':
            self.activation = gelu
        elif activation_fn == 'nn.gelu':
            self.activation = nn.GELU()
        elif activation_fn.startswith('SwiGLU'):  # combined with fc1
            self.activation = None
        else:
            raise ValueError(f'TransformerLayer {activation_fn} not implemented')
        self.flash_attention = flash_attention
        self.torch_attention = torch_attention
        self.cross_attn = cross_attn
        if lora_qv_rank is not None:
            assert lora_is_installed, 'Lora not installed'
            assert self.flash_attention or self.torch_attention, \
                'Lora finetuning only compatible with flash_attention or torch_attention'
        if self.cross_attn:
            assert not add_bias_kv
            
            self.self_attn = FlashMHACrossMaskKV(
                embed_dim,
                attention_heads,
                bias=True,
                batch_first=True,
                attention_dropout=attention_dropout,
                causal=False,
                use_rotary_emb=use_rotary_embeddings,
                device='cuda',
                lora_qv_rank=lora_qv_rank,
                lora_alpha=lora_qv_alpha,
            )
        elif self.flash_attention:
            assert not add_bias_kv
            # rotary_emb = '1d' if use_rotary_embeddings else None
            
            self.self_attn = FlashMHASelfMaskKV(
                embed_dim,
                attention_heads,
                bias=True,
                batch_first=True,
                attention_dropout=attention_dropout,
                causal=False,
                use_rotary_emb=use_rotary_embeddings,
                device='cuda',
                lora_qv_rank=lora_qv_rank,
                # dtype=torch.float16,
            )  # specify fp16 here cast linear.parameters to fp16, incompatable with autocast
            
        elif self.torch_attention:
            assert not attention_scale_d, \
                'Cannot apply attention_scale_d; Q scaling factor is hard coded in torch_attention'
            self.self_attn = CustomTorchMHASelf(
                embed_dim,
                attention_heads,
                bias=True,
                batch_first=True,
                attention_dropout=attention_dropout,
                causal=False,
                use_rotary_emb=use_rotary_embeddings,
                device='cuda',
                lora_qv_rank=lora_qv_rank,
                lora_alpha=lora_qv_alpha,
            )
        else:
            # Have to use ESM instead of pytorch because of rotary_emb
            self.self_attn = MultiheadAttention(
                embed_dim,
                attention_heads,
                dropout=attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=False,
                use_rotary_embeddings=use_rotary_embeddings,
            )

        self.deepnorm = deepnorm
        if self.deepnorm:
            assert num_layers is not None, 'deepnorm needs num_layers'
            self.num_layers = num_layers
            self.deepnorm_alpha = (2.0 * self.num_layers)**0.25
        if attention_scale_d:
            assert hasattr(self.self_attn, 'scaling'), 'attention method have no scaling attrib'
            self.self_attn.scaling = 1 / self.self_attn.head_dim

        # Pytorch LayerNorm is better than apex.FusedLayerNorm for most GPUs
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        if activation_fn == 'SwiGLU':
            self.gfc1 = SwiGLU(embed_dim, ffn_embed_dim, bias=swiglu_bias)
        elif activation_fn == 'SwiGLUB':
            self.gfc1 = SwiGLUB(embed_dim, ffn_embed_dim, bias=swiglu_bias)
        else:
            self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x, self_attn_mask=None,
        self_attn_padding_mask=None,
        need_head_weights=False,
        need_weights=False,
    ):
        residual = x
        if not self.deepnorm:
            x = self.self_attn_layer_norm(x)
        if self.flash_attention:
            assert not need_head_weights  # this may be the default, TODO: test
            assert self_attn_mask is None
            assert not need_weights, f"No attn weights returned in flash_attention"
            x = x.transpose(0, 1)
            
            if self_attn_padding_mask is not None:
                self_attn_padding_mask = ~self_attn_padding_mask
                # self_attn_padding_mask = torch.ones(
                #     self_attn_padding_mask.shape, device='cuda'
                # ) * ~self_attn_padding_mask
            x, attn = self.self_attn(
                x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                # attn_mask=self_attn_mask,
            )

            x = x.transpose(0, 1)
        elif self.torch_attention:
            assert not need_head_weights  # this may be the default, TODO: test
            assert self_attn_mask is None
            assert not need_weights, f"No attn weights returned in flash_attention"
            x = x.transpose(0, 1)
            x, attn = self.self_attn(
                x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                # attn_mask=self_attn_mask,
            )
            x = x.transpose(0, 1)
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                need_head_weights=need_head_weights,
                attn_mask=self_attn_mask,
            )
        if self.deepnorm:
            x = self.self_attn_layer_norm(
                residual*self.deepnorm_alpha + x)
        else:
            x = residual + x

        residual = x
        if not self.deepnorm:
            x = self.final_layer_norm(x)
        # x = gelu(self.fc1(x))
        if self.activation is None:
            x = self.gfc1(x)
        else:
            x = self.activation(self.fc1(x))
        x = self.fc2(x)
        if self.deepnorm:
            x = self.final_layer_norm(
                residual*self.deepnorm_alpha + x)
        else:
            x = residual + x

        return x, attn

