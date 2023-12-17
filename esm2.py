# Adapted from https://github.com/facebookresearch/esm/blob/4e0ebb7a7b875ef40178cbb11e830eb5859b4180/esm/model/esm2.py

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified to add parameter initialization and Flash Attention
# You can extend ESM2 with forward() calling esm_forward()
# For self-attention where qkv are all from x, packing linear layer
#  is done for FlashAttention.To load weights from ESM2,
#  use upgrade_state_dict_qkv_to_packed()

from typing import Union
import torch
import torch.nn as nn
import esm
from transformer_modules import TransformerLayer
import re
import warnings
from typing import Optional
import importlib


class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
        flash_attention: bool = False,
        torch_attention: bool = False, # torch experimental, incompatible with flash_attention
        scale_residual_proj: bool = True,  # scale weights of proj layers goin into residual
        init_scale: float = 1.0,  # scale weight after std, help with fp16
        init_encoder_var: float = 0.02,  # scale normal std, 0.02 used by ESM-1b
        mup_init: bool = False,
        attention_scale_d: bool = False,  # mup: attention 1/d instead of 1/sqrt(d)
        deepnorm: bool = False,
        deepnet_init: bool = False,
        tf_activation_fn: str = 'esm-gelu',  # esm uses gelu differ than nn.GELU
        ffn_embed_dim: Optional[int] = None,  # default 4 * self.embed_dim 
        lora_qv_rank: Optional[int] = None,  # Finetune with lora (https://github.com/microsoft/LoRA)
        lora_qv_alpha: int = 1,  # default 1, can tune lr instead of alpha using Adam opt
        mask_ratio_train: float = 0.15 * 0.8,
        cls_eos_tensor: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.token_dropout = token_dropout
        self.embed_scale = 1
        self.mask_ratio_train = mask_ratio_train
        self.scale_residual_proj = scale_residual_proj
        self.init_scale = init_scale
        self.init_norm_std = init_encoder_var
        self.mup_init = mup_init
        self.deepnet_init = deepnet_init
        if self.deepnet_init:
            print('Warning, deepnet_init replaces _init_residual_proj of ESM2')
            print('Warning, deepnet_init v_proj might be redundant with mup')
        if cls_eos_tensor:
            # self.cls_eos_idx for generating mask
            self.register_buffer("cls_eos_idx",
                                 torch.Tensor([self.alphabet.cls_idx,
                                               self.alphabet.eos_idx]))

        if self.mup_init:
            self.init_norm_std = (init_encoder_var / self.embed_dim)**0.5
        
        self.embed_tokens = nn.Embedding(
            len(self.alphabet),
            self.embed_dim,
            padding_idx=alphabet.padding_idx,
        )
        self.flash_attention = flash_attention
        self.torch_attention = torch_attention
        assert flash_attention + torch_attention < 2, \
            ' cannot use flash and torch attention together'
        self.deepnorm = deepnorm
        if attention_scale_d & (not mup_init):
            warnings.warn('attention_scale_d was suggested by mup,'+
                          ' you probably meant to use with mup_init=True')
        self.ffn_embed_dim = ffn_embed_dim
        if self.ffn_embed_dim is None:
            self.ffn_embed_dim = 4 * self.embed_dim  # ESM2 specific
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    self.ffn_embed_dim,  # above
                    self.attention_heads,
                    add_bias_kv=False,
                    # use_esm1b_layer_norm=True,  # better to just use pytorch layernorm
                    use_rotary_embeddings=True,
                    flash_attention=flash_attention,
                    torch_attention=torch_attention,
                    attention_scale_d=attention_scale_d,
                    deepnorm=self.deepnorm,
                    num_layers=self.num_layers,
                    activation_fn=tf_activation_fn,
                    lora_qv_rank=lora_qv_rank,
                    lora_qv_alpha=lora_qv_alpha,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = esm.modules.ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.alphabet.prepend_bos,
            self.alphabet.append_eos,
            eos_idx=self.alphabet.eos_idx,
        )
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

        self.lm_head = esm.modules.RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=len(self.alphabet),
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[],
                need_head_weights=False, return_contacts=False,
                skip_lm_head=False, skip_last_norm=False):

        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.alphabet.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            '''
            https://github.com/facebookresearch/esm/issues/267
            fully zero out masked token emb during training
            rescale just like dropout during inference
            '''
            x.masked_fill_((tokens == self.alphabet.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            # mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.alphabet.mask_idx
                                   ).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - self.mask_ratio_train
                     ) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        if not skip_last_norm:  # if extend transformer, may want unnormalized (with skip_lm_head)
            x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x

        if skip_lm_head:  # return last layer representation only
            return x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def init_weights(self):
        ''' Initialization of weights
        std is hyperparam now controlled by self.init_norm_std = init_encoder_var
        ESM default: normal init with std=0.02, then scale proj layers going into residual
            Assuming they want to scale for model depth, and residual_layer = out_proj and fc2
            count residual layers as 2*transformerlayers
            Needs validation; Others have asked, both ESM and GPT2, no clear answer
            They may mean out_proj only, but fc2 comes after layernorm and adds directly 
            to residual, then becomes the next residual
        mup: initialize with std = (init_encoder_var / self.embed_dim)**0.5
            zero q_proj
        deepnet: For encoder only 
            'self_attn.out_proj', '.fc1', '.fc2', '.v_proj' scale * (8*self.num_layers)**-0.25
        '''
        self.apply(self._init_weights)
        for name, module in self.named_modules():
            if self.mup_init:
                'zero initializing query head q_proj'
                if name.endswith('Wqkv'):
                    torch.nn.init.constant_(module.weight[:self.embed_dim], 0.)
                if name.endswith('q_proj'):
                    torch.nn.init.constant_(module.weight, 0.)
            if self.deepnet_init:
                assert not self.scale_residual_proj, 'deepnorm and scale_residual_proj redundant'
                if name.endswith(('self_attn.out_proj', '.fc1', '.fc2', '.v_proj')):
                    torch.nn.init.normal_(
                        module.weight,
                        mean=0,
                        std=self.init_norm_std * (8*self.num_layers)**-0.25
                    )
                if name.endswith('.gfc1'):
                    'Gate (first half) is not scaled'
                    torch.nn.init.normal_(
                        module.linear.weight[:,self.ffn_embed_dim:],
                        mean=0,
                        std=self.init_norm_std * (8*self.num_layers)**-0.25
                    )
                if name.endswith('Wqkv'):
                    torch.nn.init.normal_(
                        module.weight[self.embed_dim*2:],
                        mean=0,
                        std=self.init_norm_std * (8*self.num_layers)**-0.25
                    )
            elif self.scale_residual_proj:
                if name.endswith(('self_attn.out_proj', '.fc2')):
                    # nn.init.constant_(module.weight, 0)
                    self._init_residual_proj(module)

        if self.init_scale != 0:
            # Fairseq option to scale initial parameters when using FP16
            with torch.no_grad():
                for param in self.parameters():
                    param.data.mul_(self.init_scale)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.init_norm_std)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_residual_proj(self, module):
        # scale standard initialization by 1/sqrt(number of residual layers)
        nn.init.normal_(
            module.weight, mean=0, std=self.init_norm_std / ((self.num_layers*2) ** 0.5)
        )


    def upgrade_state_dict(self, state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    def upgrade_state_dict_qkv_to_packed(self, state_dict):
        '''Load weights from ESM2 by packing QKV parameters'''
        for layer in range(self.num_layers):
            for wb in ['weight', 'bias']:
                params, param_names = [], []
                for qkv in ['q_proj', 'k_proj', 'v_proj']:
                    param_name = 'layers.' + str(layer) + (
                        '.self_attn.' + qkv + '.' + wb)
                    params.append(state_dict[param_name])
                    param_names.append(param_name)
                packed_name = 'layers.' + str(layer) + '.self_attn.Wqkv.' + wb
                state_dict[packed_name] = torch.cat(params, dim=0)
                for name in param_names:
                    del state_dict[name]
        return state_dict

    def downgrade_state_dict_qkv_to_unpacked(self, state_dict):
        for layer in range(self.num_layers):
            for wb in ['weight', 'bias']:
                packed_name = 'layers.' + str(layer) + '.self_attn.Wqkv.' + wb
                qkv = ['q_proj', 'k_proj', 'v_proj']
                for i in range(3):
                    param_name = 'layers.' + str(layer) + (
                        '.self_attn.' + qkv[i] + '.' + wb)
                    state_dict[param_name] = state_dict[packed_name][
                        self.embed_dim*i: self.embed_dim*(1+i), :]
                del state_dict[packed_name]
        return state_dict

    def rename_rot_to_rotary(self, state_dict, reverse=False):
        raise RuntimeWarning('rename_rot_to_rotary no longer necessary')
        # name1 = 'self_attn.rot_emb'
        # name2 = 'self_attn.rotary_emb'
        # if reverse:
        #     state_dict = self.rename_state_dict(state_dict, name2, name1)
        # else:
        #     state_dict = self.rename_state_dict(state_dict, name1, name2)
        return state_dict

    def rename_state_dict(self, state_dict, old_str, new_str):
        '''rename any matching parameters, careful'''
        pattern = re.compile(old_str)
        state_dict = {pattern.sub(new_str, name): param for name, param in state_dict.items()}
        return state_dict



                