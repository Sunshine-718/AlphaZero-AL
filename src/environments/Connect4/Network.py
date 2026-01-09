#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam, SGD, AdamW
from einops import rearrange
from ..NetworkBase import Base


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)


def mask(out_ch, in_ch, device='cpu'):
    m = torch.tensor([[1, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 1, 0, 1, 0],
                      [0, 0, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0],
                      [1, 0, 0, 1, 0, 0, 1]], dtype=torch.float32, device=device)
    return m[None, None, :, :].expand(out_ch, in_ch, m.shape[0], m.shape[1]).clone()


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_dim),
                                  nn.SiLU(True),
                                  nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0, bias=False))
        self.dropout = nn.Dropout2d(dropout)
        self.norm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        return self.dropout(F.silu(self.norm(x + self.conv(x))))


class CNN(Base):
    def __init__(self, lr, in_dim=3, h_dim=128, out_dim=7, dropout=0.2, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(MaskedConv2d(in_dim, h_dim, mask=mask(h_dim, in_dim, device), kernel_size=7, padding=3, bias=False),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(True),
                                    nn.Dropout2d(dropout),
                                    Block(h_dim, h_dim, dropout),
                                    Block(h_dim, h_dim, dropout),
                                    Block(h_dim, h_dim, dropout))
        self.policy_middle = nn.Sequential(nn.Conv2d(h_dim, h_dim, kernel_size=(6, 1), bias=False),
                                           nn.BatchNorm2d(h_dim),
                                           nn.SiLU(True))
        self.policy_head = nn.Sequential(nn.Conv2d(h_dim, 1, kernel_size=1),
                                         nn.Flatten(),
                                         nn.LogSoftmax(dim=-1))
        self.value_head = nn.Sequential(nn.Conv2d(h_dim, 1, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                        nn.BatchNorm2d(1),
                                        nn.SiLU(True),
                                        nn.Flatten(),
                                        nn.Linear(6 * 7, 6 * 7),
                                        nn.LayerNorm(6 * 7),
                                        nn.SiLU(True),
                                        nn.Linear(6 * 7, 3),
                                        nn.LogSoftmax(dim=-1))
        self.positional_encoding = torch.zeros(1, h_dim, 1, 7)
        torch.nn.init.normal_(self.positional_encoding, 0, 0.02)
        torch.nn.init.constant_(self.policy_head[0].weight, 0)
        self.positional_encoding[:, :, :, -1] = torch.clone(self.positional_encoding[:, :, :, 0])
        self.positional_encoding[:, :, :, -2] = torch.clone(self.positional_encoding[:, :, :, 1])
        self.positional_encoding[:, :, :, -3] = torch.clone(self.positional_encoding[:, :, :, 2])
        self.positional_encoding = nn.Parameter(self.positional_encoding)
        self.device = device
        self.n_actions = out_dim
        self.apply(self.init_weights)
        self.opt = self.configure_optimizers(lr, 1e-4)
        # self.opt = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        self.to(self.device)
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self, lr, weight_decay):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for np, p in param_dict.items() if (not np.endswith('.bias')) and ('norm' not in np)]
        nodecay_params = [p for np, p in param_dict.items() if (np.endswith('.bias') or 'norm' in np)]
        optim_grops = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}
        ]
        # fused_available = 'fused' in inspect.signature(AdamW).parameters
        # use_fused = fused_available and device == 'cuda'
        # extra_args = dict(fused=True) if use_fused else dict()
        return NAdam(optim_grops, lr=lr, decoupled_weight_decay=True)

    def name(self):
        return 'CNN'

    def forward(self, x):
        hidden = self.hidden(x)
        p_laten = self.policy_middle(hidden)
        log_prob = self.policy_head(p_laten + self.positional_encoding)
        value = self.value_head(hidden)
        return log_prob, value

    def policy(self, state):
        self.eval()
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            hidden = self.hidden(state)
            p_laten = self.policy_middle(hidden)
            return self.policy_head(p_laten + self.positional_encoding).exp().cpu().numpy()

    def value(self, state):
        self.eval()
        with torch.no_grad():
            hidden = self.hidden(state)
            return self.value_head(hidden).exp().cpu().numpy()

    def predict(self, state, draw_factor=0.5):
        state = torch.from_numpy(state).float().to(self.device)
        self.eval()
        with torch.no_grad():
            log_prob, value_log_prob = self.forward(state)
            value_prob = value_log_prob.exp()
            player = state[:, -1, 0, 0].view(-1,)
            value = (-player) * draw_factor * value_prob[:, 0] + value_prob[:, 1] - value_prob[:, 2]
        return log_prob.exp().cpu().numpy(), value.cpu().view(-1, 1).numpy()


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1))
        num_patches = 6 * 7
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim))
        self.d_model = embed_dim

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'n c h w -> n (h w) c')
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) * math.sqrt(self.d_model)
        x += self.pos_embed
        return x


class ViT(Base):
    def __init__(self, lr, in_channels=3, embed_dim=128, num_action=7, depth=4, num_heads=8, dropout=0.1, device='cpu'):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, activation=nn.SiLU(True), norm_first=False, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)
        self.policy_head = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                         nn.LayerNorm(embed_dim),
                                         nn.SiLU(True),
                                         nn.Linear(embed_dim, num_action))
        self.value_head = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                        nn.LayerNorm(embed_dim),
                                        nn.SiLU(True),
                                        nn.Linear(embed_dim, 1))
        self.n_actions = num_action
        self.device = device
        self.to(device)
        self.opt = NAdam(self.parameters(), lr=lr, weight_decay=1e-4, decoupled_weight_decay=True)

    def name(self):
        return 'ViT'

    def forward(self, x, mask=None):
        cls_token = self.get_cls_token(x)
        prob_logit = self.policy_head(cls_token)
        if mask is not None:
            prob_logit.masked_fill_(~mask, -float('inf'))
        log_prob = F.log_softmax(prob_logit, dim=-1)
        value_logit = self.value_head(cls_token)
        return log_prob, value_logit

    def get_cls_token(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        return x[:, 0, :]
