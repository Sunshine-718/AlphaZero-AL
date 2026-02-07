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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from einops import rearrange
from ..NetworkBase import Base


class RMSNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x):
        if x.dtype != self.norm.weight.dtype:
            self.norm.to(dtype=x.dtype)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", mask)
        self.norm1 = RMSNorm2d(args[1])
        self.o_proj = nn.Sequential(nn.Conv2d(args[1], args[1], kernel_size=1),
                                    RMSNorm2d(args[1]))

    def forward(self, x):
        gate_proj = F.silu(F.conv2d(x, self.weight * self.mask, self.bias,
                                  self.stride, self.padding, self.dilation, self.groups))
        v_proj = F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        hidden = self.norm1(gate_proj * v_proj)
        return self.o_proj(hidden)


def mask(out_ch, in_ch, device='cpu'):
    m = torch.tensor([[1, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 1, 0, 1, 0],
                      [0, 0, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 0, 0],
                      [0, 1, 0, 1, 0, 1, 0],
                      [1, 0, 0, 1, 0, 0, 1]], dtype=torch.float32, device=device)
    return m[None, None, :, :].expand(out_ch, in_ch, m.shape[0], m.shape[1]).clone()


class ResidualSwishGLUConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.gate_proj = nn.Sequential(nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, padding=0),
                                       nn.SiLU(True))
        self.v_proj = nn.Sequential(nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, padding=0))
        self.o_proj = nn.Sequential(nn.Conv2d(in_dim // 2, out_dim, kernel_size=1, padding=0),
                                    RMSNorm2d(out_dim),
                                    nn.Dropout(dropout))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = RMSNorm2d(in_dim // 2)

    def forward(self, x):
        if self.in_dim == self.out_dim:
            return x + self.o_proj(self.norm(self.gate_proj(x) * self.v_proj(x)))
        return self.o_proj(self.norm(self.gate_proj(x) * self.v_proj(x)))


class ResidualSwiGLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super().__init__()
        self.gate_proj = nn.Sequential(nn.Linear(in_dim, out_dim, bias=True),
                                       nn.Sigmoid())
        self.v_proj = nn.Sequential(nn.Linear(in_dim, out_dim, bias=True))
        self.o_proj = nn.Sequential(nn.Linear(out_dim, out_dim, bias=True),
                                    nn.RMSNorm(out_dim, eps=1e-5),
                                    nn.Dropout(dropout))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = nn.RMSNorm(out_dim)

    def forward(self, x):
        if self.in_dim == self.out_dim:
            return F.silu(x + self.o_proj(self.norm(self.gate_proj(x) * self.v_proj(x))))
        return F.silu(self.o_proj(self.norm(self.gate_proj(x) * self.v_proj(x))))


class CNN(Base):
    def __init__(self, lr, in_dim=3, h_dim=128, out_dim=7, dropout=0.05, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(MaskedConv2d(in_dim, h_dim, mask=mask(h_dim, in_dim, device), kernel_size=7, padding=3, bias=False),
                                    nn.Dropout2d(dropout),
                                    ResidualSwishGLUConv2d(h_dim, h_dim, dropout),
                                    ResidualSwishGLUConv2d(h_dim, h_dim, dropout),
                                    ResidualSwishGLUConv2d(h_dim, h_dim, dropout))
        self.policy_head = nn.Sequential(nn.Conv2d(h_dim, h_dim // 2, kernel_size=(6, 1), bias=False),
                                         nn.BatchNorm2d(h_dim // 2),
                                         nn.Dropout2d(dropout),
                                         nn.SiLU(True),
                                         nn.Flatten(),
                                         nn.Linear(7 * (h_dim // 2), 7),
                                         nn.LogSoftmax(dim=-1))
        self.value_head = nn.Sequential(nn.Conv2d(h_dim, 1, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                        nn.BatchNorm2d(1),
                                        nn.SiLU(True),
                                        nn.Flatten(),
                                        ResidualSwiGLUBlock(6 * 7, 6 * 7, dropout),
                                        nn.Linear(6 * 7, 3),
                                        nn.LogSoftmax(dim=-1))
        nn.init.constant_(self.policy_head[-2].weight, 0)
        self.device = device
        self.n_actions = out_dim
        self.apply(self.init_weights)
        self.opt = self.configure_optimizers(lr, 1e-4)
        # self.opt = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler_warmup = LinearLR(self.opt, start_factor=1e-8, total_iters=200)
        scheduler_cosine = CosineAnnealingLR(self.opt, T_max=1000, eta_min=lr * 0.1)
        self.scheduler = SequentialLR(self.opt, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[200])
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
        return NAdam(optim_grops, lr=lr, decoupled_weight_decay=True)

    def name(self):
        return 'CNN'

    def forward(self, x):
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden)
        value = self.value_head(hidden)
        return log_prob, value

    @torch.no_grad()
    def policy(self, state):
        self.eval()
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        hidden = self.hidden(state)
        return self.policy_head(hidden).exp().cpu().numpy()

    @torch.no_grad()
    def value(self, state):
        self.eval()
        hidden = self.hidden(state)
        return self.value_head(hidden).exp().cpu().numpy()

    @torch.no_grad()
    def predict(self, state, draw_factor=0.5):
        state = torch.from_numpy(state).float().to(self.device)
        self.eval()
        log_prob, value_log_prob = self.forward(state)
        value_prob = value_log_prob.exp()
        player = state[:, -1, 0, 0].view(-1,)
        value = (-player) * draw_factor * value_prob[:, 0] + value_prob[:, 1] - value_prob[:, 2]
        return log_prob.exp().cpu().numpy(), value.cpu().view(-1, 1).numpy()
