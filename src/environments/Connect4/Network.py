#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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


class ConvGLU(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.gate = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.proj = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.proj(x)


class LinearGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.proj(x)

class ResidualGLUConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.norm = RMSNorm2d(in_dim, 1e-5)
        self.glu = ConvGLU(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.residual = in_dim == out_dim

    def forward(self, x):
        residual = 0
        if self.residual:
            residual = x
        x = self.norm(x)
        x = self.glu(x)
        x = self.conv(x)
        return self.dropout(x + residual)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., zero_out=False):
        super().__init__()
        self.glu = LinearGLU(in_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim)
        self.norm = nn.RMSNorm(in_dim, 1e-5)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.residual = in_dim == out_dim
        if zero_out:
            nn.init.constant_(self.linear.weight, 0)

    def forward(self, x):
        residual = 0
        if self.residual:
            residual = x
        x = self.norm(x)
        x = self.glu(x)
        x = self.linear(x)
        return self.dropout(x + residual)


class CNN(Base):
    def __init__(self, lr, in_dim=3, h_dim=32, out_dim=7, dropout=0.05, device='cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = nn.Sequential(ResidualGLUConv2d(in_dim, h_dim, dropout),
                                    ResidualGLUConv2d(h_dim, h_dim, dropout),
                                    ResidualGLUConv2d(h_dim, h_dim, dropout))
        self.policy_head = nn.Sequential(ResidualGLUConv2d(h_dim, 3, dropout),
                                         nn.Flatten(),
                                         ResidualBlock(6 * 7 * 3, 7, dropout=0.),
                                         nn.LogSoftmax(dim=-1))
        self.value_head = nn.Sequential(ResidualGLUConv2d(h_dim, 3, dropout),
                                        nn.Flatten(),
                                        ResidualBlock(6 * 7 * 3, 6 * 7 * 3, dropout=dropout),
                                        ResidualBlock(6 * 7 * 3, 3, dropout=0.),
                                        nn.LogSoftmax(dim=-1))
        self.device = device
        self.n_actions = out_dim
        self.apply(self.init_weights)
        nn.init.constant_(self.policy_head[-2].linear.weight, 0)
        nn.init.constant_(self.value_head[-2].linear.weight, 0)
        self.opt = self.configure_optimizers(lr, 0.01)
        scheduler_warmup = LinearLR(self.opt, start_factor=0.01, total_iters=10)
        scheduler_cosine = CosineAnnealingLR(self.opt, T_max=200, eta_min=lr * 0.1)
        self.scheduler = SequentialLR(self.opt, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[10])
        self.to(self.device)
    
    def configure_optimizers(self, lr, weight_decay):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for np, p in param_dict.items() if (not np.endswith('.bias')) and ('norm' not in np)]
        nodecay_params = [p for np, p in param_dict.items() if (np.endswith('.bias') or 'norm' in np)]
        optim_grops = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}
        ]
        return NAdam(optim_grops, lr=lr, decoupled_weight_decay=True)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def name(self):
        return 'CNN'

    def forward(self, x):
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden)
        value = self.value_head(hidden)
        return log_prob, value

    @torch.no_grad()
    def policy(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        hidden = self.hidden(state)
        return self.policy_head(hidden).exp().cpu().numpy()

    @torch.no_grad()
    def value(self, state):
        hidden = self.hidden(state)
        return self.value_head(hidden).exp().cpu().numpy()

    @torch.no_grad()
    def predict(self, state):
        t = torch.from_numpy(state)
        if self.device != 'cpu':
            t = t.pin_memory().to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            t = t.float()
        player = t[:, -1, 0, 0].view(-1)
        log_prob, value_log_prob = self.forward(t)
        value_prob = value_log_prob.exp()
        value = player * (value_prob[:, 1] - value_prob[:, 2])
        return log_prob.exp().cpu().numpy(), value.cpu().view(-1, 1).numpy()
