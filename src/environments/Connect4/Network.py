#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from ..NetworkBase import Base


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with BatchNorm, SiLU activation, and Dropout.
    Conv -> BN -> SiLU -> Dropout -> Conv -> BN -> Add -> SiLU
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.silu(out)


class CNN(Base):
    def __init__(self, lr, in_dim=3, h_dim=128, out_dim=7, dropout=0.1, device='cpu', num_res_blocks=2):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.n_actions = out_dim

        # Body: Stem + Residual Blocks
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(inplace=True),
            *[ResidualBlock(h_dim, h_dim, dropout_rate=dropout) for _ in range(num_res_blocks)]
        )

        # Heads
        self.policy_head = nn.Sequential(
            nn.Conv2d(h_dim, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3 * 6 * 7, out_dim),
            nn.LogSoftmax(dim=-1)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(h_dim, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3 * 6 * 7, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 3),
            nn.LogSoftmax(dim=-1)
        )
        self.steps_head = nn.Sequential(
            nn.Conv2d(h_dim, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3 * 6 * 7, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 43),
            nn.LogSoftmax(dim=-1)
        )
        
        self.apply(self.init_weights)
        nn.init.constant_(self.policy_head[-2].weight, 0)
        nn.init.constant_(self.value_head[-2].weight, 0)
        nn.init.constant_(self.steps_head[-2].weight, 0)
        
        self.opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        
        scheduler_warmup = LinearLR(self.opt, start_factor=0.01, total_iters=100)
        # scheduler_train = LinearLR(self.opt, start_factor=1, end_factor=0.1, total_iters=1000)
        # self.scheduler = SequentialLR(self.opt, schedulers=[scheduler_warmup, scheduler_train], milestones=[100])
        self.scheduler = scheduler_warmup
        self.to(self.device)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def name(self):
        return 'CNN'

    def forward(self, x):
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden)
        value = self.value_head(hidden)
        steps_pred = self.steps_head(hidden)
        return log_prob, value, steps_pred

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
        log_prob, value_log_prob, _ = self.forward(t)
        value_prob = value_log_prob.exp()
        value = player * (value_prob[:, 1] - value_prob[:, 2])
        return log_prob.exp().cpu().numpy(), value.cpu().view(-1, 1).numpy()
