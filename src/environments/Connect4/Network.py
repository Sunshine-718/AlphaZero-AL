#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import LinearLR, SequentialLR
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
    def __init__(self, lr, in_dim=3, h_dim=64, out_dim=7, dropout=0.2, device='cpu', num_res_blocks=3, policy_lr_scale=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.n_actions = out_dim

        # Body: Stem + Residual Blocks
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim, h_dim, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.SiLU(inplace=True),
            *[ResidualBlock(h_dim, h_dim, dropout_rate=dropout) for _ in range(num_res_blocks)]
        )

        # Dual Policy Heads (one per player)
        self.policy_head_1 = nn.Sequential(
            nn.Conv2d(h_dim, 2, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.LayerNorm(2 * 8 * 9),
            nn.Linear(2 * 8 * 9, out_dim),
            nn.LogSoftmax(dim=-1)
        )
        self.policy_head_2 = deepcopy(self.policy_head_1)
        self.value_head = nn.Sequential(
            nn.Conv2d(h_dim, 2, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 9, 2 * 8 * 9),
            nn.SiLU(inplace=True),
            nn.LayerNorm(2 * 8 * 9),
            nn.Linear(2 * 8 * 9, 3),
            nn.LogSoftmax(dim=-1)
        )
        self.steps_head = nn.Sequential(
            nn.Conv2d(h_dim, 2, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Flatten(),
            nn.LayerNorm(2 * 8 * 9),
            nn.Linear(2 * 8 * 9, 43),
            nn.LogSoftmax(dim=-1)
        )
        
        self.apply(self.init_weights)
        nn.init.constant_(self.policy_head_1[-2].weight, 0)
        nn.init.constant_(self.policy_head_2[-2].weight, 0)
        nn.init.constant_(self.value_head[-2].weight, 0)
        nn.init.constant_(self.steps_head[-2].weight, 0)
        
        self.opt = torch.optim.SGD([
            {'params': self.hidden.parameters()},
            {'params': self.value_head.parameters()},
            {'params': self.steps_head.parameters()},
            {'params': self.policy_head_1.parameters(), 'lr': lr * policy_lr_scale},
            {'params': self.policy_head_2.parameters(), 'lr': lr * policy_lr_scale},
        ], lr=lr, momentum=0.9, weight_decay=1e-4)
        
        scheduler_warmup = LinearLR(self.opt, start_factor=0.001, total_iters=100)
        scheduler_train = LinearLR(self.opt, start_factor=1, end_factor=0.01, total_iters=1000)
        self.scheduler = SequentialLR(self.opt, schedulers=[scheduler_warmup, scheduler_train], milestones=[100])
        self.to(self.device)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def name(self):
        return 'CNN'

    def _route_policy(self, hidden, player):
        """Route each sample to its player's policy head. No redundant computation."""
        idx1 = (player > 0).nonzero(as_tuple=True)[0]
        idx2 = (player < 0).nonzero(as_tuple=True)[0]
        log_prob = hidden.new_zeros(hidden.size(0), self.n_actions)
        if idx1.numel() > 0:
            log_prob[idx1] = self.policy_head_1(hidden[idx1])
        if idx2.numel() > 0:
            log_prob[idx2] = self.policy_head_2(hidden[idx2])
        return log_prob

    def forward(self, x):
        hidden = self.hidden(x)
        log_prob = self._route_policy(hidden, x[:, -1, 0, 0])
        value = self.value_head(hidden)
        steps_pred = self.steps_head(hidden)
        return log_prob, value, steps_pred

    @torch.no_grad()
    def policy(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        hidden = self.hidden(state)
        return self._route_policy(hidden, state[:, -1, 0, 0]).exp().cpu().numpy()

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
        log_prob, value_log_prob, log_steps = self.forward(t)
        # Value head outputs: [P(draw), P(p1_win), P(p2_win)] — 绝对视角，直接传递
        wdl = value_log_prob.exp()  # (batch, 3)

        steps_prob = log_steps.exp()
        idx = torch.arange(43, dtype=torch.float32, device=self.device)
        expected_steps = (steps_prob * idx).sum(dim=1)

        return (log_prob.exp().cpu().numpy(),
                wdl.cpu().numpy(),
                expected_steps.cpu().view(-1, 1).numpy())
