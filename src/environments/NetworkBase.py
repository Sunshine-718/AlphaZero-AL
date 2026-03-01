#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:20
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC
from copy import deepcopy
from sklearn.metrics import f1_score


class Base(ABC, nn.Module):
    def save(self, path=None):
        while True:
            try:
                if path is not None:
                    checkpoint = {'model_state_dict': self.state_dict(),
                                  'opt_state_dict': self.opt.state_dict(),
                                  'scheduler_state_dict': self.scheduler.state_dict()}
                    torch.save(checkpoint, path)
                    break
            except RuntimeError:
                print('Failed to save parameters, retry...')
                time.sleep(1)
                continue

    def load(self, path=None):
        if path is not None:
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                self.load_state_dict(checkpoint['model_state_dict'])
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f'Failed to load parameters.\n{e}')
        return self

    def train_step(self, dataloader, augment, ddp_model=None, n_epochs=10,
                   q_ratio=0.0, value_decay=1.0):
        model = ddp_model if ddp_model is not None else self
        p_l, v_l, s_l = [], [], []
        use_soft = value_decay < 1.0 or q_ratio > 0
        for _ in range(n_epochs):
            self.train()
            for batch in dataloader:
                state, prob, winner, steps_to_end, root_wdl = augment(batch)
                # Value target: map winner {0,1,-1} → class {0=draw, 1=win, 2=loss}
                value_class = deepcopy(winner)
                value_class[value_class == -1] = 2
                value_class = value_class.view(-1,).long()
                steps_target = steps_to_end.view(-1,).long()
                mask = (steps_target != 0).float()
                self.opt.zero_grad()
                log_p_pred, value_pred, steps_pred = model(state)
                steps_target = steps_target.clamp(0, steps_pred.shape[-1] - 1)

                if use_soft:
                    z_target = F.one_hot(value_class, 3).float()

                    # Game-length discount: γ^steps × one_hot + (1-γ^steps) × uniform
                    if value_decay < 1.0:
                        discount = (value_decay ** steps_to_end.float().view(-1)).unsqueeze(1)
                        z_target = discount * z_target + (1 - discount) * (1.0 / 3.0)

                    # Q-ratio: blend with root WDL
                    if q_ratio > 0:
                        has_q = (root_wdl.sum(dim=1, keepdim=True) > 0).float()
                        eff_q_ratio = q_ratio * has_q
                        z_target = eff_q_ratio * root_wdl + (1 - eff_q_ratio) * z_target

                    v_loss = -torch.sum(z_target * value_pred, dim=1).mean()
                else:
                    v_loss = F.nll_loss(value_pred, value_class)

                per_sample_p = -torch.sum(prob * log_p_pred, dim=1)
                p_loss = torch.mean(per_sample_p * mask)
                s_loss = F.nll_loss(steps_pred, steps_target)
                loss = p_loss + v_loss + 0.5 * s_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.opt.step()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
                s_l.append(s_loss.item())
        self.eval()

        self.scheduler.step()
        with torch.no_grad():
            _, new_v, _ = self(state)
        f1 = f1_score(value_class.cpu().numpy(), torch.argmax(new_v, dim=-1).cpu().numpy(), average='macro')
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(log_p_pred.exp() * log_p_pred, dim=-1))
            total_norm = 0
            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return np.mean(p_l), np.mean(v_l), np.mean(s_l), float(entropy), total_norm, f1
