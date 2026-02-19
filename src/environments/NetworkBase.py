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
                checkpoint = torch.load(path, map_location=self.device)
                self.load_state_dict(checkpoint['model_state_dict'])
                self.opt.load_state_dict(checkpoint['opt_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f'Failed to load parameters.\n{e}')
        return self

    def train_step(self, dataloader, augment, ddp_model=None):
        model = ddp_model if ddp_model is not None else self
        p_l, v_l, s_l, const_loss = [], [], [], []
        for _ in range(10):
            self.train()
            for batch in dataloader:
                state, prob, winner, steps_to_end = augment(batch)
                value = deepcopy(winner)
                value[value == -1] = 2
                value = value.view(-1,).long()
                steps_target = steps_to_end.view(-1).long()
                self.opt.zero_grad()
                log_p_pred, value_pred, steps_pred = model(state)
                with torch.no_grad():
                    total_const_loss = 0
                    batch_split = state.shape[0] // 2

                    log_p_flipped = torch.flip(log_p_pred[:batch_split], dims=[1])
                    log_p_target = log_p_pred[batch_split:]

                    policy_const_loss = F.kl_div(log_p_flipped, log_p_target, reduction='batchmean', log_target=True)
                    total_const_loss += policy_const_loss
                    value_const_loss = F.kl_div(value_pred[:batch_split],
                                                value_pred[batch_split:], reduction='batchmean', log_target=True)
                    total_const_loss += value_const_loss
                v_loss = F.nll_loss(value_pred, value)
                p_loss = -torch.mean(torch.sum(prob * log_p_pred, dim=1))
                s_loss = F.nll_loss(steps_pred, steps_target)
                loss = p_loss + 1.5 * v_loss + 0.5 * s_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.opt.step()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
                s_l.append(s_loss.item())
                const_loss.append(total_const_loss.item())
        self.eval()

        self.scheduler.step()
        with torch.no_grad():
            _, new_v, _ = self(state)
        f1 = f1_score(value.cpu().numpy(), torch.argmax(new_v, dim=-1).cpu().numpy(), average='macro')
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(log_p_pred.exp() * log_p_pred, dim=-1))
            total_norm = 0
            for param in self.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return np.mean(p_l), np.mean(v_l), np.mean(s_l), np.mean(const_loss), float(entropy), total_norm, f1
