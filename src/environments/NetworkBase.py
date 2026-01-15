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
if torch.cuda.is_available():
    from torch.amp import autocast


def get_gradient(model, state):
    state.requires_grad_(True)
    probs, values = model(state)
    grads = torch.autograd.grad(inputs=state,
                                outputs=[probs, values],
                                grad_outputs=[torch.ones_like(probs), torch.ones_like(values)],
                                create_graph=True,
                                retain_graph=True)[0]
    return grads


def gradient_penalty(gradient, c_lambda=10):
    if gradient is None:
        return 0
    gradient = gradient.view(len(gradient), -1)
    grad_norm = gradient.norm(2, dim=1)
    return torch.mean(grad_norm - 0.5) ** 2 * c_lambda


class Base(ABC, nn.Module):
    def save(self, path=None):
        while True:
            try:
                if path is not None:
                    checkpoint = {'model_state_dict': self.state_dict(),
                                  'opt_state_dict': self.opt.state_dict(),
                                  'scheduler_state_dict': self.scheduler.state_dict(),
                                  'scaler_state_dict': self.scaler.state_dict() if self.scaler else None}
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
                if self.scaler is not None and checkpoint['scaler_state_dict'] is not None:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                print(f'Failed to load parameters.\n{e}')
        return self

    def train_step(self, dataloader, augment):
        p_l, v_l = [], []
        self.train()
        for _ in range(5):
            for batch in dataloader:
                state, _, prob, discount, winner, next_state, _ = augment(batch)
                value = deepcopy(winner)
                value[value == -1] = 2
                value = value.view(-1,).long()
                value_oppo = deepcopy(winner)
                value_oppo[value_oppo == 1] = -2
                value_oppo = (-value_oppo).view(-1,).long()
                self.opt.zero_grad()
                if self.scaler is None:
                    log_p_pred, value_pred = self(state)
                    _, next_value_pred = self(next_state)
                    v_loss = (F.nll_loss(value_pred, value, reduction='none') * discount).mean()
                    v_loss += (F.nll_loss(next_value_pred, value_oppo, reduction='none') * discount).mean()
                    p_loss = torch.mean(torch.sum(-prob * log_p_pred - 0.03 * log_p_pred, dim=1))
                    # grad = get_gradient(self, state)
                    # gp = gradient_penalty(grad)
                    loss = p_loss + v_loss
                    loss.backward()
                    self.opt.step()
                else:
                    with autocast(self.device):
                        log_p_pred, value_pred = self(state)
                        _, next_value_pred = self(next_state)
                        v_loss = (F.nll_loss(value_pred, value, reduction='none') * discount).mean()
                        v_loss += (F.nll_loss(next_value_pred, value_oppo, reduction='none') * discount).mean()
                        p_loss = torch.mean(torch.sum(-prob * log_p_pred - 0.03 * log_p_pred, dim=1))
                        # grad = get_gradient(self, state)
                        # gp = gradient_penalty(grad)
                        loss = p_loss + v_loss
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
        self.scheduler.step()
        self.eval()
        with torch.no_grad():
            _, new_v = self(state)
        f1 = f1_score(value.cpu().numpy(), torch.argmax(new_v, dim=-1).cpu().numpy(), average='macro')
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(log_p_pred.exp() * log_p_pred, dim=-1))
            total_norm = 0
            for param in self.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return np.mean(p_l), np.mean(v_l), float(entropy), total_norm, f1
