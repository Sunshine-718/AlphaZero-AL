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
        if ddp_model is None:
            ddp_model = self
        # Access the underlying module (works for both DDP-wrapped and raw models)
        raw_model = ddp_model.module if hasattr(ddp_model, 'module') else ddp_model

        # Register forward hooks on hidden layers to capture intermediate features
        feats = {}
        hooks = []
        for i, layer in enumerate(raw_model.hidden):
            def make_hook(idx):
                def hook(m, inp, out):
                    feats[idx] = out
                return hook
            hooks.append(layer.register_forward_hook(make_hook(i)))

        p_l, v_l, const_loss = [], [], []
        ddp_model.train()
        try:
            for _ in range(10):
                for batch in dataloader:
                    state, _, prob, discount, winner, _, _ = augment(batch)
                    value = deepcopy(winner)
                    value[value == -1] = 2
                    value = value.view(-1,).long()
                    self.opt.zero_grad()
                    layer_weight = [0.1, 0.1, 0.1]
                    total_const_loss = 0
                    batch_split = state.shape[0] // 2

                    # Single forward pass through DDP wrapper — hooks capture intermediate features
                    feats.clear()
                    log_p_pred, value_pred = ddp_model(state)

                    # Per-layer consistency loss using hook-captured features
                    for i in range(len(layer_weight)):
                        feat = feats[i]
                        feat_original = feat[:batch_split]
                        feat_augmented = feat[batch_split:]
                        feat_original_flipped = torch.flip(feat_original, dims=[3])
                        layer_loss = 1 - F.cosine_similarity(feat_original_flipped, feat_augmented, dim=1).mean()
                        total_const_loss += layer_loss * layer_weight[i]

                    log_p_flipped = torch.flip(log_p_pred[:batch_split], dims=[1])
                    target_p = log_p_pred[batch_split:].exp()  # 还原回概率分布
                    policy_const_loss = F.kl_div(log_p_flipped, target_p, reduction='batchmean')
                    total_const_loss += policy_const_loss
                    value_const_loss = F.kl_div(value_pred[:batch_split], value_pred[batch_split:].exp(), reduction='batchmean')
                    total_const_loss += value_const_loss

                    v_loss = (F.nll_loss(value_pred, value, reduction='none') * discount).mean()
                    p_loss = torch.mean(torch.sum(-prob * log_p_pred, dim=1))
                    loss = p_loss + v_loss + total_const_loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.opt.step()
                    p_l.append(p_loss.item())
                    v_l.append(v_loss.item())
                    const_loss.append(total_const_loss.item())
        finally:
            for h in hooks:
                h.remove()
            self.eval()

        self.scheduler.step()
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
        return np.mean(p_l), np.mean(v_l), np.mean(const_loss), float(entropy), total_norm, f1
