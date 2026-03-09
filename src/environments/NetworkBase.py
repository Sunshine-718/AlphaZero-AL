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
from sklearn.metrics import f1_score


class Base(ABC, nn.Module):
    aux_target_offset = 0

    def encode_aux_target(self, aux_target):
        return aux_target.view(-1).long() + int(getattr(self, 'aux_target_offset', 0))

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
                   distill_alpha=0.0, value_decay=1.0, distill_temp=1.0):
        model = ddp_model if ddp_model is not None else self
        p_l, v_l, s_l = [], [], []
        use_soft = value_decay < 1.0 or distill_alpha > 0
        for _ in range(n_epochs):
            self.train()
            for batch in dataloader:
                state, prob, winner, steps_to_end, aux_target, root_wdl = augment(batch)
                # Value target in relative perspective:
                # class 0=draw, 1=win(to-move), 2=loss(to-move)
                winner_flat = winner.view(-1).long()
                turn_sign = torch.where(
                    state[:, 2, 0, 0] >= 0,
                    torch.ones_like(winner_flat),
                    -torch.ones_like(winner_flat),
                )
                value_class = torch.zeros_like(winner_flat)
                non_draw = winner_flat != 0
                value_class[non_draw & (winner_flat == turn_sign)] = 1
                value_class[non_draw & (winner_flat != turn_sign)] = 2
                aux_target = self.encode_aux_target(aux_target)
                mask = (prob.sum(dim=1) > 0).float()
                self.opt.zero_grad()
                log_p_pred, value_pred, steps_pred = model(state)
                aux_target = aux_target.clamp(0, steps_pred.shape[-1] - 1)

                if use_soft:
                    z_target = F.one_hot(value_class, 3).float()

                    # Game-length discount: γ^steps × one_hot + (1-γ^steps) × uniform
                    if value_decay < 1.0:
                        discount = (value_decay ** steps_to_end.float().view(-1)).unsqueeze(1)
                        z_target = discount * z_target + (1 - discount) * (1.0 / 3.0)

                    # Hard label loss: CE(student, z_target)
                    v_loss = -torch.sum(z_target * value_pred, dim=1).mean()

                    # Knowledge distillation from MCTS root WDL
                    if distill_alpha > 0:
                        # root_wdl is absolute [draw, p1w, p2w]; convert to relative [draw, win, loss]
                        turn_pos = (turn_sign > 0).view(-1, 1)
                        root_draw = root_wdl[:, 0:1]
                        root_win = torch.where(turn_pos, root_wdl[:, 1:2], root_wdl[:, 2:3])
                        root_loss = torch.where(turn_pos, root_wdl[:, 2:3], root_wdl[:, 1:2])
                        root_wdl_rel = torch.cat([root_draw, root_win, root_loss], dim=1)

                        has_q = (root_wdl_rel.sum(dim=1, keepdim=True) > 0).float()

                        # Teacher: softmax(log(root_wdl) / T)
                        teacher_log = torch.log(root_wdl_rel.clamp(min=1e-8))
                        teacher_soft = F.softmax(teacher_log / distill_temp, dim=1)
                        # Student: softmax(logits / T)  (value_pred is log_softmax)
                        student_log_soft = F.log_softmax(value_pred / distill_temp, dim=1)
                        # KL(teacher || student) per sample, masked by has_q
                        kl = F.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1)
                        distill_loss = (kl * has_q.squeeze(1)).mean() * (distill_temp ** 2)

                        v_loss = (1 - distill_alpha) * v_loss + distill_alpha * distill_loss
                else:
                    v_loss = F.nll_loss(value_pred, value_class)

                per_sample_p = -torch.sum(prob * log_p_pred, dim=1)
                p_loss = torch.mean(per_sample_p * mask)
                s_loss = F.nll_loss(steps_pred, aux_target)
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
