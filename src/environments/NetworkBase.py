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

    @staticmethod
    def _turn_sign_from_state(state, winner_flat):
        return torch.where(
            state[:, 2, 0, 0] >= 0,
            torch.ones_like(winner_flat),
            -torch.ones_like(winner_flat),
        )

    def _value_class_from_batch(self, state, winner):
        winner_flat = winner.view(-1).long()
        turn_sign = self._turn_sign_from_state(state, winner_flat)
        value_class = torch.zeros_like(winner_flat)
        non_draw = winner_flat != 0
        value_class[non_draw & (winner_flat == turn_sign)] = 1
        value_class[non_draw & (winner_flat != turn_sign)] = 2
        return value_class, turn_sign

    def _prepare_training_batch(self, batch, augment):
        state, prob, winner, steps_to_end, aux_target, root_wdl = augment(batch)
        value_class, turn_sign = self._value_class_from_batch(state, winner)
        return {
            'state': state,
            'prob': prob,
            'steps_to_end': steps_to_end,
            'aux_target': self.encode_aux_target(aux_target),
            'root_wdl': root_wdl,
            'value_class': value_class,
            'turn_sign': turn_sign,
            'policy_mask': (prob.sum(dim=1) > 0).float(),
        }

    @staticmethod
    def _soft_value_targets(value_class, steps_to_end, value_decay):
        z_target = F.one_hot(value_class, 3).float()
        if value_decay < 1.0:
            discount = (value_decay ** steps_to_end.float().view(-1)).unsqueeze(1)
            z_target = discount * z_target + (1 - discount) * (1.0 / 3.0)
        return z_target

    @staticmethod
    def _root_wdl_to_relative(root_wdl, turn_sign):
        turn_pos = (turn_sign > 0).view(-1, 1)
        root_draw = root_wdl[:, 0:1]
        root_win = torch.where(turn_pos, root_wdl[:, 1:2], root_wdl[:, 2:3])
        root_loss = torch.where(turn_pos, root_wdl[:, 2:3], root_wdl[:, 1:2])
        return torch.cat([root_draw, root_win, root_loss], dim=1)

    def _distill_value_loss(self, value_pred, root_wdl, turn_sign, distill_alpha, distill_temp, base_value_loss):
        if distill_alpha <= 0:
            return base_value_loss

        root_wdl_rel = self._root_wdl_to_relative(root_wdl, turn_sign)
        has_q = (root_wdl_rel.sum(dim=1, keepdim=True) > 0).float()
        teacher_log = torch.log(root_wdl_rel.clamp(min=1e-8))
        teacher_soft = F.softmax(teacher_log / distill_temp, dim=1)
        student_log_soft = F.log_softmax(value_pred / distill_temp, dim=1)
        kl = F.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1)
        distill_loss = (kl * has_q.squeeze(1)).mean() * (distill_temp ** 2)
        return (1 - distill_alpha) * base_value_loss + distill_alpha * distill_loss

    def _value_loss(self, value_pred, batch_data, use_soft, value_decay, distill_alpha, distill_temp):
        if not use_soft:
            return F.nll_loss(value_pred, batch_data['value_class'])

        z_target = self._soft_value_targets(
            batch_data['value_class'],
            batch_data['steps_to_end'],
            value_decay,
        )
        base_value_loss = -torch.sum(z_target * value_pred, dim=1).mean()
        return self._distill_value_loss(
            value_pred,
            batch_data['root_wdl'],
            batch_data['turn_sign'],
            distill_alpha,
            distill_temp,
            base_value_loss,
        )

    @staticmethod
    def _policy_loss(log_p_pred, prob, policy_mask):
        per_sample_p = -torch.sum(prob * log_p_pred, dim=1)
        return torch.mean(per_sample_p * policy_mask)

    @staticmethod
    def _aux_loss(steps_pred, aux_target):
        aux_target = aux_target.clamp(0, steps_pred.shape[-1] - 1)
        return F.nll_loss(steps_pred, aux_target)

    def _optimize_batch(self, model, batch_data, use_soft, value_decay, distill_alpha, distill_temp):
        self.opt.zero_grad()
        log_p_pred, value_pred, steps_pred = model(batch_data['state'])
        p_loss = self._policy_loss(log_p_pred, batch_data['prob'], batch_data['policy_mask'])
        v_loss = self._value_loss(
            value_pred,
            batch_data,
            use_soft,
            value_decay,
            distill_alpha,
            distill_temp,
        )
        s_loss = self._aux_loss(steps_pred, batch_data['aux_target'])
        loss = p_loss + v_loss + 0.5 * s_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.opt.step()
        return p_loss, v_loss, s_loss, log_p_pred

    @staticmethod
    def _grad_norm(parameters):
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _final_train_metrics(self, last_batch, last_log_p_pred):
        with torch.no_grad():
            _, new_v, _ = self(last_batch['state'])
            f1 = f1_score(
                last_batch['value_class'].cpu().numpy(),
                torch.argmax(new_v, dim=-1).cpu().numpy(),
                average='macro'
            )
            entropy = -torch.mean(torch.sum(last_log_p_pred.exp() * last_log_p_pred, dim=-1))
        total_norm = self._grad_norm(self.parameters())
        return float(entropy), total_norm, f1

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

    def _train_step_legacy(self, dataloader, augment, ddp_model=None, n_epochs=10,
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

    def train_step(self, dataloader, augment, ddp_model=None, n_epochs=10,
                   distill_alpha=0.0, value_decay=1.0, distill_temp=1.0):
        model = ddp_model if ddp_model is not None else self
        p_l, v_l, s_l = [], [], []
        use_soft = value_decay < 1.0 or distill_alpha > 0
        last_batch = None
        last_log_p_pred = None

        for _ in range(n_epochs):
            self.train()
            for batch in dataloader:
                last_batch = self._prepare_training_batch(batch, augment)
                p_loss, v_loss, s_loss, last_log_p_pred = self._optimize_batch(
                    model,
                    last_batch,
                    use_soft,
                    value_decay,
                    distill_alpha,
                    distill_temp,
                )
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
                s_l.append(s_loss.item())

        self.eval()
        self.scheduler.step()
        entropy, total_norm, f1 = self._final_train_metrics(last_batch, last_log_p_pred)
        return np.mean(p_l), np.mean(v_l), np.mean(s_l), float(entropy), total_norm, f1
