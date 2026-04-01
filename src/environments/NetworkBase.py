#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:20
import os
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
        offset = float(getattr(self, 'aux_target_offset', 1))
        return aux_target.view(-1).float() / offset  # normalize to [-1, 1]

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
        augmented = augment(batch)
        (state, prob, winner, steps_to_end, aux_target, root_wdl,
         valid_mask, future_root_wdl, ownership_target) = augmented
        value_class, turn_sign = self._value_class_from_batch(state, winner)
        result = {
            'state': state,
            'prob': prob,
            'steps_to_end': steps_to_end,
            'aux_target': self.encode_aux_target(aux_target),
            'root_wdl': root_wdl,
            'value_class': value_class,
            'turn_sign': turn_sign,
            'policy_mask': (prob.sum(dim=1) > 0).float(),
            'valid_mask': valid_mask,
            'future_root_wdl': future_root_wdl,
            'ownership_target': ownership_target.long(),
        }
        return result

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
    def _policy_loss(log_p_pred, prob, policy_mask, psw_beta=0.0, entropy_lambda=0.0):
        # KL(target || student) per action; nan_to_num handles 0*log(0) and 0*(-inf)
        kl_per_action = F.kl_div(log_p_pred, prob, reduction='none').nan_to_num(0.0)
        per_sample_kl = kl_per_action.sum(dim=1)

        if psw_beta > 0:
            with torch.no_grad():
                weights = 1.0 + psw_beta * per_sample_kl.detach()
            weighted_kl = per_sample_kl * weights * policy_mask
        else:
            weighted_kl = per_sample_kl * policy_mask

        p_loss = weighted_kl.mean()

        if entropy_lambda > 0:
            p_pred = log_p_pred.exp()
            entropy = -torch.sum((p_pred * log_p_pred).nan_to_num(0.0), dim=1)
            p_loss = p_loss - entropy_lambda * (entropy * policy_mask).mean()

        return p_loss

    @staticmethod
    def _aux_loss(steps_pred, aux_target):
        return F.smooth_l1_loss(steps_pred, aux_target)

    @staticmethod
    def _ownership_loss(ownership_pred, ownership_target):
        if ownership_pred is None or ownership_target is None:
            return None
        # Deliberately independent from policy_mask so terminal states still supervise ownership.
        if not torch.any(ownership_target >= 0):
            return None
        return F.nll_loss(ownership_pred, ownership_target, ignore_index=-1)

    @staticmethod
    def _td_consistency_loss(value_pred, batch_data, td_steps, value_decay=1.0):
        """N-step root-WDL consistency: KL(stopgrad(root_wdl(S_{t+k})) || v(S_t))."""
        future_root_wdl = batch_data.get('future_root_wdl')
        if future_root_wdl is None:
            return None

        future_wdl_target = Base._root_wdl_to_relative(future_root_wdl, batch_data['turn_sign'])
        future_mass = future_wdl_target.sum(dim=1)
        td_mask = (batch_data['steps_to_end'].view(-1) > td_steps) & (future_mass > 0)
        if not torch.any(td_mask):
            return None

        future_wdl_target = future_wdl_target / future_wdl_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
        if value_decay < 1.0:
            discount = value_decay ** td_steps
            future_wdl_target = discount * future_wdl_target + (1 - discount) / 3.0

        kl = F.kl_div(value_pred, future_wdl_target, reduction='none').sum(dim=1)
        return kl[td_mask].mean()

    def _optimize_batch(self, model, batch_data, use_soft, value_decay, distill_alpha, distill_temp,
                        psw_beta=0.0, entropy_lambda=0.0, td_alpha=0.0, td_steps=5):
        self.opt.zero_grad(set_to_none=True)
        outputs = model(
            batch_data['state'],
            action_mask=batch_data.get('valid_mask'),
        )
        if len(outputs) == 4:
            log_p_pred, value_pred, steps_pred, ownership_pred = outputs
        else:
            log_p_pred, value_pred, steps_pred = outputs
            ownership_pred = None
        p_loss = self._policy_loss(
            log_p_pred,
            batch_data['prob'],
            batch_data['policy_mask'],
            psw_beta,
            entropy_lambda,
        )
        v_loss = self._value_loss(
            value_pred,
            batch_data,
            use_soft,
            value_decay,
            distill_alpha,
            distill_temp,
        )

        if td_alpha > 0:
            td_loss = self._td_consistency_loss(value_pred, batch_data, td_steps, value_decay)
            if td_loss is not None:
                v_loss = (1 - td_alpha) * v_loss + td_alpha * td_loss

        aux_loss = self._aux_loss(steps_pred, batch_data['aux_target'])
        ownership_loss = self._ownership_loss(ownership_pred, batch_data.get('ownership_target'))
        if ownership_loss is None:
            ownership_loss = torch.zeros((), device=batch_data['state'].device)
        loss = p_loss + v_loss + aux_loss + getattr(self, 'ownership_loss_weight', 1.0) * ownership_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.opt.step()
        return (
            p_loss.detach(),
            v_loss.detach(),
            aux_loss.detach(),
            ownership_loss.detach(),
            log_p_pred,
            grad_norm,
        )

    def _final_train_metrics(self, last_batch, last_log_p_pred):
        with torch.no_grad():
            outputs = self(
                last_batch['state'],
                action_mask=last_batch.get('valid_mask'),
            )
            _, new_v, *_ = outputs
            f1 = f1_score(
                last_batch['value_class'].cpu().numpy(),
                torch.argmax(new_v, dim=-1).cpu().numpy(),
                average='macro',
            )
            plogp = (last_log_p_pred.exp() * last_log_p_pred).nan_to_num(0.0)
            entropy = -torch.mean(torch.sum(plogp, dim=-1))
        return float(entropy), f1

    def save(self, dir_path=None):
        """Save model weights and training state to a directory."""
        while True:
            try:
                if dir_path is not None:
                    os.makedirs(dir_path, exist_ok=True)
                    sd = {k.removeprefix('_orig_mod.'): v for k, v in self.state_dict().items()}
                    torch.save(sd, os.path.join(dir_path, 'model.pt'))
                    torch.save(self.opt.state_dict(), os.path.join(dir_path, 'optimizer.pt'))
                    torch.save(self.scheduler.state_dict(), os.path.join(dir_path, 'scheduler.pt'))
                    break
            except RuntimeError:
                print('Failed to save parameters, retry...')
                time.sleep(1)
                continue

    def load_weights_only(self, path=None, strict=True):
        """Load only model weights, skipping optimizer and scheduler state."""
        if path is None:
            return self

        model_file = os.path.join(path, 'model.pt')
        state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
        try:
            self.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if not strict:
                raise
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f'Loaded with non-strict state_dict.\nMissing: {missing}\nUnexpected: {unexpected}\n{e}')
        return self

    def load(self, path=None):
        """Load model weights and optionally training state."""
        if path is None:
            return self
        try:
            self.load_weights_only(path, strict=True)
            opt_file = os.path.join(path, 'optimizer.pt')
            sched_file = os.path.join(path, 'scheduler.pt')
            if os.path.exists(opt_file):
                try:
                    self.opt.load_state_dict(
                        torch.load(opt_file, map_location=self.device, weights_only=True))
                except (ValueError, KeyError) as e:
                    print(f'Optimizer state incompatible, using fresh state.\n{e}')
            if os.path.exists(sched_file):
                try:
                    self.scheduler.load_state_dict(
                        torch.load(sched_file, map_location=self.device, weights_only=True))
                except (ValueError, KeyError) as e:
                    print(f'Scheduler state incompatible, using fresh state.\n{e}')
        except Exception as e:
            print(f'Failed to load parameters.\n{e}')
        return self

    def train_step(self, dataloader, augment, ddp_model=None, n_epochs=10,
                   distill_alpha=0.0, value_decay=1.0, distill_temp=1.0,
                   psw_beta=0.0, entropy_lambda=0.0, td_alpha=0.0, td_steps=5):
        model = ddp_model if ddp_model is not None else self
        device = next(self.parameters()).device
        p_sum = torch.zeros(1, device=device)
        v_sum = torch.zeros(1, device=device)
        aux_sum = torch.zeros(1, device=device)
        ownership_sum = torch.zeros(1, device=device)
        n_batches = 0
        use_soft = value_decay < 1.0 or distill_alpha > 0
        last_batch = None
        last_log_p_pred = None
        last_grad_norm = 0.0

        for _ in range(n_epochs):
            self.train()
            for batch in dataloader:
                last_batch = self._prepare_training_batch(batch, augment)
                p_loss, v_loss, aux_loss, ownership_loss, last_log_p_pred, last_grad_norm = self._optimize_batch(
                    model,
                    last_batch,
                    use_soft,
                    value_decay,
                    distill_alpha,
                    distill_temp,
                    psw_beta,
                    entropy_lambda,
                    td_alpha,
                    td_steps,
                )
                p_sum += p_loss
                v_sum += v_loss
                aux_sum += aux_loss
                ownership_sum += ownership_loss
                n_batches += 1

        self.eval()
        self.scheduler.step()
        n_batches = max(n_batches, 1)
        entropy, f1 = self._final_train_metrics(last_batch, last_log_p_pred)
        return (
            (p_sum / n_batches).item(),
            (v_sum / n_batches).item(),
            (aux_sum / n_batches).item(),
            (ownership_sum / n_batches).item(),
            float(entropy),
            float(last_grad_norm),
            f1,
        )

