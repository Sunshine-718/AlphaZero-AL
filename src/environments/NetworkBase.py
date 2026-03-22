#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:20
import copy
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
        # Unpack: 6 base fields + optional valid_mask + optional future_state
        future_state = None
        valid_mask = None
        if len(augmented) == 8:
            state, prob, winner, steps_to_end, aux_target, root_wdl, valid_mask, future_state = augmented
        elif len(augmented) == 7:
            state, prob, winner, steps_to_end, aux_target, root_wdl, extra = augmented
            if extra.dtype == torch.bool:
                valid_mask = extra
            else:
                future_state = extra
        else:
            state, prob, winner, steps_to_end, aux_target, root_wdl = augmented
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
        }
        if valid_mask is not None:
            result['valid_mask'] = valid_mask
        if future_state is not None:
            result['future_state'] = future_state
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
        # KL(π || p) per action; nan_to_num handles 0*log(0) and 0*(-inf) edge cases
        kl_per_action = F.kl_div(log_p_pred, prob, reduction='none').nan_to_num(0.0)
        per_sample_kl = kl_per_action.sum(dim=1)

        # PSW: weight by surprise
        if psw_beta > 0:
            with torch.no_grad():
                weights = 1.0 + psw_beta * per_sample_kl.detach()
            weighted_kl = per_sample_kl * weights * policy_mask
        else:
            weighted_kl = per_sample_kl * policy_mask

        p_loss = weighted_kl.mean()

        # Entropy regularization: maximize H(p) = -Σ p log p
        if entropy_lambda > 0:
            p_pred = log_p_pred.exp()
            entropy = -torch.sum((p_pred * log_p_pred).nan_to_num(0.0), dim=1)
            p_loss = p_loss - entropy_lambda * (entropy * policy_mask).mean()

        return p_loss

    @staticmethod
    def _aux_loss(steps_pred, aux_target):
        return F.smooth_l1_loss(steps_pred, aux_target)

    @staticmethod
    def _td_consistency_loss(value_pred, future_wdl_target, batch_data, td_steps,
                             value_decay=1.0):
        """N-step TD consistency: KL(stopgrad(γ^k·v(S_{t+k})+(1-γ^k)·U) || v(S_t)).

        Only applies to positions with steps_to_end > td_steps (non-terminal bootstrap).
        future_wdl_target is already perspective-adjusted and detached.
        When value_decay < 1, the bootstrap target is discounted to match the
        decayed value targets used by the main value loss.
        """
        # Mask: only bootstrap from positions far enough from terminal
        td_mask = (batch_data['steps_to_end'].view(-1) > td_steps).float()
        n_valid = td_mask.sum().clamp(min=1)

        # Apply value decay discount: v(S_t) ≈ γ^k · v(S_{t+k}) + (1-γ^k) · uniform
        if value_decay < 1.0:
            discount = value_decay ** td_steps
            future_wdl_target = discount * future_wdl_target + (1 - discount) / 3.0

        # KL(future_target || current_pred)
        kl = F.kl_div(value_pred, future_wdl_target, reduction='none').sum(dim=1)
        return (kl * td_mask).sum() / n_valid

    def _ensure_target_net(self):
        """Lazy-init EMA target network for stable TD bootstrap targets.

        Uses object.__setattr__ to bypass nn.Module registration so that
        _target_net is NOT included in state_dict / saved checkpoints.
        On load, _target_net is re-created from main network weights.
        """
        if getattr(self, '_target_net', None) is None:
            target = copy.deepcopy(self)
            # target net is inference-only — drop optimizer/scheduler to save memory
            for attr in ('opt', 'scheduler', '_target_net'):
                if hasattr(target, attr):
                    delattr(target, attr)
            target.requires_grad_(False)
            target.eval()
            # Store as plain attribute, not a registered submodule
            object.__setattr__(self, '_target_net', target)

    @torch.no_grad()
    def _update_target_net(self, tau):
        """Soft update: θ_target = τ·θ_target + (1-τ)·θ_online"""
        for p_tgt, p_src in zip(self._target_net.parameters(), self.parameters()):
            p_tgt.data.lerp_(p_src.data, 1 - tau)

    def _prepare_future_wdl(self, batch_data):
        """用 target network 计算 future state 的 WDL target。"""
        future_state = batch_data['future_state']
        with torch.no_grad():
            _, future_value_log, _ = self._target_net(future_state)
            future_wdl = future_value_log.exp()  # (B, 3) [draw, win, loss]

        # Determine perspective: compare turn plane of current vs future state
        current_turn = batch_data['state'][:, 2, 0, 0]
        future_turn = future_state[:, 2, 0, 0]
        diff_perspective = (current_turn * future_turn) < 0

        # Swap win/loss for different perspective
        swap = diff_perspective.unsqueeze(1)
        win = torch.where(swap, future_wdl[:, 2:3], future_wdl[:, 1:2])
        loss = torch.where(swap, future_wdl[:, 1:2], future_wdl[:, 2:3])
        return torch.cat([future_wdl[:, 0:1], win, loss], dim=1)

    def _optimize_batch(self, model, batch_data, use_soft, value_decay, distill_alpha, distill_temp,
                        psw_beta=0.0, entropy_lambda=0.0, td_alpha=0.0, td_steps=5,
                        target_tau=0.995):
        has_future = 'future_state' in batch_data
        need_target = td_alpha > 0 and has_future

        # 用 target network 计算 future state WDL（在主前向传播之前，避免 inplace op 破坏计算图）
        future_wdl_target = None
        if need_target:
            self._ensure_target_net()
            future_wdl_target = self._prepare_future_wdl(batch_data)

        self.opt.zero_grad(set_to_none=True)
        log_p_pred, value_pred, steps_pred = model(batch_data['state'],
                                                    action_mask=batch_data.get('valid_mask'))
        p_loss = self._policy_loss(log_p_pred, batch_data['prob'], batch_data['policy_mask'],
                                   psw_beta, entropy_lambda)
        v_loss = self._value_loss(
            value_pred,
            batch_data,
            use_soft,
            value_decay,
            distill_alpha,
            distill_temp,
        )

        # N-step TD consistency loss
        if future_wdl_target is not None:
            td_loss = self._td_consistency_loss(value_pred, future_wdl_target, batch_data, td_steps,
                                                    value_decay)
            v_loss = (1 - td_alpha) * v_loss + td_alpha * td_loss

        aux_loss = self._aux_loss(steps_pred, batch_data['aux_target'])
        loss = p_loss + v_loss + aux_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.opt.step()
        if need_target:
            self._update_target_net(target_tau)
        return p_loss.detach(), v_loss.detach(), aux_loss.detach(), log_p_pred, grad_norm

    def _final_train_metrics(self, last_batch, last_log_p_pred):
        with torch.no_grad():
            _, new_v, _ = self(last_batch['state'])
            f1 = f1_score(
                last_batch['value_class'].cpu().numpy(),
                torch.argmax(new_v, dim=-1).cpu().numpy(),
                average='macro'
            )
            plogp = last_log_p_pred.exp() * last_log_p_pred
            plogp = plogp.nan_to_num(0.0)  # mask 导致的 0 * -inf → NaN 置零
            entropy = -torch.mean(torch.sum(plogp, dim=-1))
        return float(entropy), f1

    def save(self, dir_path=None):
        """Save model weights and training state to a directory.

        Directory structure:
            dir_path/model.pt       — model state_dict
            dir_path/optimizer.pt   — optimizer state_dict
            dir_path/scheduler.pt   — scheduler state_dict
        """
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

    def load(self, path=None):
        """Load model weights and optionally training state.

        Supports three formats:
        - Directory (new): path/model.pt, path/optimizer.pt, path/scheduler.pt
        - Single file (legacy v1): {'model_state_dict': ..., 'opt_state_dict': ...}
        - Single file (legacy v2): bare state_dict
        """
        if path is None:
            return self
        try:
            if os.path.isdir(path):
                model_file = os.path.join(path, 'model.pt')
                self.load_state_dict(
                    torch.load(model_file, map_location=self.device, weights_only=True),
                    strict=False)
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
            else:
                # Legacy single-file format
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                if 'model_state_dict' in checkpoint:
                    self.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    try:
                        self.opt.load_state_dict(checkpoint['opt_state_dict'])
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except (ValueError, KeyError) as e:
                        print(f'Optimizer/scheduler state incompatible, using fresh state.\n{e}')
                else:
                    self.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f'Failed to load parameters.\n{e}')
        # Invalidate target net so it re-inits from the newly loaded weights
        object.__setattr__(self, '_target_net', None)
        return self

    def train_step(self, dataloader, augment, ddp_model=None, n_epochs=10,
                   distill_alpha=0.0, value_decay=1.0, distill_temp=1.0,
                   psw_beta=0.0, entropy_lambda=0.0, td_alpha=0.0, td_steps=5,
                   target_tau=0.995):
        model = ddp_model if ddp_model is not None else self
        device = next(self.parameters()).device
        p_sum = torch.zeros(1, device=device)
        v_sum = torch.zeros(1, device=device)
        aux_sum = torch.zeros(1, device=device)
        n_batches = 0
        use_soft = value_decay < 1.0 or distill_alpha > 0
        last_batch = None
        last_log_p_pred = None
        last_grad_norm = 0.0

        for _ in range(n_epochs):
            self.train()
            for batch in dataloader:
                last_batch = self._prepare_training_batch(batch, augment)
                p_loss, v_loss, aux_loss, last_log_p_pred, last_grad_norm = self._optimize_batch(
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
                    target_tau,
                )
                p_sum += p_loss
                v_sum += v_loss
                aux_sum += aux_loss
                n_batches += 1

        self.eval()
        self.scheduler.step()
        n_batches = max(n_batches, 1)
        entropy, f1 = self._final_train_metrics(last_batch, last_log_p_pred)
        return (
            (p_sum / n_batches).item(),
            (v_sum / n_batches).item(),
            (aux_sum / n_batches).item(),
            float(entropy),
            float(last_grad_norm),
            f1,
        )
