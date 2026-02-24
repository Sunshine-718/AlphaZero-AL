#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:59
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim, row, col, replay_ratio=0.25, device='cpu'):
        self.state = torch.empty((capacity, state_dim, row, col), dtype=torch.int8, device=device)
        self.prob = torch.empty((capacity, action_dim), dtype=torch.float32, device=device)
        self.winner = torch.full((capacity, 1), 0, dtype=torch.int8, device=device)
        self.steps_to_end = torch.full((capacity, 1), 0, dtype=torch.int8, device=device)
        self.replay_ratio = replay_ratio
        self.device = device
        self.current_capacity = capacity
        self._ptr = 0

    def save(self, path):
        state_dict = {
            'state': self.state,
            'prob': self.prob,
            'winner': self.winner,
            'steps_to_end': self.steps_to_end,
            '_ptr': self._ptr,
            'current_capacity': self.current_capacity
        }
        torch.save(state_dict, path)

    def load(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            capacity = min(self.state.shape[0], state_dict['state'].shape[0])
            self.state[:capacity].copy_(state_dict['state'][:capacity])
            self.prob[:capacity].copy_(state_dict['prob'][:capacity])
            self.winner[:capacity].copy_(state_dict['winner'][:capacity])
            if 'steps_to_end' in state_dict:
                self.steps_to_end[:capacity].copy_(state_dict['steps_to_end'][:capacity])
            self._ptr = state_dict['_ptr']
        except Exception as e:
            print(e)
        return self

    def __len__(self):
        return min(self._ptr, len(self.state))

    def is_full(self):
        return len(self) >= len(self.state)

    def reset(self):
        self.state = torch.empty_like(self.state)
        self.prob = torch.empty_like(self.prob)
        self.winner = torch.empty_like(self.winner)
        self.steps_to_end = torch.empty_like(self.steps_to_end)
        self._ptr = 0

    def to(self, device='cpu'):
        self.state = self.state.to(device)
        self.prob = self.prob.to(device)
        self.winner = self.winner.to(device)
        self.steps_to_end = self.steps_to_end.to(device)
        self.device = device

    def store(self, state, prob, winner, steps_to_end=0):
        idx = self._ptr % self.current_capacity
        self._ptr += 1
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        self.state[idx] = state
        if isinstance(prob, np.ndarray):
            prob = torch.from_numpy(prob).float().to(self.device)
        self.prob[idx] = prob
        self.winner[idx] = int(winner)
        self.steps_to_end[idx] = int(steps_to_end)
        return idx

    def get(self, indices):
        return self.state[indices].float(), self.prob[indices], self.winner[indices], self.steps_to_end[indices]

    def sample(self, batch_size):
        total_samples = len(self)
        assert len(self) > 0
        max_samples = int(total_samples * self.replay_ratio) if len(self) > 10000 / self.replay_ratio else min(total_samples, 10000)

        idx = torch.from_numpy(np.random.randint(0, len(self), max_samples, dtype=np.int64))

        dataset = TensorDataset(*self.get(idx))
        return DataLoader(dataset, batch_size, True)
