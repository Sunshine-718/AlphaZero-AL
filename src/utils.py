#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 21/Jul/2024  03:52
import numpy as np
from operator import itemgetter


class Elo:
    def __init__(self, init_A=1500, init_B=1500):
        self.R_A = init_A
        self.R_B = init_B

    def update(self, result_a, k=32):
        """
        :param result_a: 1 (A win), 0.5 (draw), 0 (A lose).
        :param k: factor, default: 32.
        """
        expected_a = 1 / (1 + pow(10, (self.R_B - self.R_A) / 400))
        expected_b = 1 / (1 + pow(10, (self.R_A - self.R_B) / 400))
        self.R_A = max(self.R_A + k * (result_a - expected_a), 1500)
        self.R_B = max(self.R_B + k * ((1 - result_a) - expected_b), 1500)
        return self.R_A, self.R_B


def evaluate_rollout(env, limit=1000):
    player = env.turn
    for _ in range(limit):
        if env.done():
            break
        action_probs = rollout_policy_fn(env)
        max_action = max(action_probs, key=itemgetter(1))[0]
        env.step(max_action)
    else:
        print('Warning: rollout reached move limit.')
    winner = env.winPlayer()
    if winner == 0:
        return 0
    else:
        return 1 if winner == player else -1


def policy_value_fn(env):
    valid = env.valid_move()
    action_probs = np.ones(len(valid)) / len(valid)
    return list(zip(valid, action_probs)), evaluate_rollout(env.copy())


def rollout_policy_fn(env):
    valid = env.valid_move()
    probs = np.random.rand(len(valid))
    return list(zip(valid, probs))


def softmax(x):
    probs = np.exp(x - np.max(x))
    return probs / np.sum(probs)


class RolloutAdapter:
    """Random rollout adapter — 将随机 rollout 包装成 NN predict 接口。
    供 MCTSPlayer (纯 MCTS 基线) 使用。"""
    n_actions = 7

    def predict(self, state):
        from src.environments.Connect4 import Env
        batch = state.shape[0] if state.ndim == 4 else 1
        if state.ndim == 3:
            state = state[np.newaxis, ...]
        all_probs = np.ones((batch, self.n_actions), dtype=np.float32) / self.n_actions
        all_vals = np.zeros((batch, 1), dtype=np.float32)
        all_ml = np.full((batch, 1), 0.5, dtype=np.float32)
        for i in range(batch):
            env = Env()
            env.board = (state[i, 0] - state[i, 1]).astype(np.float32)
            env.turn = 1 if state[i, 2, 0, 0] > 0 else -1
            all_vals[i, 0] = evaluate_rollout(env.copy())
        return all_probs, all_vals, all_ml
