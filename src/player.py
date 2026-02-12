#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 10/Aug/2024  23:47
import numpy as np
from abc import abstractmethod, ABC
from src.utils import policy_value_fn, softmax
from .MCTS import MCTS, MCTS_AZ
from .MCTS_cpp import BatchedMCTS


class Player(ABC):
    def __init__(self):
        self.win_rate = float('nan')
        self.mcts = None

    def reset_player(self):
        pass

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, *args, **kwargs):
        raise NotImplementedError


class NetworkPlayer(Player):
    def __init__(self, net, deterministic=True):
        super().__init__()
        self.net = net
        self.pv_fn = self.net
        self.deterministic = deterministic

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def get_action(self, env, *args, **kwargs):
        valid = env.valid_move()
        probs = self.net.policy(env.current_state())
        action_probs = tuple(zip(valid, probs.flatten()[valid]))
        actions, probs = list(zip(*action_probs))
        probs = np.array(probs, dtype=np.float32)
        probs /= probs.sum()
        if self.deterministic:
            action = actions[np.argmax(probs)]
        else:
            action = np.random.choice(actions, p=probs)

        full_probs = np.zeros(self.net.n_actions, dtype=np.float32)
        for a, p in action_probs:
            full_probs[a] = p
        return action, full_probs


class Human(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, env, *args, **kwargs):
        move = int(input('Your move: '))
        return move, None


class MCTSPlayer(Player):
    def __init__(self, c_puct=4, n_playout=1000, discount=1):
        super().__init__()
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout, discount, None)

    @property
    def discount(self):
        return self.mcts.root.discount

    def reset_player(self):
        self.mcts.prune_root(-1)

    def get_action(self, env):
        action = self.mcts.get_action(env)
        self.mcts.prune_root(action)
        return action


class AlphaZeroPlayer(MCTSPlayer):
    def __init__(self, policy_value_fn, c_init=1.25, n_playout=100, discount=None, alpha=None, is_selfplay=0, cache_size=5000):
        self.pv_fn = policy_value_fn
        self.mcts = MCTS_AZ(policy_value_fn, c_init, n_playout, discount, alpha, cache_size)
        self.is_selfplay = is_selfplay
        try:
            self.n_actions = policy_value_fn.n_actions
        except AttributeError:
            self.n_actions = None

    def reload(self, policy_value_fn, c_puct=None, n_playout=None, alpha=None, is_self_play=None):
        self.pv_fn = policy_value_fn
        self.mcts.policy = policy_value_fn
        if c_puct is not None:
            self.mcts.c_init = c_puct
        if n_playout is not None:
            self.mcts.n_playout = n_playout
        if alpha is not None:
            self.mcts.alpha = alpha
        if is_self_play is not None:
            self.is_selfplay = is_self_play
        self.n_actions = policy_value_fn.n_actions
        self.mcts.refresh_cache()

    def to(self, device='cpu'):
        self.pv_fn.to(device)

    def train(self):
        self.mcts.train()

    def eval(self):
        self.mcts.eval()

    def get_action(self, env, temp=0):
        action_probs = np.zeros((self.n_actions,), dtype=np.float32)
        actions, visits = self.mcts.get_action_visits(env)
        visit_dist = softmax(np.log(visits) / max(temp, 1e-8))
        action_probs[list(actions)] = visit_dist
        if temp == 0:
            probs = np.zeros((len(visits),), dtype=np.float32)
            probs[np.where(np.array(visits) == max(visits))] = 1 / list(visits).count(max(visits))
        else:
            probs = visit_dist
        action = np.random.choice(actions, p=probs)
        if self.is_selfplay:
            self.mcts.prune_root(action)
        else:
            self.reset_player()
        return action, action_probs


class BatchedAlphaZeroPlayer:
    def __init__(self, policy_value_fn, n_envs, c_init=1.25, c_base=500, n_playout=100, discount=1, alpha=0.3,
                 game_name='Connect4', board_converter=None, cache_size=0):
        self.pv_func = policy_value_fn
        self.mcts = BatchedMCTS(n_envs, c_init, c_base, discount, alpha, n_playout,
                                game_name=game_name, board_converter=board_converter,
                                cache_size=cache_size)
        self.n_envs = n_envs
        self.n_actions = self.mcts.action_size
        self.discount = discount

    def to(self, device='cpu'):
        self.pv_fn.to(device)

    def reset_player(self):
        for i in range(self.n_envs):
            self.mcts.reset_env(i)

    def get_action(self, current_boards, turns, temps=None):
        assert len(current_boards) == self.n_envs
        if temps is None:
            temps = [0 for _ in range(self.n_envs)]

        self.mcts.batch_playout(self.pv_func, current_boards, turns)

        visits = self.mcts.get_visits_count()

        batch_actions = []
        batch_probs = []

        for i in range(self.n_envs):
            visit = visits[i]
            temp = temps[i]

            # 训练策略目标：使用原始 count 归一化分布，不受 temp 影响
            action_probs = np.zeros(self.n_actions, dtype=np.float32)
            valid_mask = visit > 0
            if valid_mask.any():
                action_probs[valid_mask] = visit[valid_mask] / visit[valid_mask].sum()
            else:
                action_probs = np.ones(self.n_actions, dtype=np.float32) / self.n_actions

            # 动作采样：根据 temp 缩放后的分布采样
            if temp <= 1e-6:
                action = np.argmax(visit)
            else:
                valid_actions = np.where(valid_mask)[0]
                scaled = visit[valid_mask] ** (1.0 / temp)
                sample_dist = scaled / scaled.sum()
                action = np.random.choice(valid_actions, p=sample_dist)

            batch_actions.append(action)
            batch_probs.append(action_probs)
        actions_array = np.array(batch_actions, dtype=np.int32)
        self.mcts.prune_roots(actions_array)
        return batch_actions, np.array(batch_probs)
