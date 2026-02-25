#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 10/Aug/2024  23:47
import numpy as np
from abc import abstractmethod, ABC
from .utils import softmax
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
    """Pure MCTS baseline (uniform prior + random rollout) — wraps C++ BatchedMCTS."""
    def __init__(self, c_puct=4, n_playout=1000, eps=0.0):
        super().__init__()
        self.mcts = BatchedMCTS(
            batch_size=1, c_init=c_puct, c_base=500,
            alpha=0, n_playout=n_playout,
            noise_epsilon=0.0, fpu_reduction=0.0, use_symmetry=False,
        )

    def reset_player(self):
        self.mcts.reset_env(0)

    def get_action(self, env, *args, **kwargs):
        board = env.board[np.newaxis, ...]
        turns = np.array([env.turn], dtype=np.int32)
        self.mcts.rollout_playout(board, turns)
        visits = self.mcts.get_visits_count()[0]
        action = int(np.argmax(visits))
        self.reset_player()
        return action, None


class AlphaZeroPlayer(Player):
    """AlphaZero MCTS — wraps C++ BatchedMCTS with n_envs=1."""
    def __init__(self, policy_value_fn, c_init=1.25, n_playout=100, alpha=None, is_selfplay=0,
                 cache_size=0, eps=0.25, fpu_reduction=0.4, use_symmetry=True,
                 mlh_slope=0.0, mlh_cap=0.2):
        super().__init__()
        self.pv_fn = policy_value_fn
        self.is_selfplay = is_selfplay
        self.n_actions = getattr(policy_value_fn, 'n_actions', 7) if policy_value_fn else 7
        self._noise_eps = eps
        self._alpha = alpha if alpha is not None else 0.3
        self._c_init = c_init if c_init is not None else 1.25
        self._c_base = 500
        self._n_playout = n_playout if n_playout is not None else 100
        self._cache_size = cache_size
        self._fpu_reduction = fpu_reduction
        self._use_symmetry = use_symmetry
        self._mlh_slope = mlh_slope
        self._mlh_cap = mlh_cap

        # alpha=None → eval mode, no noise from construction
        init_eps = eps if alpha is not None else 0.0

        if policy_value_fn is not None and c_init is not None and n_playout is not None:
            self.mcts = self._make_mcts(init_eps)
        else:
            self.mcts = None  # gui_play creates with None, then reload

    def _make_mcts(self, noise_eps):
        return BatchedMCTS(
            batch_size=1, c_init=self._c_init, c_base=self._c_base,
            alpha=self._alpha, n_playout=self._n_playout,
            cache_size=self._cache_size, noise_epsilon=noise_eps,
            fpu_reduction=self._fpu_reduction, use_symmetry=self._use_symmetry,
            mlh_slope=self._mlh_slope, mlh_cap=self._mlh_cap,
        )

    def reload(self, policy_value_fn, c_puct=None, n_playout=None, alpha=None, is_self_play=None):
        self.pv_fn = policy_value_fn
        if c_puct is not None:
            self._c_init = c_puct
        if n_playout is not None:
            self._n_playout = n_playout
        if alpha is not None:
            self._alpha = alpha
        if is_self_play is not None:
            self.is_selfplay = is_self_play
        self.n_actions = policy_value_fn.n_actions
        # Recreate MCTS with updated params
        self.mcts = self._make_mcts(self._noise_eps)

    def to(self, device='cpu'):
        self.pv_fn.to(device)

    def train(self):
        if self.mcts:
            self.mcts.set_noise_epsilon(self._noise_eps)

    def eval(self):
        if self.mcts:
            self.mcts.set_noise_epsilon(0.0)

    def reset_player(self):
        if self.mcts:
            self.mcts.reset_env(0)

    def get_action(self, env, temp=0):
        board = env.board[np.newaxis, ...]  # (1, 6, 7)
        turns = np.array([env.turn], dtype=np.int32)

        self.mcts.batch_playout(self.pv_fn, board, turns)
        visits = self.mcts.get_visits_count()[0]

        action_probs = np.zeros(self.n_actions, dtype=np.float32)
        valid_mask = visits > 0

        if not valid_mask.any():
            return 0, action_probs

        action_probs[valid_mask] = visits[valid_mask] / visits[valid_mask].sum()

        if temp <= 1e-6:
            action = int(np.argmax(visits))
        else:
            valid_actions = np.where(valid_mask)[0]
            log_visits = np.log(visits[valid_mask])
            sample_dist = softmax(log_visits / temp)
            action = np.random.choice(valid_actions, p=sample_dist)

        if self.is_selfplay:
            self.mcts.prune_roots(np.array([action], dtype=np.int32))
        else:
            self.reset_player()

        return action, action_probs


class BatchedAlphaZeroPlayer:
    def __init__(self, policy_value_fn, n_envs, c_init=1.25, c_base=500, n_playout=100, alpha=0.3, noise_epsilon=0.25, fpu_reduction=0.4,
                 game_name='Connect4', board_converter=None, cache_size=0, use_symmetry=True,
                 noise_steps=0, noise_eps_min=0.1, mlh_slope=0.0, mlh_cap=0.2):
        self.pv_func = policy_value_fn
        self.mcts = BatchedMCTS(n_envs, c_init, c_base, alpha, n_playout,
                                game_name=game_name, board_converter=board_converter,
                                cache_size=cache_size, noise_epsilon=noise_epsilon, fpu_reduction=fpu_reduction,
                                use_symmetry=use_symmetry, mlh_slope=mlh_slope, mlh_cap=mlh_cap)
        self.n_envs = n_envs
        self.n_actions = self.mcts.action_size
        self.noise_eps_init = noise_epsilon
        self.noise_steps = noise_steps
        self.noise_eps_min = noise_eps_min

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

            # 已结束的游戏可能所有 visit 都为 0（终局状态被正确识别），
            # 此时返回 action=0 和全零 probs，调用方不会使用这些值。
            if not valid_mask.any():
                batch_actions.append(0)
                batch_probs.append(action_probs)
                continue

            action_probs[valid_mask] = visit[valid_mask] / visit[valid_mask].sum()

            # 动作采样：根据 temp 缩放后的分布采样
            if temp <= 1e-6:
                action = np.argmax(visit)
            else:
                valid_actions = np.where(valid_mask)[0]
                # 用 log 域计算 N^(1/temp)，等价但数值稳定
                log_visits = np.log(visit[valid_mask])
                sample_dist = softmax(log_visits / temp)
                action = np.random.choice(valid_actions, p=sample_dist)

            batch_actions.append(action)
            batch_probs.append(action_probs)
        actions_array = np.array(batch_actions, dtype=np.int32)
        self.mcts.prune_roots(actions_array)
        return batch_actions, np.array(batch_probs)
