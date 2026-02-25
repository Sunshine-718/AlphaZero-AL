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
    """AlphaZero MCTS player — 支持单环境 (n_envs=1) 和批量并行 (n_envs>1) 两种模式。

    单环境模式：get_action(env, temp) — 用于 play.py / gui_play.py / pipeline Elo 评估
    批量模式：  get_batch_action(boards, turns, temps) — 用于 client.py 分布式自对弈
    """
    def __init__(self, policy_value_fn, n_envs=1, c_init=1.25, c_base=500,
                 n_playout=100, alpha=None, is_selfplay=0,
                 cache_size=0, noise_epsilon=0.25, fpu_reduction=0.4, use_symmetry=True,
                 game_name='Connect4', board_converter=None,
                 noise_steps=0, noise_eps_min=0.1,
                 mlh_slope=0.0, mlh_cap=0.2, mlh_threshold=0.8,
                 # 向后兼容：旧调用方用 eps= 而非 noise_epsilon=
                 eps=None):
        super().__init__()
        self.pv_fn = policy_value_fn
        self.is_selfplay = is_selfplay
        self.n_envs = n_envs
        self.n_actions = getattr(policy_value_fn, 'n_actions', 7) if policy_value_fn else 7

        # eps= 是旧参数名的兼容，优先使用 eps（若显式传入），否则用 noise_epsilon
        self._noise_eps = eps if eps is not None else noise_epsilon
        self._alpha = alpha if alpha is not None else 0.3
        self._c_init = c_init if c_init is not None else 1.25
        self._c_base = c_base
        self._n_playout = n_playout if n_playout is not None else 100
        self._cache_size = cache_size
        self._fpu_reduction = fpu_reduction
        self._use_symmetry = use_symmetry
        self._game_name = game_name
        self._board_converter = board_converter
        self._mlh_slope = mlh_slope
        self._mlh_cap = mlh_cap
        self._mlh_threshold = mlh_threshold

        # 噪声衰减（批量自对弈用）
        self.noise_eps_init = self._noise_eps
        self.noise_steps = noise_steps
        self.noise_eps_min = noise_eps_min

        # alpha=None → eval mode, no noise from construction
        init_eps = self._noise_eps if alpha is not None else 0.0

        if policy_value_fn is not None and c_init is not None and n_playout is not None:
            self.mcts = self._make_mcts(init_eps)
        else:
            self.mcts = None  # gui_play creates with None, then reload

    def _make_mcts(self, noise_eps):
        return BatchedMCTS(
            batch_size=self.n_envs, c_init=self._c_init, c_base=self._c_base,
            alpha=self._alpha, n_playout=self._n_playout,
            game_name=self._game_name, board_converter=self._board_converter,
            cache_size=self._cache_size, noise_epsilon=noise_eps,
            fpu_reduction=self._fpu_reduction, use_symmetry=self._use_symmetry,
            mlh_slope=self._mlh_slope, mlh_cap=self._mlh_cap,
            mlh_threshold=self._mlh_threshold,
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
            for i in range(self.n_envs):
                self.mcts.reset_env(i)

    # ── 单环境接口（play.py / gui_play.py / pipeline Elo）────────────────────

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
            self.mcts.reset_env(0)

        return action, action_probs

    # ── 批量环境接口（client.py 分布式自对弈）────────────────────────────────

    def get_batch_action(self, current_boards, turns, temps=None):
        assert len(current_boards) == self.n_envs
        if temps is None:
            temps = [0 for _ in range(self.n_envs)]

        self.mcts.batch_playout(self.pv_fn, current_boards, turns)
        visits = self.mcts.get_visits_count()

        batch_actions = []
        batch_probs = []

        for i in range(self.n_envs):
            visit = visits[i]
            temp = temps[i]

            action_probs = np.zeros(self.n_actions, dtype=np.float32)
            valid_mask = visit > 0

            if not valid_mask.any():
                batch_actions.append(0)
                batch_probs.append(action_probs)
                continue

            action_probs[valid_mask] = visit[valid_mask] / visit[valid_mask].sum()

            if temp <= 1e-6:
                action = np.argmax(visit)
            else:
                valid_actions = np.where(valid_mask)[0]
                log_visits = np.log(visit[valid_mask])
                sample_dist = softmax(log_visits / temp)
                action = np.random.choice(valid_actions, p=sample_dist)

            batch_actions.append(action)
            batch_probs.append(action_probs)

        actions_array = np.array(batch_actions, dtype=np.int32)
        self.mcts.prune_roots(actions_array)
        return batch_actions, np.array(batch_probs)
