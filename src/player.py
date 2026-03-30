#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 10/Aug/2024  23:47
import numpy as np
from abc import abstractmethod, ABC
from .utils import softmax
from .MCTS_cpp import BatchedMCTS
from .symmetry import (get_sym_config, apply_sym_board, apply_sym_action,
                       inverse_sym_visits, inverse_sym_stats)


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
        mask = np.asarray(env.valid_mask(), dtype=bool)[None, :]
        probs = self.net.policy(env.current_state(), action_mask=mask)
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
    """Pure MCTS baseline (uniform prior + random rollout) — wraps C++ BatchedMCTS.

    Args:
        n_trees: Number of independent trees for root-parallel search.
            Each tree explores the same root position independently;
            visit counts are summed across all trees for action selection.
    """
    def __init__(self, c_puct=4, n_playout=1000, eps=0.0, game_name='Connect4', n_trees=1):
        super().__init__()
        self.n_trees = max(1, n_trees)
        self.mcts = BatchedMCTS(
            batch_size=self.n_trees, c_init=c_puct, c_base=500,
            alpha=0, n_playout=n_playout, game_name=game_name,
            noise_epsilon=0.0, fpu_reduction=0.0, use_symmetry=False,
        )

    def reset_player(self):
        for i in range(self.n_trees):
            self.mcts.reset_env(i)

    def get_action(self, env, *args, **kwargs):
        K = self.n_trees
        board = np.tile(env.board, (K, 1, 1))
        turns = np.full(K, env.turn, dtype=np.int32)
        self.mcts.rollout_playout(board, turns)
        visits = self.mcts.get_visits_count()  # (K, action_size)
        agg = visits.sum(axis=0) if K > 1 else visits[0]
        action = int(np.argmax(agg))
        self.reset_player()
        return action, None


class AlphaZeroPlayer(Player):
    """AlphaZero MCTS player — 支持单环境 (n_envs=1) 和批量并行 (n_envs>1) 两种模式。

    单环境模式：get_action(env, temp) — 用于 play.py / gui_play.py / pipeline Elo 评估
    批量模式：  get_batch_action(boards, turns, temps) — 用于 client.py 分布式自对弈

    Root-parallel (n_trees > 1):
        eval() 模式下保留最小噪声 (0.05) 以确保多棵树有不同的搜索路径。
        无噪声时 NN 模式下所有树会走完全相同的路径，root-parallel 退化。
    """

    _ROOT_PARALLEL_MIN_NOISE = 0.05  # eval 模式下 root-parallel 的最小噪声
    def __init__(self, policy_value_fn, n_envs=1, c_init=1.25, c_base=500,
                 n_playout=100, alpha=None, is_selfplay=0,
                 cache_size=0, noise_epsilon=0.25, fpu_reduction=0.4, use_symmetry=True,
                 game_name='Connect4', board_converter=None,
                 noise_steps=0, noise_eps_min=0.1,
                 mlh_slope=0.0, mlh_cap=0.2,
                 score_utility_factor=0.0, score_scale=8.0,
                 value_decay=1.0,
                 n_trees=1,
                 vl_batch=1,
                 sym_ensemble=False,
                 # 向后兼容：旧调用方用 eps= 而非 noise_epsilon=
                 eps=None):
        super().__init__()
        self.pv_fn = policy_value_fn
        self.is_selfplay = is_selfplay
        self.n_envs = n_envs
        self.n_trees = max(1, n_trees)
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
        self._score_utility_factor = score_utility_factor
        self._score_scale = score_scale
        self._value_decay = value_decay
        self._vl_batch = max(1, vl_batch)
        self._time_budget = None  # 秒；None 表示按 n_playout 固定次数搜索
        self.sym_ensemble = sym_ensemble

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

    def _sym_config(self):
        """返回当前游戏的对称配置。"""
        return get_sym_config(self._game_name)

    def _make_mcts(self, noise_eps):
        # sym_ensemble: 固定 K 棵树（K 种对称），关闭 C++ 层随机对称
        if self.sym_ensemble:
            cfg = self._sym_config()
            bs = len(cfg['sym_ids']) if cfg else 1
            use_sym = False
        elif self.n_envs == 1 and self.n_trees > 1:
            bs = self.n_trees
            use_sym = self._use_symmetry
        else:
            bs = self.n_envs
            use_sym = self._use_symmetry
        return BatchedMCTS(
            batch_size=bs, c_init=self._c_init, c_base=self._c_base,
            alpha=self._alpha, n_playout=self._n_playout,
            game_name=self._game_name, board_converter=self._board_converter,
            cache_size=self._cache_size, noise_epsilon=noise_eps,
            fpu_reduction=self._fpu_reduction, use_symmetry=use_sym,
            mlh_slope=self._mlh_slope, mlh_cap=self._mlh_cap,
            score_utility_factor=self._score_utility_factor,
            score_scale=self._score_scale,
            value_decay=self._value_decay,
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
        if self.pv_fn is not None and hasattr(self.pv_fn, 'train'):
            self.pv_fn.train()
        if self.mcts:
            self.mcts.set_noise_epsilon(self._noise_eps)

    def eval(self):
        if self.pv_fn is not None and hasattr(self.pv_fn, 'eval'):
            self.pv_fn.eval()
        if self.mcts:
            # sym_ensemble 不需要噪声：4 棵树看到不同棋盘，搜索路径自然不同
            if self.sym_ensemble:
                self.mcts.set_noise_epsilon(0.0)
            elif self.n_trees > 1:
                # Root-parallel 需要噪声来分化树的搜索路径
                self.mcts.set_noise_epsilon(self._ROOT_PARALLEL_MIN_NOISE)
            else:
                self.mcts.set_noise_epsilon(0.0)

    def reset_player(self):
        if self.mcts:
            if self.sym_ensemble:
                cfg = self._sym_config()
                bs = len(cfg['sym_ids']) if cfg else 1
            elif self.n_envs == 1 and self.n_trees > 1:
                bs = self.n_trees
            else:
                bs = self.n_envs
            for i in range(bs):
                self.mcts.reset_env(i)

    # ── 单环境接口（play.py / gui_play.py / pipeline Elo）────────────────────

    def get_action(self, env, temp=0):
        if self.sym_ensemble:
            return self._get_action_sym_ensemble(env, temp)

        K = self.n_trees
        board = np.tile(env.board, (K, 1, 1))    # (K, H, W)
        turns = np.full(K, env.turn, dtype=np.int32)

        self.mcts.batch_playout(self.pv_fn, board, turns,
                                vl_batch=self._vl_batch, time_budget=self._time_budget)
        all_visits = self.mcts.get_visits_count()  # (K, action_size)
        visits = all_visits.sum(axis=0) if K > 1 else all_visits[0]

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
            self.mcts.prune_roots(np.full(K, action, dtype=np.int32))
        else:
            for i in range(K):
                self.mcts.reset_env(i)

        return action, action_probs

    def _get_action_sym_ensemble(self, env, temp=0):
        """对称并行搜索：K 种对称同时搜索，逆变换后合并 visit counts。"""
        cfg = self._sym_config()
        sym_ids = cfg['sym_ids']
        game = self._game_name
        K = len(sym_ids)
        boards = np.stack([apply_sym_board(env.board, s, game) for s in sym_ids])
        turns = np.full(K, env.turn, dtype=np.int32)

        self.mcts.batch_playout(self.pv_fn, boards, turns,
                                vl_batch=self._vl_batch, time_budget=self._time_budget)
        all_visits = self.mcts.get_visits_count()  # (K, action_size)

        # 逆变换每棵树的 visits 回原始坐标
        for i, sym_id in enumerate(sym_ids):
            all_visits[i] = inverse_sym_visits(all_visits[i], sym_id, game)

        visits = all_visits.sum(axis=0)

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

        # 各树的 action 需按对称映射后再 prune
        if self.is_selfplay:
            sym_actions = np.array(
                [apply_sym_action(action, s, game) for s in sym_ids], dtype=np.int32)
            self.mcts.prune_roots(sym_actions)
        else:
            for i in range(K):
                self.mcts.reset_env(i)

        return action, action_probs

    # ── 批量环境接口（client.py 分布式自对弈）────────────────────────────────

    def get_batch_action(self, current_boards, turns, temps=None):
        assert len(current_boards) == self.n_envs
        if temps is None:
            temps = [0 for _ in range(self.n_envs)]

        self.mcts.batch_playout(self.pv_fn, current_boards, turns,
                                vl_batch=self._vl_batch, time_budget=self._time_budget)
        visits = self.mcts.get_visits_count()
        # 获取根节点 WDL 分布（绝对视角：draw, p1_win, p2_win）
        root_stats = self.mcts.get_root_stats()
        root_wdls = np.stack([root_stats['root_D'], root_stats['root_P1W'], root_stats['root_P2W']], axis=1)  # (n_envs, 3)

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
        return batch_actions, np.array(batch_probs), root_wdls
