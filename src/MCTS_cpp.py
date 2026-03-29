import time

import numpy as np
from src import mcts_cpp
from src.Cache import LRUCache


# 游戏名 → C++ 后端类的映射
_BACKENDS = {
    'Connect4': mcts_cpp.BatchedMCTS_Connect4,
    'Othello': mcts_cpp.BatchedMCTS_Othello,
}


def _default_convert_board(board, turns):
    """Default 3-plane conversion in relative perspective."""
    plane_x = (board == turns[:, None, None]).astype(np.float32)
    plane_o = (board == -turns[:, None, None]).astype(np.float32)
    plane_turn = np.ones_like(board, dtype=np.float32) * turns[:, None, None]
    return np.stack([plane_x, plane_o, plane_turn], axis=1)


def _relative_wdl_to_absolute(wdl_rel, turns):
    """Convert relative WDL [draw, win(to-move), loss(to-move)] to absolute [draw, p1w, p2w]."""
    d = wdl_rel[:, 0]
    w = wdl_rel[:, 1]
    l = wdl_rel[:, 2]
    p1w = np.where(turns == 1, w, l)
    p2w = np.where(turns == 1, l, w)
    return d, p1w, p2w


class BatchedMCTS:
    def __init__(self, batch_size, c_init, c_base, alpha, n_playout,
                 game_name='Connect4', board_converter=None, cache_size=0, noise_epsilon=0.25, fpu_reduction=0.4,
                 use_symmetry=True, mlh_slope=0.0, mlh_cap=0.2, value_decay=1.0,
                 score_utility_factor=0.0, score_scale=8.0):
        backend_cls = _BACKENDS[game_name]
        self.mcts = backend_cls(batch_size)

        # 通过 config 属性设置搜索参数
        cfg = self.mcts.config
        cfg.c_init = c_init
        cfg.c_base = c_base
        cfg.dirichlet_alpha = alpha
        cfg.noise_epsilon = noise_epsilon
        cfg.fpu_reduction = fpu_reduction
        cfg.use_symmetry = use_symmetry
        cfg.mlh_slope = mlh_slope
        cfg.mlh_cap = mlh_cap
        cfg.score_utility_factor = score_utility_factor
        cfg.score_scale = score_scale
        cfg.value_decay = value_decay

        self.n_playout = n_playout
        self.batch_size = batch_size
        self.action_size = backend_cls.action_size
        self.board_shape = backend_cls.board_shape
        self._convert_board = board_converter or _default_convert_board
        # cache_size=0 表示禁用置换表；>0 则启用 LRU 置换表
        self.cache = LRUCache(cache_size) if cache_size > 0 else None
        # 预分配 board 转换 buffer，避免 MCTS 热循环里频繁 malloc
        self._conv_buf = np.zeros((batch_size, 3, *self.board_shape), dtype=np.float32)
        self._game_name = game_name
        self._rollout_eval = None

    def _should_early_exit(self, step, remaining_steps):
        """判断搜索是否已收敛：最优动作的访问数领先第二名超过剩余步数。

        Args:
            step: 已完成的模拟步数
            remaining_steps: 剩余可用模拟步数（float，可以是估算值）
        Returns:
            True if best action is unreachable by runner-up
        """
        if step < 8:
            return False
        counts = np.array(self.mcts.get_all_counts()).reshape(self.batch_size, self.action_size)
        # 对每个 env 取 top-2 访问数
        top2_idx = np.argpartition(counts, -2, axis=1)[:, -2:]
        top2 = np.take_along_axis(counts, top2_idx, axis=1)
        best = top2.max(axis=1)
        second = top2.min(axis=1)
        return bool(np.all(best - second > remaining_steps))

    def batch_playout(self, pv_func, current_boards, turns, n_playout=None,
                      vl_batch=1, time_budget=None):
        """批量 MCTS 搜索。

        Args:
            pv_func: 神经网络 predict 函数
            current_boards: shape [batch_size, *board_shape], X=1, O=-1
            turns: shape [batch_size,], 1 or -1
            n_playout: 固定模拟次数（与 time_budget 二选一）
            vl_batch: Virtual Loss 批量大小（1=原路径，>1=VL 树并行）
            time_budget: 搜索时间预算（秒）。设置后按时间搜索，n_playout 作为上限
        """
        current_boards = current_boards.astype(np.int8)
        turns = turns.astype(np.int32)
        max_n = n_playout if n_playout is not None else self.n_playout
        use_time = time_budget is not None and time_budget > 0

        # Sync score_scale to network so Python-side atan mapping matches C++ terminal_aux
        if hasattr(pv_func, 'score_scale'):
            pv_func.score_scale = self.mcts.config.score_scale

        if vl_batch <= 1:
            # === 非 VL 路径 ===
            t0 = time.perf_counter() if use_time else 0.0
            for step in range(max_n):
                leaf_boards, term_d, term_p1w, term_p2w, is_term, leaf_turns, valid_masks = \
                    self.mcts.search_batch(current_boards, turns)

                term_mask = is_term.astype(bool)
                d_vals = term_d.copy()
                p1w_vals = term_p1w.copy()
                p2w_vals = term_p2w.copy()
                moves_left = np.zeros(self.batch_size, dtype=np.float32)

                # 只对非终局 leaf 调用 NN（或查置换表）
                non_term_mask = ~term_mask
                probs = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
                if non_term_mask.any():
                    non_term_indices = np.where(non_term_mask)[0]
                    n_non_term = len(non_term_indices)
                    if self.cache is None:
                        # 置换表未启用：全部送 NN，写入预分配 buffer
                        conv = self._convert_board(leaf_boards[non_term_mask], leaf_turns[non_term_mask])
                        self._conv_buf[:n_non_term] = conv
                        non_term_masks = valid_masks[non_term_mask].astype(bool, copy=False)
                        non_term_probs, non_term_wdl, non_term_ml = pv_func.predict(
                            self._conv_buf[:n_non_term], action_mask=non_term_masks
                        )
                        probs[non_term_mask] = non_term_probs
                        # predict() returns relative WDL and is converted to absolute before backprop
                        d_abs, p1w_abs, p2w_abs = _relative_wdl_to_absolute(
                            non_term_wdl, leaf_turns[non_term_mask]
                        )
                        d_vals[non_term_mask] = d_abs
                        p1w_vals[non_term_mask] = p1w_abs
                        p2w_vals[non_term_mask] = p2w_abs
                        moves_left[non_term_mask] = non_term_ml.flatten()
                    else:
                        # 置换表启用：先查缓存，cache miss 的再批量送 NN
                        miss_indices = []
                        for i in non_term_indices:
                            key = leaf_boards[i].tobytes() + leaf_turns[i].item().to_bytes(1, 'little', signed=True)
                            if key in self.cache:
                                p, wdl, ml = self.cache.get(key)
                                probs[i] = p
                                d_vals[i] = wdl[0]
                                if leaf_turns[i] == 1:
                                    p1w_vals[i] = wdl[1]
                                    p2w_vals[i] = wdl[2]
                                else:
                                    p1w_vals[i] = wdl[2]
                                    p2w_vals[i] = wdl[1]
                                moves_left[i] = ml
                            else:
                                miss_indices.append(i)

                        if miss_indices:
                            n_miss = len(miss_indices)
                            miss_boards = leaf_boards[miss_indices]
                            miss_turns  = leaf_turns[miss_indices]
                            conv = self._convert_board(miss_boards, miss_turns)
                            self._conv_buf[:n_miss] = conv
                            miss_masks = valid_masks[miss_indices].astype(bool, copy=False)
                            miss_probs, miss_wdl, miss_ml = pv_func.predict(
                                self._conv_buf[:n_miss], action_mask=miss_masks
                            )
                            miss_ml = miss_ml.flatten()
                            for j, i in enumerate(miss_indices):
                                probs[i]  = miss_probs[j]
                                d_vals[i] = miss_wdl[j, 0]
                                if miss_turns[j] == 1:
                                    p1w_vals[i] = miss_wdl[j, 1]
                                    p2w_vals[i] = miss_wdl[j, 2]
                                else:
                                    p1w_vals[i] = miss_wdl[j, 2]
                                    p2w_vals[i] = miss_wdl[j, 1]
                                moves_left[i] = miss_ml[j]
                                key = leaf_boards[i].tobytes() + leaf_turns[i].item().to_bytes(1, 'little', signed=True)
                                self.cache.put(key, (miss_probs[j].copy(), miss_wdl[j].copy(), miss_ml[j].item()))
                                self.cache._od[key]['state'] = conv[j:j+1]
                                self.cache._od[key]['valid_mask'] = miss_masks[j:j+1].copy()

                self.mcts.backprop_batch(
                    np.ascontiguousarray(probs, dtype=np.float32),
                    np.ascontiguousarray(d_vals, dtype=np.float32),
                    np.ascontiguousarray(p1w_vals, dtype=np.float32),
                    np.ascontiguousarray(p2w_vals, dtype=np.float32),
                    np.ascontiguousarray(moves_left, dtype=np.float32),
                    is_term
                )

                # Early exit check
                if use_time:
                    done = step + 1
                    elapsed = time.perf_counter() - t0
                    if elapsed >= time_budget:
                        break
                    avg_step = elapsed / done
                    remaining_steps = (time_budget - elapsed) / avg_step
                    if self._should_early_exit(done, remaining_steps):
                        break
        else:
            # === VL 路径：K 次 VL 模拟 × N 棵树 = N*K 个叶节点/迭代 ===
            K = vl_batch
            remaining = max_n
            t0 = time.perf_counter() if use_time else 0.0
            total_sims = 0  # 已完成的模拟数（含 warm-up）

            # Warm-up：先做 1 次非 VL 迭代确保所有根节点已展开
            # 避免 VL 首批 K 次模拟全部落在未展开的根节点上浪费 K-1 次
            if remaining > 0:
                leaf_boards, term_d, term_p1w, term_p2w, is_term, leaf_turns, valid_masks = \
                    self.mcts.search_batch(current_boards, turns)
                probs = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
                d_vals = term_d.copy()
                p1w_vals = term_p1w.copy()
                p2w_vals = term_p2w.copy()
                moves_left = np.zeros(self.batch_size, dtype=np.float32)
                non_term_mask = ~is_term.astype(bool)
                if non_term_mask.any():
                    conv = self._convert_board(leaf_boards[non_term_mask], leaf_turns[non_term_mask])
                    nn_probs, nn_wdl, nn_ml = pv_func.predict(
                        conv, action_mask=valid_masks[non_term_mask].astype(bool, copy=False)
                    )
                    probs[non_term_mask] = nn_probs
                    d_abs, p1w_abs, p2w_abs = _relative_wdl_to_absolute(
                        nn_wdl, leaf_turns[non_term_mask])
                    d_vals[non_term_mask] = d_abs
                    p1w_vals[non_term_mask] = p1w_abs
                    p2w_vals[non_term_mask] = p2w_abs
                    moves_left[non_term_mask] = nn_ml.flatten()
                self.mcts.backprop_batch(
                    np.ascontiguousarray(probs, dtype=np.float32),
                    np.ascontiguousarray(d_vals, dtype=np.float32),
                    np.ascontiguousarray(p1w_vals, dtype=np.float32),
                    np.ascontiguousarray(p2w_vals, dtype=np.float32),
                    np.ascontiguousarray(moves_left, dtype=np.float32),
                    is_term)
                remaining -= 1
                total_sims += 1

            while remaining > 0:
                # 时间预算模式：检查是否已超时
                if use_time:
                    elapsed = time.perf_counter() - t0
                    if elapsed >= time_budget:
                        break
                    # 估算剩余可用模拟数用于 early exit
                    if total_sims > 0:
                        avg_sim = elapsed / total_sims
                        remaining_by_time = (time_budget - elapsed) / avg_sim
                        if self._should_early_exit(total_sims, remaining_by_time):
                            break

                cur_K = min(K, remaining)
                remaining -= cur_K
                cur_total = self.batch_size * cur_K

                # VL Selection：每棵树 cur_K 次模拟，返回 N*cur_K 个叶节点
                leaf_boards, term_d, term_p1w, term_p2w, is_term, leaf_turns, sym_ids, valid_masks = \
                    self.mcts.search_batch_vl(cur_K, current_boards, turns)

                # try/finally 确保 VL 被清除：若 NN 推理抛异常，
                # backprop_batch_vl 内的 remove_all_vl 不会执行，
                # 导致 n_inflight 永久残留在树中
                try:
                    term_mask = is_term.astype(bool)
                    d_vals = term_d.copy()
                    p1w_vals = term_p1w.copy()
                    p2w_vals = term_p2w.copy()
                    moves_left = np.zeros(cur_total, dtype=np.float32)

                    non_term_mask = ~term_mask
                    probs = np.zeros((cur_total, self.action_size), dtype=np.float32)

                    if non_term_mask.any():
                        if self.cache is None:
                            conv = self._convert_board(leaf_boards[non_term_mask], leaf_turns[non_term_mask])
                            nn_probs, nn_wdl, nn_ml = pv_func.predict(
                                conv, action_mask=valid_masks[non_term_mask].astype(bool, copy=False)
                            )
                            probs[non_term_mask] = nn_probs
                            d_abs, p1w_abs, p2w_abs = _relative_wdl_to_absolute(
                                nn_wdl, leaf_turns[non_term_mask]
                            )
                            d_vals[non_term_mask] = d_abs
                            p1w_vals[non_term_mask] = p1w_abs
                            p2w_vals[non_term_mask] = p2w_abs
                            moves_left[non_term_mask] = nn_ml.flatten()
                        else:
                            non_term_indices = np.where(non_term_mask)[0]
                            miss_indices = []
                            for i in non_term_indices:
                                key = leaf_boards[i].tobytes() + leaf_turns[i].item().to_bytes(1, 'little', signed=True)
                                if key in self.cache:
                                    p, wdl, ml = self.cache.get(key)
                                    probs[i] = p
                                    d_vals[i] = wdl[0]
                                    if leaf_turns[i] == 1:
                                        p1w_vals[i] = wdl[1]
                                        p2w_vals[i] = wdl[2]
                                    else:
                                        p1w_vals[i] = wdl[2]
                                        p2w_vals[i] = wdl[1]
                                    moves_left[i] = ml
                                else:
                                    miss_indices.append(i)

                            if miss_indices:
                                miss_boards = leaf_boards[miss_indices]
                                miss_turns  = leaf_turns[miss_indices]
                                conv = self._convert_board(miss_boards, miss_turns)
                                miss_masks = valid_masks[miss_indices].astype(bool, copy=False)
                                miss_probs, miss_wdl, miss_ml = pv_func.predict(
                                    conv, action_mask=miss_masks
                                )
                                miss_ml = miss_ml.flatten()
                                for j, i in enumerate(miss_indices):
                                    probs[i]  = miss_probs[j]
                                    d_vals[i] = miss_wdl[j, 0]
                                    if miss_turns[j] == 1:
                                        p1w_vals[i] = miss_wdl[j, 1]
                                        p2w_vals[i] = miss_wdl[j, 2]
                                    else:
                                        p1w_vals[i] = miss_wdl[j, 2]
                                        p2w_vals[i] = miss_wdl[j, 1]
                                    moves_left[i] = miss_ml[j]
                                    key = leaf_boards[i].tobytes() + leaf_turns[i].item().to_bytes(1, 'little', signed=True)
                                    self.cache.put(key, (miss_probs[j].copy(), miss_wdl[j].copy(), miss_ml[j].item()))
                                    self.cache._od[key]['state'] = conv[j:j+1]
                                    self.cache._od[key]['valid_mask'] = miss_masks[j:j+1].copy()

                    self.mcts.backprop_batch_vl(
                        cur_K,
                        np.ascontiguousarray(probs, dtype=np.float32),
                        np.ascontiguousarray(d_vals, dtype=np.float32),
                        np.ascontiguousarray(p1w_vals, dtype=np.float32),
                        np.ascontiguousarray(p2w_vals, dtype=np.float32),
                        np.ascontiguousarray(moves_left, dtype=np.float32),
                        is_term,
                        sym_ids
                    )
                except BaseException:
                    # NN 推理失败 / KeyboardInterrupt / SystemExit：
                    # 手动清除所有残留的 VL (n_inflight)，防止树状态永久污染
                    self.mcts.remove_all_vl(cur_K)
                    raise

                total_sims += cur_K

        return self

    def refresh_cache(self, pv_func):
        """网络权重更新后调用，用新 NN 重新计算缓存中所有条目的 prob, wdl, moves_left"""
        if self.cache is None or len(self.cache) == 0:
            return self
        if hasattr(pv_func, 'score_scale'):
            pv_func.score_scale = self.mcts.config.score_scale
        od = self.cache._od
        keys = list(od.keys())
        states = np.concatenate([od[k]['state'] for k in keys], axis=0)
        valid_masks = None
        if all('valid_mask' in od[k] for k in keys):
            valid_masks = np.concatenate([od[k]['valid_mask'] for k in keys], axis=0)
        new_probs, new_wdl, new_ml = pv_func.predict(states, action_mask=valid_masks)
        new_ml = new_ml.flatten()
        for j, k in enumerate(keys):
            od[k]['value'] = (new_probs[j].copy(), new_wdl[j].copy(), new_ml[j].item())
        return self

    def _get_rollout_evaluator(self):
        """懒加载 C++ RolloutEvaluator 实例"""
        if self._rollout_eval is None:
            re_cls_name = f'RolloutEvaluator_{self._game_name}'
            self._rollout_eval = getattr(mcts_cpp, re_cls_name)()
        return self._rollout_eval

    def rollout_playout(self, current_boards, turns):
        """纯 MCTS：使用 RolloutEvaluator 在 C++ 内完成整个搜索"""
        current_boards = current_boards.astype(np.int8)
        turns = turns.astype(np.int32)
        evaluator = self._get_rollout_evaluator()
        self.mcts.search(evaluator, current_boards, turns, self.n_playout)
        return self

    def set_noise_epsilon(self, eps):
        self.mcts.config.noise_epsilon = eps

    def set_mlh_params(self, slope, cap):
        cfg = self.mcts.config
        cfg.mlh_slope = slope
        cfg.mlh_cap = cap

    def set_score_utility_params(self, factor, scale):
        cfg = self.mcts.config
        old_scale = cfg.score_scale
        cfg.score_utility_factor = factor
        cfg.score_scale = scale
        # score_scale 变更后缓存中的 aux 值失效（基于旧 scale 的 atan 映射）
        if scale != old_scale and self.cache is not None and len(self.cache) > 0:
            self.cache._od.clear()

    def set_c_init(self, val):
        self.mcts.config.c_init = val

    def set_c_base(self, val):
        self.mcts.config.c_base = val

    def set_alpha(self, val):
        self.mcts.config.dirichlet_alpha = val

    def set_fpu_reduction(self, val):
        self.mcts.config.fpu_reduction = val

    def set_use_symmetry(self, val):
        self.mcts.config.use_symmetry = val

    def set_value_decay(self, val):
        self.mcts.config.value_decay = val

    def reset_env(self, index):
        self.mcts.reset_env(index)
        return self

    def seed(self, seed):
        self.mcts.set_seed(seed)

    def prune_roots(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        self.mcts.prune_roots(actions)
        return self

    def get_visits_count(self):
        counts = self.mcts.get_all_counts()
        return np.array(counts).reshape(self.batch_size, self.action_size)

    def get_mcts_probs(self):
        counts = self.get_visits_count()
        return counts / counts.sum(axis=1, keepdims=True)

    def get_root_stats(self):
        """返回所有 env 的 root 节点统计信息（绝对视角）。

        Returns:
            dict with keys:
                'root_N':    (batch_size,)            root 访问次数
                'root_Q':    (batch_size,)            root Q 值（当前落子方视角）
                'root_M':    (batch_size,)            root auxiliary estimate
                'root_D':    (batch_size,)            root 和棋率（绝对）
                'root_P1W':  (batch_size,)            root P1 胜率（绝对）
                'root_P2W':  (batch_size,)            root P2 胜率（绝对）
                'N':         (batch_size, action_size)  各 action 子节点访问次数
                'Q':         (batch_size, action_size)  各 action 子节点 Q 值
                'prior':     (batch_size, action_size)  各 action NN 先验概率
                'noise':     (batch_size, action_size)  各 action Dirichlet 噪声
                'M':         (batch_size, action_size)  child auxiliary estimate
                'D':         (batch_size, action_size)  各 action 和棋率（绝对）
                'P1W':       (batch_size, action_size)  各 action P1 胜率（绝对）
                'P2W':       (batch_size, action_size)  各 action P2 胜率（绝对）
        """
        raw = self.mcts.get_all_root_stats()  # (batch_size, 6 + action_size*8)
        B, A = self.batch_size, self.action_size
        # root_M / M are game-specific auxiliary predictions:
        # Connect4 = remaining plies, Othello = terminal disc diff in root perspective.

        root_info = raw[:, :6]
        children = raw[:, 6:].reshape(B, A, 8)

        return {
            'root_N':   root_info[:, 0],
            'root_Q':   root_info[:, 1],
            'root_M':   root_info[:, 2],
            'root_D':   root_info[:, 3],
            'root_P1W': root_info[:, 4],
            'root_P2W': root_info[:, 5],
            'N':        children[:, :, 0],
            'Q':        children[:, :, 1],
            'prior':    children[:, :, 2],
            'noise':    children[:, :, 3],
            'M':        children[:, :, 4],
            'D':        children[:, :, 5],
            'P1W':      children[:, :, 6],
            'P2W':      children[:, :, 7],
        }
