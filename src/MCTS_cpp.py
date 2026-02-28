import numpy as np
from src import mcts_cpp
from src.Cache import LRUCache


# 游戏名 → C++ 后端类的映射
_BACKENDS = {
    'Connect4': mcts_cpp.BatchedMCTS_Connect4,
    # 添加新游戏:
    # 'TicTacToe': mcts_cpp.BatchedMCTS_TicTacToe,
}


def _default_convert_board(board, turns):
    """默认 3 通道转换: player1 平面, player2 平面, turn 平面"""
    plane_x = (board == 1).astype(np.float32)
    plane_o = (board == -1).astype(np.float32)
    plane_turn = np.ones_like(board, dtype=np.float32) * turns[:, None, None]
    return np.stack([plane_x, plane_o, plane_turn], axis=1)


class BatchedMCTS:
    def __init__(self, batch_size, c_init, c_base, alpha, n_playout,
                 game_name='Connect4', board_converter=None, cache_size=0, noise_epsilon=0.25, fpu_reduction=0.4,
                 use_symmetry=True, mlh_slope=0.0, mlh_cap=0.2, mlh_threshold=0.8):
        backend_cls = _BACKENDS[game_name]
        self.mcts = backend_cls(batch_size, c_init, c_base, alpha, noise_epsilon, fpu_reduction, use_symmetry,
                                mlh_slope, mlh_cap, mlh_threshold)
        self.n_playout = n_playout
        self.batch_size = batch_size
        self.action_size = backend_cls.action_size
        self.board_shape = backend_cls.board_shape
        self._convert_board = board_converter or _default_convert_board
        # cache_size=0 表示禁用置换表；>0 则启用 LRU 置换表
        self.cache = LRUCache(cache_size) if cache_size > 0 else None
        # 预分配 board 转换 buffer，避免 MCTS 热循环里频繁 malloc
        self._conv_buf = np.zeros((batch_size, 3, *self.board_shape), dtype=np.float32)

    def batch_playout(self, pv_func, current_boards, turns, n_playout=None):
        """ current_boards: shape [batch_size, *board_shape], X: 1, O: -1
            turns: shape [batch_size, ], 1, -1
            n_playout: override self.n_playout if provided"""
        current_boards = current_boards.astype(np.int8)
        turns = turns.astype(np.int32)
        n = n_playout if n_playout is not None else self.n_playout

        for _ in range(n):
            leaf_boards, term_d, term_p1w, term_p2w, is_term, leaf_turns = self.mcts.search_batch(current_boards, turns)

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
                    non_term_probs, non_term_wdl, non_term_ml = pv_func.predict(self._conv_buf[:n_non_term])
                    probs[non_term_mask] = non_term_probs
                    # predict() 返回绝对视角 (d, p1w, p2w)
                    d_vals[non_term_mask] = non_term_wdl[:, 0]
                    p1w_vals[non_term_mask] = non_term_wdl[:, 1]
                    p2w_vals[non_term_mask] = non_term_wdl[:, 2]
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
                            p1w_vals[i] = wdl[1]
                            p2w_vals[i] = wdl[2]
                            moves_left[i] = ml
                        else:
                            miss_indices.append(i)

                    if miss_indices:
                        n_miss = len(miss_indices)
                        miss_boards = leaf_boards[miss_indices]
                        miss_turns  = leaf_turns[miss_indices]
                        conv = self._convert_board(miss_boards, miss_turns)
                        self._conv_buf[:n_miss] = conv
                        miss_probs, miss_wdl, miss_ml = pv_func.predict(self._conv_buf[:n_miss])
                        miss_ml = miss_ml.flatten()
                        for j, i in enumerate(miss_indices):
                            probs[i]  = miss_probs[j]
                            d_vals[i] = miss_wdl[j, 0]
                            p1w_vals[i] = miss_wdl[j, 1]
                            p2w_vals[i] = miss_wdl[j, 2]
                            moves_left[i] = miss_ml[j]
                            key = leaf_boards[i].tobytes() + leaf_turns[i].item().to_bytes(1, 'little', signed=True)
                            self.cache.put(key, (miss_probs[j].copy(), miss_wdl[j].copy(), miss_ml[j].item()))
                            self.cache._od[key]['state'] = conv[j:j+1]

            self.mcts.backprop_batch(
                np.ascontiguousarray(probs, dtype=np.float32),
                np.ascontiguousarray(d_vals, dtype=np.float32),
                np.ascontiguousarray(p1w_vals, dtype=np.float32),
                np.ascontiguousarray(p2w_vals, dtype=np.float32),
                np.ascontiguousarray(moves_left, dtype=np.float32),
                is_term
            )
        return self

    def refresh_cache(self, pv_func):
        """网络权重更新后调用，用新 NN 重新计算缓存中所有条目的 prob, wdl, moves_left"""
        if self.cache is None or len(self.cache) == 0:
            return self
        od = self.cache._od
        keys = list(od.keys())
        states = np.concatenate([od[k]['state'] for k in keys], axis=0)
        new_probs, new_wdl, new_ml = pv_func.predict(states)
        new_ml = new_ml.flatten()
        for j, k in enumerate(keys):
            od[k]['value'] = (new_probs[j].copy(), new_wdl[j].copy(), new_ml[j].item())
        return self

    def rollout_playout(self, current_boards, turns):
        """纯 MCTS：整个 playout 循环在 C++ 内完成（random rollout + uniform prior）"""
        current_boards = current_boards.astype(np.int8)
        turns = turns.astype(np.int32)
        self.mcts.rollout_playout(current_boards, turns, self.n_playout)
        return self

    def set_noise_epsilon(self, eps):
        self.mcts.set_noise_epsilon(eps)

    def set_mlh_params(self, slope, cap, threshold):
        self.mcts.set_mlh_params(slope, cap, threshold)

    def set_c_init(self, val):
        self.mcts.set_c_init(val)

    def set_c_base(self, val):
        self.mcts.set_c_base(val)

    def set_alpha(self, val):
        self.mcts.set_alpha(val)

    def set_fpu_reduction(self, val):
        self.mcts.set_fpu_reduction(val)

    def set_use_symmetry(self, val):
        self.mcts.set_use_symmetry(val)

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
                'root_M':    (batch_size,)            root 预期剩余步数
                'root_D':    (batch_size,)            root 和棋率（绝对）
                'root_P1W':  (batch_size,)            root P1 胜率（绝对）
                'root_P2W':  (batch_size,)            root P2 胜率（绝对）
                'N':         (batch_size, action_size)  各 action 子节点访问次数
                'Q':         (batch_size, action_size)  各 action 子节点 Q 值
                'prior':     (batch_size, action_size)  各 action NN 先验概率
                'noise':     (batch_size, action_size)  各 action Dirichlet 噪声
                'M':         (batch_size, action_size)  各 action 预期剩余步数
                'D':         (batch_size, action_size)  各 action 和棋率（绝对）
                'P1W':       (batch_size, action_size)  各 action P1 胜率（绝对）
                'P2W':       (batch_size, action_size)  各 action P2 胜率（绝对）
        """
        raw = self.mcts.get_all_root_stats()  # (batch_size, 6 + action_size*8)
        B, A = self.batch_size, self.action_size

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
