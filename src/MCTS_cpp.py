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
    def __init__(self, batch_size, c_init, c_base, discount, alpha, n_playout,
                 game_name='Connect4', board_converter=None, cache_size=0):
        backend_cls = _BACKENDS[game_name]
        self.mcts = backend_cls(batch_size, c_init, c_base, discount, alpha)
        self.n_playout = n_playout
        self.batch_size = batch_size
        self.action_size = backend_cls.action_size
        self.board_shape = backend_cls.board_shape
        self._convert_board = board_converter or _default_convert_board
        # cache_size=0 表示禁用置换表；>0 则启用 LRU 置换表
        self.cache = LRUCache(cache_size) if cache_size > 0 else None
        # 预分配 board 转换 buffer，避免 MCTS 热循环里频繁 malloc
        self._conv_buf = np.zeros((batch_size, 3, *self.board_shape), dtype=np.float32)

    def batch_playout(self, pv_func, current_boards, turns):
        """ current_boards: shape [batch_size, *board_shape], X: 1, O: -1
            turns: shape [batch_size, ], 1, -1"""
        current_boards = current_boards.astype(np.int8)
        turns = turns.astype(np.int32)

        for _ in range(self.n_playout):
            leaf_boards, term_vals, is_term, leaf_turns = self.mcts.search_batch(current_boards, turns)

            term_mask = is_term.astype(bool)
            values = term_vals.copy()

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
                    non_term_probs, non_term_vals = pv_func.predict(self._conv_buf[:n_non_term])
                    probs[non_term_mask] = non_term_probs
                    values[non_term_mask] = non_term_vals.flatten()
                else:
                    # 置换表启用：先查缓存，cache miss 的再批量送 NN
                    miss_indices = []
                    for i in non_term_indices:
                        # key = 棋盘原始字节 + turn 字节（43 bytes，含 turn 信息）
                        key = leaf_boards[i].tobytes() + leaf_turns[i].item().to_bytes(1, 'little', signed=True)
                        if key in self.cache:
                            p, v = self.cache.get(key)
                            probs[i] = p
                            values[i] = v
                        else:
                            miss_indices.append(i)

                    if miss_indices:
                        n_miss = len(miss_indices)
                        miss_boards = leaf_boards[miss_indices]
                        miss_turns  = leaf_turns[miss_indices]
                        conv = self._convert_board(miss_boards, miss_turns)
                        self._conv_buf[:n_miss] = conv
                        miss_probs, miss_vals = pv_func.predict(self._conv_buf[:n_miss])
                        miss_vals = miss_vals.flatten()
                        for j, i in enumerate(miss_indices):
                            probs[i]  = miss_probs[j]
                            values[i] = miss_vals[j]
                            key = leaf_boards[i].tobytes() + leaf_turns[i].item().to_bytes(1, 'little', signed=True)
                            self.cache.put(key, (miss_probs[j].copy(), miss_vals[j].item()))
                            # 覆盖 state 字段为 converted board，供 refresh_cache 批量重算
                            self.cache._od[key]['state'] = conv[j:j+1]

            self.mcts.backprop_batch(
                np.ascontiguousarray(probs, dtype=np.float32),
                np.ascontiguousarray(values, dtype=np.float32),
                is_term
            )
        return self

    def refresh_cache(self, pv_func):
        """网络权重更新后调用，用新 NN 重新计算缓存中所有条目的 prob 和 value"""
        if self.cache is None or len(self.cache) == 0:
            return self
        od = self.cache._od
        keys = list(od.keys())
        states = np.concatenate([od[k]['state'] for k in keys], axis=0)
        new_probs, new_vals = pv_func.predict(states)
        new_vals = new_vals.flatten()
        for j, k in enumerate(keys):
            od[k]['value'] = (new_probs[j].copy(), new_vals[j].item())
        return self

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
