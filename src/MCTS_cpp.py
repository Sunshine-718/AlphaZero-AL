import numpy as np
from src import mcts_cpp


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
                 game_name='Connect4', board_converter=None):
        backend_cls = _BACKENDS[game_name]
        self.mcts = backend_cls(batch_size, c_init, c_base, discount, alpha)
        self.n_playout = n_playout
        self.batch_size = batch_size
        self.action_size = backend_cls.action_size
        self.board_shape = backend_cls.board_shape
        self._convert_board = board_converter or _default_convert_board

    def batch_playout(self, pv_func, current_boards, turns):
        """ current_boards: shape [batch_size, *board_shape], X: 1, O: -1
            turns: shape [batch_size, ], 1, -1"""
        current_boards = current_boards.astype(np.int8)
        turns = turns.astype(np.int32)

        for _ in range(self.n_playout):
            leaf_boards, term_vals, is_term, leaf_turns = self.mcts.search_batch(current_boards, turns)

            term_mask = is_term.astype(bool)
            values = term_vals.copy()

            # 只对非终局 leaf 调用 NN
            non_term_mask = ~term_mask
            if non_term_mask.any():
                non_term_converted = self._convert_board(leaf_boards[non_term_mask], leaf_turns[non_term_mask])
                non_term_probs, non_term_vals = pv_func.predict(non_term_converted)

                probs = np.zeros((self.batch_size, self.action_size), dtype=np.float32)
                probs[non_term_mask] = non_term_probs
                values[non_term_mask] = non_term_vals.flatten()
            else:
                probs = np.zeros((self.batch_size, self.action_size), dtype=np.float32)

            self.mcts.backprop_batch(
                np.ascontiguousarray(probs, dtype=np.float32),
                np.ascontiguousarray(values, dtype=np.float32),
                is_term
            )
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
