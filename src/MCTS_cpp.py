import numpy as np
from src import mcts_cpp


class BatchedMCTS:
    def __init__(self, batch_size, c_init, c_base, discount, alpha, n_playout):
        self.mcts = mcts_cpp.BatchedMCTS(batch_size, c_init, c_base, discount, alpha)
        self.n_playout = n_playout
        self.batch_size = batch_size

    @staticmethod
    def _convert_board(board, turns):
        plane_x = (board == 1).astype(np.float32)
        plane_o = (board == -1).astype(np.float32)
        plane_turn = np.ones_like(board, dtype=np.float32) * turns[:, None, None]
        return np.stack([plane_x, plane_o, plane_turn], axis=1)

    def batch_playout(self, pv_func, current_boards, turns):
        """ current_boards: shape [batch_size, 6, 7], X: 1, O: -1
            turns: shape [batch_size, ], 1, -1"""
        current_boards = current_boards.astype(np.int8)
        turns = turns.astype(np.int32)

        for _ in range(self.n_playout):
            leaf_boards, _, is_term, leaf_turns = self.mcts.search_batch(current_boards, turns)

            probs, values = pv_func.predict(self._convert_board(leaf_boards, leaf_turns))
            self.mcts.backprop_batch(
                np.ascontiguousarray(probs, dtype=np.float32), 
                np.ascontiguousarray(values.flatten(), dtype=np.float32), 
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
        return np.array(counts).reshape(self.batch_size, -1)

    def get_mcts_probs(self):
        counts = self.get_visits_count()
        return counts / counts.sum(axis=1, keepdims=True)
