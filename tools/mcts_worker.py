"""
轻量 MCTS worker — 不 import torch，专供多进程超参扫描。
"""
import sys, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.environments.Connect4.env import Env
from src.MCTS import MCTS_AZ
from src.utils import evaluate_rollout


class RolloutAdapter:
    n_actions = 7

    def predict(self, state):
        env = Env()
        env.board = (state[0, 0] - state[0, 1]).astype(np.float32)
        env.turn = 1 if state[0, 2, 0, 0] > 0 else -1
        probs = np.ones((1, self.n_actions), dtype=np.float32) / self.n_actions
        val = evaluate_rollout(env.copy())
        return probs, np.array([[val]], dtype=np.float32)


def run_single(board, turn, c_init, alpha, fpu_reduction, n_playout, discount=1.0):
    """单次 MCTS 运行 (子进程 worker)"""
    env = Env()
    env.board = board.copy()
    env.turn = turn
    adapter = RolloutAdapter()
    mcts = MCTS_AZ(
        policy_value_fn=adapter,
        c_init=c_init,
        n_playout=n_playout,
        discount=discount,
        alpha=alpha,
        cache_size=0,
        eps=0.25,
        fpu_reduction=fpu_reduction,
        use_symmetry=False,
    )
    mcts.c_base = 100000
    if alpha is None:
        mcts.eval()
    list(mcts.get_action_visits(env.copy()))
    total = sum(c.n_visits for c in mcts.root.children.values())
    col_pcts = {}
    col_qs = {}
    for a, nd in mcts.root.children.items():
        col_pcts[a] = nd.n_visits / total * 100
        col_qs[a] = nd.Q
    return col_pcts, col_qs
