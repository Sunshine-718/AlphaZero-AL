import numpy as np
from typing import List, Tuple


class BatchedMCTS:
    def __init__(self, n_envs: int) -> None: ...

    def reset_env(self, env_idx: int) -> None: ...

    def set_seed(self, seed: int) -> None: ...

    def get_all_counts(self) -> List[int]: ...

    def prune_roots(self, actions: np.ndarray) -> None: ...

    def search_batch(
        self,
        input_boards: np.ndarray,
        turns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    def backprop_batch(
        self,
        policy_logits: np.ndarray,
        d_vals: np.ndarray,
        p1w_vals: np.ndarray,
        p2w_vals: np.ndarray,
        moves_left: np.ndarray,
        is_term: np.ndarray,
    ) -> None: ...

    def remove_all_vl(self, K: int) -> None: ...

    def search_batch_vl(
        self,
        K: int,
        input_boards: np.ndarray,
        turns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

    def backprop_batch_vl(
        self,
        K: int,
        policy_logits: np.ndarray,
        d_vals: np.ndarray,
        p1w_vals: np.ndarray,
        p2w_vals: np.ndarray,
        moves_left: np.ndarray,
        is_term: np.ndarray,
        sym_ids: np.ndarray,
    ) -> None: ...

    def get_all_root_stats(self) -> np.ndarray: ...
