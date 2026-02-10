import numpy as np
from typing import Tuple, List, Union

class BatchedMCTS:
    def __init__(self, 
                 n_envs: int, 
                 c_init: float, 
                 c_base: float, 
                 discount: float, 
                 alpha: float) -> None:
        """
        初始化 BatchedMCTS 引擎。
        
        Args:
            n_envs: 并行环境数量 (Batch Size)
            c_init: PUCT 参数 c_init (通常 1.25)
            c_base: PUCT 参数 c_base (通常 19652)
            discount: 折扣因子 gamma (通常 0.99)
            alpha: Dirichlet 噪声参数 (通常 0.3 - 1.0)
        """
        ...

    def reset_env(self, env_idx: int) -> None:
        """重置指定索引的环境状态。"""
        ...

    def set_seed(self, seed: int) -> None:
        """
        设置随机数种子。
        Args:
            seed: 种子值。如果为 -1，则使用随机设备重新生成随机种子。
        """
        ...

    def get_all_counts(self) -> List[int]:
        """获取所有动作的访问计数 (Flattened)。"""
        ...

    def prune_roots(self, actions: np.ndarray) -> None:
        """
        根据采取的动作修剪树（将根节点移动到子节点）。
        
        Args:
            actions: [Batch] int32 数组，表示每个环境采取的动作索引。
        """
        ...

    def search_batch(self, 
                     input_boards: np.ndarray, 
                     turns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        并行执行 MCTS 模拟。
        
        Args:
            input_boards: [Batch, 6, 7] int8 数组。1=先手, -1=后手, 0=空。
            turns: [Batch] int32 数组。1=先手轮次, -1=后手轮次。
            
        Returns:
            Tuple 包含:
            1. leaf_boards: [Batch, 6, 7] int8。需要评估的叶子节点状态。
            2. leaf_values: [Batch] float32。如果是终端状态，包含真实价值；否则无意义。
            3. is_term: [Batch] uint8 (bool)。1 表示该叶子节点是终局，0 表示需要 NN 评估。
        """
        ...

    def backprop_batch(self, 
                       policy_logits: np.ndarray, 
                       values: np.ndarray, 
                       is_term: np.ndarray) -> None:
        """
        反向传播神经网络的预测结果。
        
        Args:
            policy_logits: [Batch, 7] float32。神经网络输出的策略 Logits（或概率）。
            values: [Batch] float32。神经网络输出的价值 [-1, 1]。
            is_term: [Batch] uint8 (bool)。是否忽略 values 使用 search_batch 产生的终端价值。
        """
        ...