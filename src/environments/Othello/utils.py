import torch
import numpy as np
from numba import njit


@njit(fastmath=True)
def board_to_state(board, turn):
    temp = np.zeros((1, 3, board.shape[0], board.shape[1]), dtype=np.float32)
    # Relative perspective:
    #   ch0 = side-to-move stones, ch1 = opponent stones
    temp[:, 0] = board == turn
    temp[:, 1] = board == -turn
    temp[:, 2] = np.ones((board.shape[0], board.shape[1]), dtype=np.float32) * turn
    return temp


@njit(fastmath=True)
def check_full(board):
    return len(np.where(board == 0)[0]) == 0


# D4 群 8 种对称的坐标变换 (r,c) → (r',c')
_SYM_TRANSFORMS = [
    lambda r, c: (r, c),         # 0: 恒等
    lambda r, c: (c, 7 - r),     # 1: 顺时针 90°
    lambda r, c: (7 - r, 7 - c), # 2: 旋转 180°
    lambda r, c: (7 - c, r),     # 3: 顺时针 270°
    lambda r, c: (r, 7 - c),     # 4: 水平翻转
    lambda r, c: (7 - r, c),     # 5: 垂直翻转
    lambda r, c: (c, r),         # 6: 主对角线翻转
    lambda r, c: (7 - c, 7 - r), # 7: 副对角线翻转
]


def _apply_sym_state(state, sym_id):
    """对 (N, 3, 8, 8) 状态张量应用对称变换 sym_id"""
    if sym_id == 0:
        return state
    # 构建坐标映射
    perm = torch.zeros(64, dtype=torch.long)
    transform = _SYM_TRANSFORMS[sym_id]
    for i in range(64):
        nr, nc = transform(i // 8, i % 8)
        perm[i] = nr * 8 + nc
    # 对每个通道 flatten → permute → reshape
    N, C, H, W = state.shape
    flat = state.view(N, C, 64)
    result = torch.zeros_like(flat)
    result[:, :, perm] = flat
    return result.view(N, C, H, W)


def _apply_sym_policy(policy, sym_id):
    """对 (N, 65) 策略张量应用对称变换 sym_id"""
    if sym_id == 0:
        return policy
    perm = torch.zeros(64, dtype=torch.long)
    transform = _SYM_TRANSFORMS[sym_id]
    for i in range(64):
        nr, nc = transform(i // 8, i % 8)
        perm[i] = nr * 8 + nc
    board_part = policy[:, :64]
    pass_part = policy[:, 64:]
    result = torch.zeros_like(board_part)
    result[:, perm] = board_part
    return torch.cat([result, pass_part], dim=1)


def augment(batch):
    """对训练数据应用保持初始局面不变的 4 种对称变换进行数据增强。

    Othello 初始局面只在 Klein 四元群 {恒等, 180°旋转, 主对角线翻转, 副对角线翻转} 下不变。
    90°/270° 旋转和水平/垂直翻转会交换初始黑白子位置，不是合法对称。
    """
    state, prob, winner, steps_to_end, root_wdl = batch

    states_all = [state]
    probs_all = [prob]

    for sym_id in (2, 6, 7):  # 180°旋转, 主对角线翻转, 副对角线翻转
        states_all.append(_apply_sym_state(state, sym_id))
        probs_all.append(_apply_sym_policy(prob, sym_id))

    state = torch.cat(states_all, dim=0)
    prob = torch.cat(probs_all, dim=0)
    winner = winner.repeat(4, 1)
    steps_to_end = steps_to_end.repeat(4, 1)
    root_wdl = root_wdl.repeat(4, 1)

    return state, prob, winner, steps_to_end, root_wdl


def inspect(net, board=None):
    if board is None:
        board = np.zeros((8, 8), dtype=np.float32)
        board[3, 3] = -1; board[3, 4] = 1
        board[4, 3] = 1;  board[4, 4] = -1
    with torch.no_grad():
        state0 = board_to_state(board, 1)
        probs0, wdl0, _ = net.predict(state0)
        probs0 = probs0.flatten()
        value0 = float(wdl0[0, 1] - wdl0[0, 2])  # W - L (to-move)

        # 标准开局第一手 (2,3)
        board[2, 3] = 1
        board[3, 3] = 1  # 翻转
        state1 = board_to_state(board, -1)
        probs1, wdl1, _ = net.predict(state1)
        probs1 = probs1.flatten()
        value1 = float(wdl1[0, 1] - wdl1[0, 2])

    # 显示 top-10 动作
    top_x = np.argsort(probs0)[::-1][:10]
    top_o = np.argsort(probs1)[::-1][:10]
    print("Black (P1) top actions:")
    for idx in top_x:
        label = f"({idx // 8},{idx % 8})" if idx < 64 else "pass"
        print(f"  {label}: {probs0[idx] * 100:.2f}%")
    print(f"State-value Black: {value0:.4f}\n")
    print("White (P2) top actions:")
    for idx in top_o:
        label = f"({idx // 8},{idx % 8})" if idx < 64 else "pass"
        print(f"  {label}: {probs1[idx] * 100:.2f}%")
    print(f"State-value White: {value1:.4f}")
    return probs0, value0, probs1, value1
